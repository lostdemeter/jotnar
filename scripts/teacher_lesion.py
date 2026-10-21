"""Teacher lesion Phase A: block sweep on Marigold-V2 DiT (read-only use).

Loads the teacher EXACTLY like their component path (4-bit NF4 DiT +
LoRA r128 trainables + bf16 VAE + cached prompt embeds — never the
text encoder) and mirrors their inference nodes
(QwenImageEncode/2509Step/Decode/SelectChannel) with lesion hooks.
MUST run under ~/marigold-env/bin/python with a CLEAN env
(cuda_env.sh POISONS it — proven: subprocess fails under sourced
env, passes clean). Sibling tree untouched (weight/prompt FILES
read; zero sibling imports). Weights read-only (sha256 banked
pre/post; a single modified byte voids the round).
Usage (GPU solo):
  teacher_lesion.py --mode baseline [--tag teacher-base-A]
  teacher_lesion.py --mode lesion --targets transformer_blocks.3,transformer_blocks.56 [--tag ...]
  teacher_lesion.py --mode parity --cli-dir DIR (vs sibling-CLI .npy, model-res)
Eval set (frozen): first 2 available frames of ai_018_001,
ai_021_001, ai_022_001, ai_023_005 (8 frames, deterministic).
Judge: lstsq-align + AbsRel per stratum (masks & valid), GT meters.
"""
import argparse
import fnmatch
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

SIB = "/home/balrog/Documents/OpenCode"
CKPT = "/home/balrog/Documents/OpenCode/marigold-v2/assets/checkpoints"
QWEN = os.path.join(CKPT, "Qwen-Image-Edit-2509")
TRAINABLES = os.path.join(CKPT, "Marigold-V2/depth/Log-stage2/trainables.safetensors")
EMBDIR = os.path.join(CKPT, "Marigold-V2/qwen_text_embeddings")
PREFIX = "qwen_edit_2509_qwen_depth_realimg512"
SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")
EVAL_SCENES = ["ai_018_001", "ai_021_001", "ai_022_001", "ai_023_005"]
MW, MH = 768, 576
SEED = 2025

LORA_TARGETS = ["img_in", "txt_in", "to_q", "to_k", "to_v", "to_out.0",
                "attn.add_k_proj", "attn.add_v_proj", "attn.add_q_proj",
                "attn.to_add_out", "norm.linear", "a_to_out", "b_to_out",
                "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
                "txt_mlp.net.2", "ff_a.0", "ff_a.2", "ff_b.0", "ff_b.2",
                "norm_out.linear"]


def need_env():
    try:
        import diffusers  # noqa
        import peft  # noqa
        import bitsandbytes  # noqa
        import cv2  # noqa
    except ImportError as e:
        raise SystemExit(f"teacher harness needs marigold-env python: {e}")


def eval_frames():
    out = []
    for s in EVAL_SCENES:
        ids = []
        for p in sorted(glob.glob(os.path.join(SHARD, s, "rgb_cam_00_fr*.png"))):
            m = re.search(r"fr(\d+)\.png$", p)
            fr = m.group(1)
            if os.path.isfile(os.path.join(
                    SHARD, s, f"depth_plane_cam_00_fr{fr}.png")):
                ids.append(fr)
                if len(ids) == 2:
                    break
        assert len(ids) == 2, f"scene {s}: need 2 frames, got {ids}"
        out += [(s, fr) for fr in ids]
    return out


def load_all():
    import torch
    from diffusers import (AutoencoderKLQwenImage,
                           QwenImageTransformer2DModel)
    from diffusers import BitsAndBytesConfig as DBC
    from peft import LoraConfig, prepare_model_for_kbit_training
    dev = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("teacher harness needs CUDA (fail loud, no CPU fallback)")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        QWEN, subfolder="transformer", local_files_only=True,
        torch_dtype=torch.bfloat16,
        quantization_config=DBC(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                llm_int8_skip_modules=[
                                    "transformer_blocks.0.img_mod"]))
    prepare_model_for_kbit_training(transformer,
                                    use_gradient_checkpointing=False)
    transformer.enable_gradient_checkpointing()
    # dequantize proj_out (mirror their helper: fnmatch + bnb dequant)
    from bitsandbytes.functional import dequantize_4bit
    import torch.nn as nn
    named = dict(transformer.named_modules())
    for name, mod in list(named.items()):
        if not any(fnmatch.fnmatch(name, "*" + p + "*" )
                   for p in ("proj_out",)):
            continue
        if hasattr(mod, "weight") and hasattr(
                getattr(mod, "weight", None), "quant_state"):
            w = mod.weight
            dq = dequantize_4bit(w.data, w.quant_state).to(torch.bfloat16)
            new = nn.Linear(dq.shape[1], dq.shape[0], bias=True,
                            dtype=torch.bfloat16)
            new.weight.data.copy_(dq)
            if mod.bias is not None:
                new.bias.data.copy_(mod.bias.data.to(torch.bfloat16))
            new.requires_grad_(False)
            parent, _, leaf = name.rpartition(".")
            setattr(transformer.get_submodule(parent), leaf, new)
    lconf = LoraConfig(r=128, lora_alpha=128, lora_dropout=0.0,
                       init_lora_weights="gaussian",
                       target_modules=[t for t in LORA_TARGETS
                                       if "proj_out" not in t])
    transformer.add_adapter(lconf)
    for n, p in transformer.named_parameters():
        if "lora_" in n:
            p.data = p.data.to(torch.bfloat16)
    from safetensors.torch import load_file
    sd = load_file(TRAINABLES, device="cpu")
    tdiff, tvae = {}, {}
    for k, v in sd.items():
        if k.startswith("Diffuser."):
            tdiff[k[len("Diffuser."):]] = v
        elif k.startswith("VAE."):
            tvae[k[len("VAE."):]] = v
    res = transformer.load_state_dict(tdiff, strict=False)
    print(f"trainables->DiT: missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}", flush=True)
    proj_unexp = [k for k in res.unexpected_keys if "proj_out" in k]
    bad = [k for k in res.unexpected_keys if "proj_out" not in k]
    assert not bad, f"unexpected trainable keys (typo-guard): {bad[:5]}"
    print(f"proj_out keys skipped as designed: {len(proj_unexp)}", flush=True)
    transformer.requires_grad_(False)
    vae = AutoencoderKLQwenImage.from_pretrained(
        QWEN, subfolder="vae", local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev)
    vres = vae.load_state_dict(tvae, strict=False)
    print(f"trainables->VAE: missing={len(vres.missing_keys)} "
          f"unexpected={len(vres.unexpected_keys)}", flush=True)
    assert not vres.unexpected_keys, vres.unexpected_keys
    vae.requires_grad_(False)
    vae.eval()
    transformer.eval()
    transformer.to(dev)
    print(f"VRAM: alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.1f}GB", flush=True)
    return transformer, vae


def load_embeds():
    import torch

    def _full(p):
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(p, map_location="cpu")

    pe = _full(os.path.join(EMBDIR, f"{PREFIX}_prompt_embeds.pt")).contiguous()
    pm = _full(os.path.join(EMBDIR, f"{PREFIX}_prompt_mask.pt")).contiguous()
    if pm.dtype != torch.bool:
        pm = pm > 0
    return pe, pm


def infer_one(transformer, vae, pe, pm, rgb_path, seed, hooks=()):
    """One frame -> (pred_768x576 float64, per-stage seconds). Hooks:
    [(module_path_or_fnmatch, )] zero outputs (fire-counted)."""
    import time as _t
    import numpy as np
    import torch
    from PIL import Image
    import cv2
    from diffusers import QwenImageEditPipeline
    torch.manual_seed(seed)
    im = np.asarray(Image.open(rgb_path).convert("RGB"))
    chw = np.transpose(im, (2, 0, 1))
    rs = cv2.resize(np.transpose(chw, (1, 2, 0)).astype(np.float32),
                    (MW, MH), interpolation=cv2.INTER_LANCZOS4)
    x = (torch.from_numpy(np.transpose(rs, (2, 0, 1))).float()
         / 255.0 * 2.0 - 1.0).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    dev = "cuda"
    dt = torch.bfloat16
    handles, fires = [], {}

    def _zero(path):
        def _h(m, i, o):
            fires[path] = fires.get(path, 0) + 1
            if isinstance(o, tuple):
                return type(o)(torch.zeros_like(t) for t in o)
            return torch.zeros_like(o)
        return _h

    named = dict(transformer.named_modules())
    named.update({"VAE." + n: m for n, m in vae.named_modules()})
    handles = []
    for pat in hooks:
        hit = [n for n in named if n == pat or fnmatch.fnmatch(n, pat)]
        assert hit, f"lesion target matches nothing: {pat}"
        for n in hit:
            handles.append(named[n].register_forward_hook(_zero(n)))
    def _match_batch(t, B):
        # mirror QwenImageEdit2509Step._match_batch exactly (incl. truncate)
        if t.size(0) == B:
            return t
        if t.size(0) == 1 and B > 1:
            return t.expand(B, *t.shape[1:])
        if t.size(0) > B:
            return t[:B]
        reps = (B + t.size(0) - 1) // t.size(0)
        return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]

    t0 = _t.time()
    try:
        # eval context mirrors their validation EXACTLY (validate_steps:
        # no_grad + autocast bf16 — without it, prepare()'s fp32-kept
        # modules (e.g. skipped blocks.0.img_mod) hard-error on bf16
        # inputs; autocast runs them in bf16 like production).
        import torch.amp as _amp
        with torch.no_grad(), _amp.autocast("cuda", dtype=torch.bfloat16):
            enc = vae.encode(x.unsqueeze(2)).latent_dist.sample()
            m = torch.tensor(vae.config.latents_mean, device=dev, dtype=dt)
            s = torch.tensor(vae.config.latents_std, device=dev, dtype=dt)
            lat = (enc - m.view(1, -1, 1, 1, 1)) * (1.0 / s).view(1, -1, 1, 1, 1)
            lat4 = lat[:, :, 0]
            B, C, H, W = lat4.shape
            packed = QwenImageEditPipeline._pack_latents(
                lat4, batch_size=B, num_channels_latents=C,
                height=H, width=W).to(dt)
            ts = torch.full((B,), 0.499, device=dev, dtype=dt)
            emb = _match_batch(pe, B).to(device=dev, dtype=dt)
            msk = _match_batch(pm, B).to(device=dev, dtype=torch.bool)
            pred = transformer(
                hidden_states=packed, timestep=ts,
                encoder_hidden_states=emb,
                encoder_hidden_states_mask=msk,
                img_shapes=[[(1, H // 2, W // 2)]] * B,
                txt_seq_lens=msk.sum(dim=1).tolist(),
                guidance=None, return_dict=False)[0]
            unp = QwenImageEditPipeline._unpack_latents(
                pred, height=H * 8, width=W * 8, vae_scale_factor=8)
            lat2 = (lat.to(vae.dtype) - unp.to(vae.dtype))
            lat2 = lat2 / (1.0 / s.to(vae.dtype)).view(1, -1, 1, 1, 1) \
                + m.to(vae.dtype).view(1, -1, 1, 1, 1)
            dec = vae.decode(lat2).sample[:, :, 0]
            ch = dec.mean(dim=1, keepdim=True).float().cpu().numpy()[0, 0]
    finally:
        for h in handles:
            h.remove()
    wall = _t.time() - t0
    return ch, wall, dict(fires)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "lesion", "parity",
                                       "pilot"], required=True)
    ap.add_argument("--tag", default="teacher-base-A")
    ap.add_argument("--targets", default="",
                    help="comma-separated module paths/patterns (lesion)")
    ap.add_argument("--cli-dir", default="",
                    help="sibling-CLI .npy dir (parity mode)")
    ap.add_argument("--out", default="",
                    help="bank JSON path (default per mode under samples/)")
    args = ap.parse_args()
    need = ["diffusers", "peft", "bitsandbytes", "cv2"]
    try:
        import diffusers  # noqa
    except ImportError as e:
        raise SystemExit(f"run under ~/marigold-env/bin/python: {e}")
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    frames = eval_frames()
    print(f"eval frames: {len(frames)}", flush=True)

    if args.mode == "parity":
        assert args.cli_dir, "--cli-dir required"
        import glob as _g
        transformer, vae = load_all()
        pe, pm = load_embeds()
        dbs = []
        for (s, fr) in frames[:2]:
            cand = sorted(_g.glob(os.path.join(
                args.cli_dir, f"{s}_fr{fr}.npy")))
            assert len(cand) == 1, f"CLI file for {s}/{fr}: {cand}"
            ref = np.load(cand[0]).astype(np.float64)
            p = os.path.join(SHARD, s, f"rgb_cam_00_fr{fr}.png")
            pred, wall, _ = infer_one(transformer, vae, pe, pm, p,
                                      SEED)
            mine = pred.astype(np.float64)
            assert ref.shape == mine.shape, (ref.shape, mine.shape)
            mse = float(np.mean((ref - mine) ** 2))
            db = 10 * np.log10(1.0 / max(mse, 1e-300))
            dbs.append(db)
            print(f"parity {s}/fr{fr}: {db:.1f}dB shape {ref.shape} "
                  f"wall {wall:.0f}s", flush=True)
        print(f"PARITY {'OK' if min(dbs) >= 40 else 'FAIL'} "
              f"min {min(dbs):.1f}dB (bar 40dB)", flush=True)
        return

    transformer, vae = load_all()
    pe, pm = load_embeds()

    def judge(pred576, s, fr):
        base = os.path.join(SHARD, s)
        depth = np.asarray(Image.open(os.path.join(
            base, f"depth_plane_cam_00_fr{fr}.png"))).astype(np.float64) / 1000.0
        rgbn = np.asarray(Image.open(os.path.join(
            base, f"rgb_cam_00_fr{fr}.png")).convert("RGB")).astype(np.float64) / 255.0
        up = np.asarray(Image.fromarray(pred576.astype(np.float32)).resize(
            (depth.shape[1], depth.shape[0]), Image.BILINEAR),
            dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.0)
        # GT percentile-affine normalize (their NormalizeDepthPercentileAffine:
        # p2/p98 on valid -> [-1,1]); plain AbsRel, NO alignment (their metric).
        dv = depth[valid]
        lo, hi = np.percentile(dv, [2.0, 98.0])
        denom = max(hi - lo, 1e-12)
        rel = np.clip(2.0 * (depth - lo) / denom - 1.0, -1.0, 1.0)
        masks = H.strata_masks(depth, rgbn)
        per = {}
        for k in H.STRATA:
            m = masks[k] & valid
            if int(m.sum()) == 0:
                per[k] = None
                continue
            per[k] = float(np.abs(up[m] - rel[m]).mean())
        return per

    def run_eval(hooks=()):
        per_frame, walls = {}, []
        for i, (s, fr) in enumerate(frames):
            p = os.path.join(SHARD, s, f"rgb_cam_00_fr{fr}.png")
            pred, wall, fires = infer_one(transformer, vae, pe, pm, p,
                                          SEED + i, hooks)
            walls.append(wall)
            per_frame[f"{s}_fr{fr}"] = judge(pred, s, fr)
            print(f"eval {s}/fr{fr}: wall {wall:.0f}s "
                  f"meanAbsRel "
                  f"{np.mean([v for v in per_frame[f'{s}_fr{fr}'].values() if v is not None]):.4f}",
                  flush=True)
        return per_frame, walls, fires

    if args.mode == "baseline":
        per_frame, walls, _ = run_eval()
        rep = {"tag": args.tag, "mode": "baseline", "seed": SEED,
               "frames": {n: {"strata": v} for n, v in per_frame.items()},
               "walls": walls}
        outp = args.out or os.path.join(
            ROOT, "samples", "marigold", "teacher_baseline_A.json")
        json.dump(rep, open(outp, "w"), indent=1)
        print(f"wrote {outp}", flush=True)
        return

    if args.mode == "pilot":
        targets = [t for t in args.targets.split(",") if t]
        assert targets, "--targets required (2 pilot blocks)"
        for t in targets:
            t0 = time.time()
            per_frame, walls, fires = run_eval(hooks=[t])
            print(f"pilot {t}: per-eval wall {np.mean(walls):.0f}s "
                  f"(frames {len(walls)}) fires={fires}", flush=True)
        print(f"pilot total {time.time() - t0:.0f}s", flush=True)
        return

    targets = [t for t in args.targets.split(",") if t]
    outp = args.out or os.path.join(
        ROOT, "samples", "marigold", "teacher_lesion_A.json")
    rows = []
    if os.path.isfile(outp):
        rows = json.load(open(outp)).get("rows", [])
        print(f"resuming {outp} ({len(rows)} rows banked)", flush=True)
    done = {r["target"] for r in rows}
    for t in targets:
        if t in done:
            print(f"skip {t} (banked)", flush=True)
            continue
        per_frame, walls, fires = run_eval(hooks=[t])
        row = {"target": t, "fires": fires, "walls": walls,
               "frames": per_frame}
        rows.append(row)
        json.dump({"rows": rows}, open(outp, "w"), indent=1)
        print(f"banked {t} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
