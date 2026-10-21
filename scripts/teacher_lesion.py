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


def infer_one(transformer, vae, pe, pm, rgb_path, seed, hooks=(),
              rgb_array=None):
    """One frame -> (pred_768x576 float32, wall_s, fires). Hooks:
    [(module_path_or_fnmatch, )] zero outputs (fire-counted).
    rgb_array (H,W,3 float in [0,1]) bypasses PNG read when given."""
    import time as _t
    import numpy as np
    import torch
    from PIL import Image
    import cv2
    from diffusers import QwenImageEditPipeline
    torch.manual_seed(seed)
    if rgb_array is not None:
        im = (np.clip(np.asarray(rgb_array, dtype=np.float64), 0, 1)
              * 255.0 + 0.5).astype(np.uint8)
    else:
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
        wild = any(c in pat for c in "*?[")
        hit = [n for n in named
               if n == pat or n == "VAE." + pat
               or (wild and fnmatch.fnmatch(n, pat))]
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
    ap.add_argument("--subset-file", default="",
                    help="JSON {rowname: [paths]} for subset-curve rows "
                    "(one row per group, hooks = group paths)")
    ap.add_argument("--dry-run", action="store_true",
                    help="load + verify all targets match + VRAM, no eval")
    ap.add_argument("--cli-dir", default="",
                    help="sibling-CLI .npy dir (parity mode)")
    ap.add_argument("--out", default="",
                    help="bank JSON path (default per mode under samples/)")
    ap.add_argument("--mapstats", action="store_true",
                    help="bank per-frame map digests+stats (sha256 of "
                    "model-res float32 bytes; mean/std; corr vs GT and "
                    "vs unlesioned ref at native res; null corr when "
                    "either side has zero variance)")
    ap.add_argument("--save-maps", default="",
                    help="materialize lesioned-teacher maps: model-res "
                    "float32 .npy per (lesion-target-set, fixture frame) "
                    "into DIR (Phase-29 targets; fixture RGB in, "
                    "BILINEAR-upsampled native maps banked)")
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

    if args.save_maps:
        import train_student as T
        outdir = args.save_maps
        os.makedirs(outdir, exist_ok=True)
        basins = json.load(open(os.path.join(
            ROOT, "samples", "marigold",
            "teacher_basins_H.json")))["basins"]
        order = ["A", "B", "C", "D", "E"]
        hrows = {r["target"]: r for r in json.load(open(
            os.path.join(ROOT, "samples", "marigold",
                         "teacher_lesion_H.json")))["rows"]}
        man = json.load(open(os.path.join(
            FIXDIR, "teacher_manifest.json")))
        names = ["church", "dogs", "squirrel", "train"]
        manifest = {}
        for bi, b in enumerate(order):
            tgt = basins[b][0]
            idx = hrows[tgt]["subset"]
            hooks = [f"transformer_blocks.{i}" for i in idx]
            manifest[tgt] = {"basin": b, "subset": idx, "files": {}}
            for j, n in enumerate(names):
                row = next(r for r in man["fixtures"] if r["name"] == n)
                Hh, Ww = row["shape"]
                _, im3, _ = T.load_frame(n)
                pred, wall, fires = infer_one(
                    transformer, vae, pe, pm, None, SEED + j,
                    hooks, rgb_array=im3)
                assert sum(fires.values()) == len(hooks), fires
                up = np.asarray(Image.fromarray(
                    pred.astype(np.float32)).resize(
                        (Ww, Hh), Image.BILINEAR), dtype=np.float32)
                fn = f"lm-{b}_{n}.npy"
                np.save(os.path.join(outdir, fn),
                        np.ascontiguousarray(up))
                manifest[tgt]["files"][n] = fn
                print(f"materialize {tgt}/{n}: wall {wall:.0f}s -> {fn}",
                      flush=True)
        json.dump(manifest, open(os.path.join(outdir, "manifest.json"),
                                 "w"), indent=1)
        print(f"materialized {outdir}", flush=True)
        return

    if args.dry_run:
        targets = [t for t in args.targets.split(",") if t]
        named = dict(transformer.named_modules())
        named.update({"VAE." + n: m for n, m in vae.named_modules()})
        for pat in targets:
            wild = any(c in pat for c in "*?[")
            hit = [n for n in named
                   if n == pat or n == "VAE." + pat
                   or (wild and fnmatch.fnmatch(n, pat))]
            print(f"dry {pat}: {len(hit)} modules", flush=True)
            assert hit, f"lesion target matches nothing: {pat}"
        print(f"DRY OK ({len(targets)} targets)", flush=True)
        return

    def judge(pred576, s, fr, refmap=None, do_stats=False):
        import hashlib
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
        ms = None
        if do_stats:
            a = np.ascontiguousarray(pred576, dtype=np.float32)
            mu, sd = float(a.mean()), float(a.std())

            def _corr(x, y):
                xs = x.reshape(-1).astype(np.float64)
                ys = y.reshape(-1).astype(np.float64)
                if float(xs.std()) == 0.0 or float(ys.std()) == 0.0:
                    return None
                return float(((xs - xs.mean()) * (ys - ys.mean())).mean()
                             / (xs.std() * ys.std()))

            ms = {"sha": hashlib.sha256(a.tobytes()).hexdigest(),
                  "mean": mu, "std": sd,
                  "corr_gt": _corr(up[valid], depth[valid]),
                  "corr_ref": (_corr(
                      np.ascontiguousarray(pred576, dtype=np.float32),
                      refmap) if refmap is not None else None)}
        return per, ms

    def run_eval(hooks=(), refmaps=None, do_stats=False):
        from collections import Counter
        per_frame, walls, fall, mstats = {}, [], Counter(), {}
        for i, (s, fr) in enumerate(frames):
            p = os.path.join(SHARD, s, f"rgb_cam_00_fr{fr}.png")
            pred, wall, fires = infer_one(transformer, vae, pe, pm, p,
                                          SEED + i, hooks)
            walls.append(wall)
            fall.update(fires)
            per, ms = judge(
                pred, s, fr,
                refmap=(refmaps.get(f"{s}_fr{fr}")
                        if refmaps else None),
                do_stats=do_stats)
            per_frame[f"{s}_fr{fr}"] = per
            if ms is not None:
                mstats[f"{s}_fr{fr}"] = ms
            print(f"eval {s}/fr{fr}: wall {wall:.0f}s "
                  f"meanAbsRel "
                  f"{np.mean([v for v in per_frame[f'{s}_fr{fr}'].values() if v is not None]):.4f}",
                  flush=True)
        return per_frame, walls, dict(fall), mstats

    def refmaps_all():
        """Unlesioned model-res maps for corr_ref (8 evals, RAM-resident)."""
        import torch
        out = {}
        for i, (s, fr) in enumerate(frames):
            p = os.path.join(SHARD, s, f"rgb_cam_00_fr{fr}.png")
            pred, _, _ = infer_one(transformer, vae, pe, pm, p,
                                   SEED + i, ())
            out[f"{s}_fr{fr}"] = np.ascontiguousarray(
                pred, dtype=np.float32)
        return out

    if args.mode == "baseline":
        rm = refmaps_all() if args.mapstats else None
        per_frame, walls, _, mstats = run_eval(
            refmaps=rm, do_stats=args.mapstats)
        rep = {"tag": args.tag, "mode": "baseline", "seed": SEED,
               "frames": {n: {"strata": v} for n, v in per_frame.items()},
               "walls": walls}
        if args.mapstats:
            rep["mapstats"] = mstats
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
            per_frame, walls, fires, _ = run_eval(hooks=[t])
            print(f"pilot {t}: per-eval wall {np.mean(walls):.0f}s "
                  f"(frames {len(walls)}) fires={fires}", flush=True)
        print(f"pilot total {time.time() - t0:.0f}s", flush=True)
        return

    targets = [t for t in args.targets.split(",") if t]
    groups = [(t, [t], None) for t in targets]
    if args.subset_file:
        subs = json.load(open(args.subset_file))
        groups = [(f"subset_{k}",
                   [f"transformer_blocks.{i}" for i in v], list(v))
                  for k, v in sorted(subs.items())]
    outp = args.out or os.path.join(
        ROOT, "samples", "marigold", "teacher_lesion_A.json")
    rows = []
    if os.path.isfile(outp):
        rows = json.load(open(outp)).get("rows", [])
        print(f"resuming {outp} ({len(rows)} rows banked)", flush=True)
    done = {r["target"] for r in rows}
    rm = None
    if args.mapstats and any(n not in done for n, _, _ in groups):
        print("mapstats: computing unlesioned refmaps once", flush=True)
        rm = refmaps_all()
    for name, hooks, idx in groups:
        if name in done:
            print(f"skip {name} (banked)", flush=True)
            continue
        per_frame, walls, fires, mstats = run_eval(
            hooks=hooks, refmaps=rm, do_stats=args.mapstats)
        row = {"target": name, "fires": fires, "walls": walls,
               "frames": per_frame}
        if idx is not None:
            row["subset"] = idx
        if args.mapstats:
            row["mapstats"] = mstats
        rows.append(row)
        json.dump({"rows": rows}, open(outp, "w"), indent=1)
        print(f"banked {name} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
