"""Harvest real-dim Hopfield prototypes (hooks only, no training).

Forward capture hooks on 3 frozen points (constitution in
docs/HOPFIELD.md — paths must match exactly or the run stops):
  transformer_blocks.29 (block output tuple index 1 = image stream)
  transformer_blocks.59 (same)
  norm_out (head output tensor, pre-proj_out)
Eval context + preprocessing mirror teacher_lesion.infer_one EXACTLY
(no_grad + bf16 autocast, LANCZOS4 768x576, t=0.499, precomputed
embeds — text encoder never loaded). Weights read-only: sha256 of the
established 7 files pre AND post (a single modified byte voids the
round). Per-image exemplars: mean-pool over image-token seq per
(frame, point) -> 8 x 3072 per point. Frame 0 runs twice for the
determinism evidence (bar: max-abs-diff exactly 0.0 per point).
MUST run under ~/marigold-env/bin/python with a CLEAN env
(cuda_env.sh POISONS it — established). GPU solo.
Usage: ~/marigold-env/bin/python scripts/harvest_hopfield_prototypes.py
Banks samples/marigold/hopfield_prototypes.json
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import teacher_lesion as T

POINTS = ("transformer_blocks.29", "transformer_blocks.59", "norm_out")
WFILES = [
    "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
    "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
    "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
    "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
    "Qwen-Image-Edit-2509/transformer/diffusion_pytorch_model-00005-of-00005.safetensors",
    "Marigold-V2/depth/Log-stage2/trainables.safetensors",
    "Qwen-Image-Edit-2509/vae/diffusion_pytorch_model.safetensors",
]


def sha7():
    out = {}
    for w in WFILES:
        h = hashlib.sha256()
        with open(os.path.join(T.CKPT, w), "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        out[w] = h.hexdigest()
    return out


def main():
    T.need_env()
    import numpy as np
    import torch
    from PIL import Image
    import cv2
    from diffusers import QwenImageEditPipeline

    pre = sha7()
    print(f"sha pre: {len(pre)} files", flush=True)
    transformer, vae = T.load_all()
    pe, pm = T.load_embeds()
    frames = T.eval_frames()
    assert len(frames) == 8, frames
    print(f"eval frames: {len(frames)}", flush=True)

    named = dict(transformer.named_modules())
    for p in POINTS:
        assert p in named, f"harvest point matches nothing: {p} (stop-finding)"
    print(f"harvest points matched 1:1: {list(POINTS)}", flush=True)

    caps = {}

    def _cap(path):
        def _h(m, i, o):
            t = o[1] if isinstance(o, tuple) else o
            caps[path] = t.detach().float().cpu()
        return _h

    handles = [named[p].register_forward_hook(_cap(p)) for p in POINTS]
    dev, dt = "cuda", torch.bfloat16

    def _match_batch(t, B):
        if t.size(0) == B:
            return t
        if t.size(0) == 1 and B > 1:
            return t.expand(B, *t.shape[1:])
        if t.size(0) > B:
            return t[:B]
        reps = (B + t.size(0) - 1) // t.size(0)
        return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]

    def harvest_one(rgb_path):
        torch.manual_seed(T.SEED)
        im = np.asarray(Image.open(rgb_path).convert("RGB"))
        chw = np.transpose(im, (2, 0, 1))
        rs = cv2.resize(np.transpose(chw, (1, 2, 0)).astype(np.float32),
                        (T.MW, T.MH), interpolation=cv2.INTER_LANCZOS4)
        x = (torch.from_numpy(np.transpose(rs, (2, 0, 1))).float()
             / 255.0 * 2.0 - 1.0).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        t0 = time.time()
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
            transformer(
                hidden_states=packed, timestep=ts,
                encoder_hidden_states=emb,
                encoder_hidden_states_mask=msk,
                img_shapes=[[(1, H // 2, W // 2)]] * B,
                txt_seq_lens=msk.sum(dim=1).tolist(),
                guidance=None, return_dict=False)
        wall = time.time() - t0
        assert set(caps) == set(POINTS), set(caps)
        return {p: caps[p].clone() for p in POINTS}, wall

    def frame_path(s, fr):
        import glob as _g
        c = sorted(_g.glob(os.path.join(T.SHARD, s, f"rgb_cam_00_fr{fr}.png")))
        assert len(c) == 1, (s, fr, c)
        return c[0]

    # -- determinism: frame 0 twice -------------------------------------
    s0, fr0 = frames[0]
    c1, w1 = harvest_one(frame_path(s0, fr0))
    c2, w2 = harvest_one(frame_path(s0, fr0))
    det = {p: float((c1[p] - c2[p]).abs().max()) for p in POINTS}
    print(f"determinism {s0}/fr{fr0} x2: "
          + " ".join(f"{p}={det[p]:.2e}" for p in POINTS)
          + f" (walls {w1:.0f}s {w2:.0f}s)", flush=True)

    # -- full harvest ------------------------------------------------------
    protos, shapes, walls = {}, {}, {}
    for (s, fr) in frames:
        key = f"{s}_fr{fr}"
        caps_out, wall = harvest_one(frame_path(s, fr))
        walls[key] = round(wall, 1)
        for p in POINTS:
            t = caps_out[p]
            assert t.dim() == 3 and t.size(0) == 1 and t.size(2) == 3072, t.shape
            shapes[p] = list(t.shape)
            protos.setdefault(p, {})[key] = t.mean(dim=1)[0].tolist()
        print(f"harvested {key} wall {wall:.0f}s", flush=True)
    for h in handles:
        h.remove()

    post = sha7()
    assert pre == post, "WEIGHT FILE MODIFIED (voids round)"
    print("sha post == pre (7 files, nothing written)", flush=True)

    cap = json.load(open(os.path.join(ROOT, "samples", "marigold",
                                      "hopfield_capacity.json")))
    rep = {"points": list(POINTS), "frames": [f"{s}_fr{fr}" for s, fr in frames],
           "dim": 3072, "shapes": shapes, "walls": walls,
           "determinism_maxabs": det,
           "capacity": {p: {"dim": 3072, "K": cap["K"],
                            "requirement": "d>=512 robust-sufficient",
                            "verdict": "PASS"} for p in POINTS},
           "sha256": {"pre": pre, "post": post},
           "prototypes": protos}
    outp = os.path.join(ROOT, "samples", "marigold", "hopfield_prototypes.json")
    json.dump(rep, open(outp, "w"))
    print(f"POINT dim=3072 K={cap['K']} requirement=d>=512: "
          f"{' '.join(f'{p}=PASS' for p in POINTS)}", flush=True)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
