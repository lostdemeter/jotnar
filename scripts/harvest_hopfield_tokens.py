"""Token-level Hopfield harvest (hooks only, no training).

Audit-FIRST session: (1) harvest FULL token captures [1,1728,3072]
at the 3 frozen points x 8 frozen frames; (2) full separability audit
on GPU over ALL 13,824 tokens/point (cross-frame pairs); (3) bank
stride-96 subset (18 tokens per frame/point -> 432 patterns fp16);
(4) retrieval battery ONLY if all 3 points clear the frozen bar
(max cross-frame < 0.9 AND mean cross-frame < 0.5) — else dead, no
battery (stated). Eval context mirrors infer_one exactly. Weights
read-only: sha256 of the established 7 files pre AND post.
MUST run under ~/marigold-env/bin/python with a CLEAN env. GPU solo.
Usage: ~/marigold-env/bin/python scripts/harvest_hopfield_tokens.py
Banks samples/marigold/hopfield_tokens.npy + .json
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
STRIDE = 96
TOKS = list(range(0, 1728, STRIDE))
assert len(TOKS) == 18, len(TOKS)
MAXCOS_BAR = 0.9
MEANCOS_BAR = 0.5
BATTERY_F = (0.0, 0.05, 0.2)
BATTERY_SEEDS = (0, 1, 2)
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
        # mirror QwenImageEdit2509Step._match_batch exactly (incl. truncate)
        if t.size(0) == B:
            return t
        if t.size(0) == 1 and B > 1:
            return t.expand(B, *t.shape[1:])
        if t.size(0) > B:
            return t[:B]
        reps = (B + t.size(0) - 1) // t.size(0)
        return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]

    def harvest_one(rgb_path):
        import glob as _g
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

    keys = [f"{s}_fr{fr}" for s, fr in frames]
    # -- determinism: frame 0 twice (full captures) -----------------------
    c1, _ = harvest_one(frame_path(*frames[0]))
    c2, _ = harvest_one(frame_path(*frames[0]))
    det = {p: float((c1[p] - c2[p]).abs().max()) for p in POINTS}
    print("determinism %s x2: %s" % (
        keys[0], " ".join(f"{p}={det[p]:.2e}" for p in POINTS)), flush=True)

    # -- full harvest --------------------------------------------------------
    full = {p: {} for p in POINTS}
    for (s, fr), key in zip(frames, keys):
        out, wall = harvest_one(frame_path(s, fr))
        for p in POINTS:
            assert out[p].shape == (1, 1728, 3072), out[p].shape
            full[p][key] = out[p][0]
        print(f"harvested {key} wall {wall:.0f}s", flush=True)
    for h in handles:
        h.remove()

    # -- full separability audit on GPU (all 13,824 tokens/point) ------------
    seps = {}
    with torch.no_grad():
        for p in POINTS:
            X = torch.stack([full[p][k] for k in keys]).reshape(-1, 3072).cuda()
            Xn = X / X.norm(dim=1, keepdim=True)
            frid = torch.arange(8, device="cuda").repeat_interleave(1728)
            mx, ssum, cnt, wmx, wsum, wcnt = -2.0, 0.0, 0, -2.0, 0.0, 0
            B = 3456
            for a in range(0, 13824, B):
                Ga = Xn[a:a + B] @ Xn.T
                brows = torch.arange(a, min(a + B, 13824))
                cross = frid[brows, None] != frid[None, :]
                within = frid[brows, None] == frid[None, :]
                within[brows - a, brows] = False  # self-pairs out
                mx = max(mx, float(Ga[cross].max()))
                ssum += float(Ga[cross].sum())
                cnt += int(cross.sum())
                wmx = max(wmx, float(Ga[within].max()))
                wsum += float(Ga[within].sum())
                wcnt += int(within.sum())
            seps[p] = {"n": 13824, "cross_max": mx,
                       "cross_mean": ssum / cnt,
                       "within_max": wmx, "within_mean": wsum / wcnt,
                       "clears": bool(mx < MAXCOS_BAR
                                      and ssum / cnt < MEANCOS_BAR)}
            print(f"{p}: cross max={mx:.4f} mean={ssum / cnt:+.4f} "
                  f"| within max={wmx:.4f} mean={wsum / wcnt:+.4f} "
                  f"{'CLEARS' if seps[p]['clears'] else 'fails'}", flush=True)

    # -- bank stride subset ----------------------------------------------------
    rows, toks = [], []
    for p in POINTS:
        for k in keys:
            for t in TOKS:
                rows.append([p, k, t])
                v = full[p][k][t].numpy().astype(np.float64)
                toks.append(v / np.linalg.norm(v))  # normalize-before-store
    arr = np.array(toks)                            # invariant (retrieval-identical)
    outn = os.path.join(ROOT, "samples", "marigold", "hopfield_tokens.npy")
    import numpy as _np
    sub = arr.astype(_np.float16)
    assert bool(_np.isfinite(sub).all()), "fp16 subset must be finite"
    _np.save(outn, sub)
    print(f"banked subset {arr.shape} fp16 "
          f"({os.path.getsize(outn) / 1e6:.2f}MB)", flush=True)

    post = sha7()
    assert pre == post, "WEIGHT FILE MODIFIED (voids round)"
    print("sha post == pre (7 files, nothing written)", flush=True)

    # -- conditional battery ------------------------------------------------------
    battery = {}
    if all(seps[p]["clears"] for p in POINTS):
        import hopfield_retrieval as R
        for pi, p in enumerate(POINTS):
            M64 = arr[pi * 144:(pi + 1) * 144].astype(np.float64)
            Mn = M64 / np.linalg.norm(M64, axis=1, keepdims=True)
            fr_of = [rows[pi * 144 + i][1] for i in range(144)]
            hit = tot = 0
            lv = {}
            for li, f in enumerate(BATTERY_F):
                h = 0
                for i in range(144):
                    seeds = (0,) if f == 0.0 else BATTERY_SEEDS
                    for s in seeds:
                        q = R.corrupt(Mn[i], f, s, li)
                        w, _, _ = R.retrieve(Mn, q)
                        h += fr_of[w] == fr_of[i]
                        tot += 1
                lv[str(f)] = {"hit": h,
                              "tot": 144 * (1 if f == 0.0 else 3)}
                hit += h
            acc = hit / tot
            battery[p] = {"levels": lv, "accuracy": acc, "trials": tot,
                          "chance": 0.125, "bar": 0.25,
                          "verdict": ("alive" if acc >= 0.25 else "dead")}
            print(f"battery {p}: acc={acc:.4f} verdict={battery[p]['verdict']}",
                  flush=True)
    else:
        print("separability fails: NO battery runs (stated)", flush=True)

    rep = {"points": list(POINTS), "frames": keys, "stride": STRIDE,
           "subset": {"count": len(rows), "dtype": "float16",
                      "bytes": os.path.getsize(outn),
                      "capacity": "432/144 << N_exp(3072)=1.3e97 PASS"},
           "index_rows": rows,
           "separability_full": seps, "bar": [MAXCOS_BAR, MEANCOS_BAR],
           "determinism_maxabs": det,
           "sha256": {"pre": pre, "post": post},
           "battery": battery}
    outp = os.path.join(ROOT, "samples", "marigold", "hopfield_tokens.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
