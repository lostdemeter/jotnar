"""demo_temporal: PNG sequence -> enhanced PNGs via temporal IIR chain.

Backend: numpy integer emu. State (detail triples) carried across frames;
flow is uniform (dx,dy) per pair in v1 (synthetic/exact) -- per-pixel RIFE
flows land with the Phase-2 producer (same as motion).
Usage:
  python3 demo_temporal.py indir outdir [--dx 0 --dy 0] [--beta 0.5]
  python3 demo_temporal.py --selftest   # translating bar, prints parity/flow
Fidelity: sequence parity vs float oracle >=40dB per frame (proves wiring).
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain.holo_phi import apply_gain_int
from chain import temporal as T
from chain import oracle as O

BAR_DB = 40.0


def srgb_to_linear(im01):
    return np.power(np.clip(im01, 0, 1), 2.2).astype(np.float32)


def linear_to_srgb(lin01):
    return np.power(np.clip(lin01, 0, 1), 1.0 / 2.2)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def run_frames(frames01, flows, beta):
    """frames01: list of RGB [0,1]; flows[i]: (H,W,2) or None (flows[0]
    unused). Returns (outs_srgb, per-frame Y parity dB vs oracle, infos)."""
    import phi_core.lattice as S
    st = T.new_state()
    outs, ds, infos = [], [], []
    ys = []
    for i, (rgb01, fl) in enumerate(zip(frames01, flows)):
        rgb_lin = srgb_to_linear(rgb01).astype(np.float32)
        y = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
             + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)
        ys.append(y)
        yt, st, info = T.step(y, fl, st, blur="iso", beta=beta)
        yv = np.clip(S.decode(yt[0], yt[1]) * (1 - yt[2].astype(np.float64)), 0, 1)
        yo = O.temporal_frames_float(ys, flows[:i + 1], beta=beta)[-1]
        ds.append(psnr(yv, np.clip(yo, 0, 1)))
        rgb_enh = apply_gain_int(rgb_lin, y, yt)
        outs.append(np.clip(linear_to_srgb(rgb_enh) * 255.0, 0, 255).astype(np.uint8))
        infos.append(info)
    return outs, ds, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", nargs="?")
    ap.add_argument("outdir", nargs="?", default="/tmp/tdemo")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest or not a.indir:
        N = 48
        frames, flows = [], [None]
        for t in range(4):
            yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                                 indexing="ij")
            g = np.where(xx > 0.5 - 2 * t / N, 0.85, 0.15)
            frames.append(np.stack([g] * 3, -1))
            if t > 0:
                flows.append(np.full((N, N, 2), [0.0, 2.0]))
        outs, ds, _ = run_frames(frames, flows, a.beta)
        os.makedirs("/tmp/tdemo", exist_ok=True)
        for i, o in enumerate(outs):
            Image.fromarray(o).save(f"/tmp/tdemo/self{i}.png")
        ok = all(v >= BAR_DB for v in ds)
        print(f"selftest: parity {[f'{v:.1f}' for v in ds]}dB -> "
              f"{'GO' if ok else 'NO-GO'}; wrote /tmp/tdemo/self*.png")
        sys.exit(0 if ok else 1)

    paths = sorted(glob.glob(os.path.join(a.indir, "*.png")))
    assert paths, f"no PNGs in {a.indir}"
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
              for p in paths]
    H, W, _ = frames[0].shape
    uni = np.full((H, W, 2), [a.dy, a.dx])
    flows = [None] + [uni] * (len(frames) - 1)
    outs, ds, _ = run_frames(frames, flows, a.beta)
    os.makedirs(a.outdir, exist_ok=True)
    for p, o in zip(paths, outs):
        Image.fromarray(o).save(os.path.join(a.outdir, os.path.basename(p)))
    ok = all(v >= BAR_DB for v in ds)
    print(f"enhanced {len(frames)} frames dx={a.dx} dy={a.dy} -> {a.outdir}")
    print(f"parity {[f'{v:.1f}' for v in ds]}dB -> {'GO' if ok else 'NO-GO'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
