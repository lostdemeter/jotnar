"""demo_router: PNG sequence -> routed enhancement (still/temporal per frame).

The first meta-structure in action: the router selects AMONG modes per frame
while temporal state carries ACROSS frames. Reports per-frame decisions.
Usage:
  python3 demo_router.py indir outdir [--dx 0 --dy 0] [--beta 0.5]
  python3 demo_router.py --selftest   # mixed static+moving, decisions printed
Fidelity: still frames == still-only outputs; routed parity >=40dB/frame.
"""
import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
import phi_core.lattice as S
from chain.holo_phi import apply_gain_int
from chain import router as R
from chain import oracle as O

BAR_DB = 40.0


def srgb_to_linear(im01):
    return np.power(np.clip(im01, 0, 1), 2.2).astype(np.float32)


def linear_to_srgb(lin01):
    return np.power(np.clip(lin01, 0, 1), 1.0 / 2.2)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir", nargs="?")
    ap.add_argument("outdir", nargs="?", default="/tmp/rdemo")
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest or not a.indir:
        N = 48
        rng = np.random.default_rng(23)
        yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                             indexing="ij")
        base = np.where(xx > 0.5, 0.85, 0.15)
        frames, flows = [], []
        for _ in range(3):
            frames.append(np.stack([np.clip(
                base + rng.normal(0, 0.02, base.shape), 0, 1)] * 3, -1))
            flows.append(np.zeros((N, N, 2)))
        for t in range(1, 5):
            bar = np.where(xx > 0.5 - 3 * t / N, 0.85, 0.15)
            frames.append(np.stack([np.clip(
                bar + rng.normal(0, 0.02, bar.shape), 0, 1)] * 3, -1))
            flows.append(np.full((N, N, 2), [0.0, 3.0]))
        ys = [(0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2])
              for f in frames]
        outs, decisions, _ = R.run_sequence(ys, flows, beta=a.beta, blur="iso")
        modes = ["still" if d == "still" else "temporal" for d in decisions]
        refs = O.temporal_frames_float(ys, flows, beta=a.beta, modes=modes)
        ds = []
        for i, (t, r) in enumerate(zip(outs, refs)):
            yv = np.clip(S.decode(t[0], t[1]) * (1 - t[2].astype(np.float64)), 0, 1)
            ds.append(psnr(yv, np.clip(r, 0, 1)))
        os.makedirs("/tmp/rdemo", exist_ok=True)
        ok = all(v >= BAR_DB for v in ds)
        print(f"decisions: {decisions}")
        print(f"parity {[f'{v:.1f}' for v in ds]}dB -> "
              f"{'GO' if ok else 'NO-GO'} (wrote nothing; use indir for files)")
        sys.exit(0 if ok else 1)

    paths = sorted(glob.glob(os.path.join(a.indir, "*.png")))
    assert paths, f"no PNGs in {a.indir}"
    frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
              for p in paths]
    H, W, _ = frames[0].shape
    uni = np.full((H, W, 2), [a.dy, a.dx])
    flows = [uni] * len(frames)
    ys = [(0.2126 * f[:, :, 0] + 0.7152 * f[:, :, 1] + 0.0722 * f[:, :, 2])
          for f in frames]
    outs, decisions, _ = R.run_sequence(ys, flows, beta=a.beta, blur="iso")
    os.makedirs(a.outdir, exist_ok=True)
    for p, t, y in zip(paths, outs, ys):
        rgb_lin = srgb_to_linear(np.asarray(
            Image.open(p).convert("RGB"), dtype=np.float64) / 255.0).astype(np.float32)
        yl = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
              + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)
        enh = apply_gain_int(rgb_lin, yl, t)
        Image.fromarray((np.clip(linear_to_srgb(enh), 0, 1) * 255).astype(np.uint8)
                        ).save(os.path.join(a.outdir, os.path.basename(p)))
    print(f"routed {len(frames)} frames -> {a.outdir}")
    print(f"decisions: {decisions}")


if __name__ == "__main__":
    main()
