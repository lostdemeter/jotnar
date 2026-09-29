"""Calibration: measure maxima on real frames -> frozen integer scale M.

Doctrine (phi-core): measure offline on representative inputs, cover with
margin, freeze into chain/M.json. Calibration NEVER runs in shipped path.
Usage: python3 calibrate.py [samples_dir]
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phi_core.lattice as S
try:
    from chain.holo_phi import gaussian_kernel
except ModuleNotFoundError:
    from holo_phi import gaussian_kernel

BAR = 40.0


def srgb_to_linear(x):
    return np.power(np.clip(x, 0, 1), 2.2)


def main():
    sdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "samples")
    paths = sorted(glob.glob(os.path.join(sdir, "*.png"))) or sorted(
        glob.glob("/home/thorin/Documents/OpenCode/rife_reverse/samples/*.png"))
    assert paths, "no sample images found"
    k = gaussian_kernel()
    wmax = float(k.max())
    pmax, amax = 0.0, 0.0
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        lin = srgb_to_linear(im)
        y = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
        a = np.sqrt(np.maximum(y, 0))
        pmax = max(pmax, float(a.max()) * wmax)  # max single product per tap
        amax = max(amax, float(np.abs(a).max()), float(y.max()))
    m_acc = S.BIAS + int(round(S.K * math.log(max(pmax, 1e-12)) / S.LN_PHI)) + 512
    m_cov = S.BIAS + int(round(S.K * math.log(max(amax, 1e-12)) / S.LN_PHI)) + 512
    # tensor-domain coverage (#LIB-003): Sobel/8 responses stay small by
    # construction; assert the frozen m_cov covers them (else add a scale).
    tmax = 0.0
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        lin = srgb_to_linear(im)
        y = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]
        a = np.sqrt(np.maximum(y, 0))
        from chain.oracle import _corr_replicate, SOBEL_X, SOBEL_Y
        gx = _corr_replicate(a, SOBEL_X)
        gy = _corr_replicate(a, SOBEL_Y)
        tmax = max(tmax, float(np.abs(gx).max()), float(np.abs(gy).max()),
                   float(((gx * gx + gy * gy)).max()))
    m_ten = S.BIAS + int(round(S.K * math.log(max(tmax, 1e-12)) / S.LN_PHI)) + 512
    assert m_cov >= m_ten, f"tensor domain exceeds m_cov: need {m_ten}, have {m_cov}"
    print(f"tensor check: tmax={tmax:.4f} m_ten={m_ten} <= m_cov={m_cov} OK")
    m = max(m_acc, m_cov)  # single-scale v1: must cover BOTH products and values
    out = {"m": int(m), "m_acc": int(m_acc), "m_cov": int(m_cov),
           "pmax": pmax, "amax": amax, "paths": paths}
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "M.json")
    with open(mp, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"calibrated m={m} (acc={m_acc} cov={m_cov}) pmax={pmax:.4f} amax={amax:.4f}")
    print(f"wrote {mp}")


if __name__ == "__main__":
    main()
