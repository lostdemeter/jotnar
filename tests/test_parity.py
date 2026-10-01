"""Parity gate: int chain vs float oracle (same true-amplitude math).

Comparison basis: units linear-light RGB [0,1], peak=1.0, oracle float64 with
edge-replicate blur and identical kernel/sigma/radius/beta. Default
use_alpha=False (Debt 2 verdict: parabola removed; gain-clip + out-clip
protect shadows/highlights). Alpha-ablation parity is gated separately.
Hue-preservation gate proves chroma direction unchanged (phase claim).
Bar >=40dB. Usage: python3 tests/test_parity.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from chain.holo_phi import enhance_image_int, gaussian_kernel
from chain.oracle import enhance_image_float

BAR_DB = 40.0
FAIL = []


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def load_rgb01(path):
    im = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return np.power(im, 2.2).astype(np.float32)


def main():
    rng = np.random.default_rng(1)
    # synthetic: gradient + step + noise (edge-heavy, stresses conv + clip)
    Hh, Ww = 64, 64
    gx, gy = np.meshgrid(np.linspace(0, 1, Ww), np.linspace(0, 1, Hh))
    synth = np.stack([(gx + gy) / 2, gx, np.full_like(gx, 0.4)], axis=-1)
    synth[30:34, :, :] = 0.9
    synth = np.clip(synth + rng.normal(0, 0.01, synth.shape), 0, 1).astype(np.float32)
    for tag, rgb in [("synthetic", synth)]:
        for beta in (0.0, 0.5):
            io, _ = enhance_image_float(rgb, beta=beta)
            ii, _, _ = enhance_image_int(rgb, beta=beta)
            d = psnr(ii, io)
            check(f"parity-{tag}-b{beta}", d >= BAR_DB, f"{d:.2f}dB (bar {BAR_DB})")
    # beta=0 must be near-identity (sanity: no enhancement, no damage)
    io0, _ = enhance_image_float(synth, beta=0.0)
    d0 = psnr(io0, synth)
    check("oracle-identity", d0 >= BAR_DB, f"{d0:.2f}dB")
    # real sample if present (sibling checkout, else in-repo sample)
    _cand_ext = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "rife_reverse", "samples", "f_012.png")
    cand = _cand_ext if os.path.isfile(_cand_ext) else os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "samples", "input_example.png")
    if os.path.exists(cand):
        rgb = load_rgb01(cand)
        io, _ = enhance_image_float(rgb, beta=0.5)
        ii, _, _ = enhance_image_int(rgb, beta=0.5)
        d = psnr(ii, io)
        check("parity-foreman", d >= BAR_DB, f"{d:.2f}dB")
        # Debt-3 gate: hue direction preserved -- chroma ratios before/after
        # agree (only amplitude changes). Basis: luma > 0.02 AND no output
        # channel at the gamut rail (clipped pixels change hue by declaration,
        # not by damage -- output clip [0,1] is integer and audited).
        y = rgb.mean(axis=-1)
        rail = (ii >= 0.99).any(axis=-1) | (ii <= 0.001).any(axis=-1)
        m_ = (y > 0.02) & (~rail)
        r0 = rgb[:, :, 0][m_] / np.maximum(rgb.sum(axis=-1)[m_], 1e-9)
        r1 = ii[:, :, 0][m_] / np.maximum(ii.sum(axis=-1)[m_], 1e-9)
        herr = float(np.abs(r1 - r0).max())
        check("hue-preserved", herr < 0.05, f"maxchroma-drift={herr:.3f}")
    # Debt-2 ablation: deprecated alpha path still matches its oracle
    io_a, _ = enhance_image_float(synth, beta=0.5, use_alpha=True)
    ii_a, _, _ = enhance_image_int(synth, beta=0.5, use_alpha=True)
    check("parity-alpha-ablation", psnr(ii_a, io_a) >= BAR_DB,
          f"{psnr(ii_a, io_a):.2f}dB (deprecated path, still gated)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
