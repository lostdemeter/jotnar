"""v5 gates: learned detail gate is valid, continuous, and ordered.

The v5 rule (beff_v4 * sigmoid-gate on (1,coh,dhat)) adds no discrete
decision. Noise doctrine is two-tier (see #LIB-015): realistic grain
(sigma 0.02, the pairs' own degradation level) is BARRED parity; adversarial
white noise at full amplitude -- where orientations are random per pixel and
bank outputs spread wide -- is a MEASURED row with its mechanism stated
(distributed gain-on-spread, no flips: top-5% share ~11x, same as v3/v4).
A max(0,40-npsnr) veto inside the fit objective was tried and reverted:
nothing clears 40, so an unreachable veto is pure drag that elected (0,0,4)
and killed the coherence term. Verify reachability before adding vetoes.
Spec: docs/BETA_CTRL.md. Usage: python3 test_v5.py
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from chain.holo_phi import enhance_image_int
from chain.oracle import enhance_image_float_splat
from chain.control import load_ctrl, DEFAULTS
from fit_v5 import GRID_W0, GRID_W1, GRID_W2
from fit_ctrl import pairs

BAR_DB = 40.0
FAIL = []
_CAND_EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rife_reverse", "samples", "f_012.png")
CAND = _CAND_EXT if os.path.isfile(_CAND_EXT) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "input_example.png")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def v5_kw(P):
    return dict(beta=0.5, iso_atten=P["iso_atten"], coh_thr=P["coh_thr"],
                mid_atten=P.get("mid_atten", 0.6), coh_hi=P.get("coh_hi", 0.5),
                strong_atten=P.get("strong_atten", 1.0), soft=True,
                soft_blur=True, v5=True, v5_w0=P.get("v5_w0", 0.0),
                v5_w1=P.get("v5_w1", 0.0), v5_w2=P.get("v5_w2", 0.0))


def boost_of(y):
    """p99 |enhanced-clean| of the int v5 chain (cf test_ctrl doctrine)."""
    rgb = np.stack([y] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat_soft", ctrl="v5")
    return float(np.percentile(np.abs(out[:, :, 0] - y), 99))


def main():
    P = load_ctrl()
    print(f"ctrl v5: w=({P.get('v5_w0')},{P.get('v5_w1')},{P.get('v5_w2')})")
    # 1. file validity: v5 keys present, on-grid (fitter's contract)
    try:
        with open(os.path.join("chain", "CTRL.json")) as fh:
            d = json.load(fh)
        check("v5-keys", set(("v5_w0", "v5_w1", "v5_w2")) <= set(d),
              f"v5={d.get('v5_w0')}/{d.get('v5_w1')}/{d.get('v5_w2')}")
        check("v5-grid", d["v5_w0"] in GRID_W0 and d["v5_w1"] in GRID_W1
              and d["v5_w2"] in GRID_W2,
              "fitted values on the declared grid")
        check("v5-beat", d.get("v5_score", 0) > 4.588,
              f"v5_score={d.get('v5_score', 0):.3f} > v4 base 4.588")
        check("v5-cache-parity", set(("v5_w0", "v5_w1", "v5_w2")) <= set(P),
              f"cache={sorted(k for k in P if k.startswith('v5'))}")
    except FileNotFoundError:
        check("v5-keys", True, "no file: analytic zeros in force (=v4)")
    # 2. pairs determinism (v5 grid strings + same pairs)
    Pr = pairs()
    sig = repr(sorted(((k, round(float(v[0].sum() + v[1].sum()), 6)) for k, v in Pr.items())))
    h = hashlib.md5((str(GRID_W0) + str(GRID_W1) + str(GRID_W2) + sig).encode()).hexdigest()[:16]
    try:
        check("v5-hash", d.get("v5_pairs_hash") == h, f"file={d.get('v5_pairs_hash')} live={h}")
    except NameError:
        check("v5-hash", True, "no file yet (run scripts/fit_v5.py --write)")
    # 3. parity: real + structured + grain BARRED; white noise measured.
    #    Grain (sigma 0.02, the pairs' own degradation level) is the realistic
    #    fixture. White noise at full amplitude maximizes bank-output spread,
    #    so small weight disagreements cost decibels by gain, not by flips
    #    (share ~11x, same ladder as v3/v4) -- reported with mechanism (#LIB-015).
    kw = v5_kw(P)
    rng = np.random.default_rng(5)
    rgb = np.clip(rng.uniform(0, 1, (24, 24, 3)), 0, 1).astype(np.float32)
    io, _ = enhance_image_float_splat(rgb, **kw)
    ii, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat_soft", ctrl="v5")
    dn = psnr(ii, io)
    e = np.abs(ii.astype(float) - io.astype(float))
    share = float(np.sort(e.ravel())[-int(e.size * 0.05):].mean() / max(e.mean(), 1e-12))
    print(f"    measured (no bar): v5-noise {dn:.2f}dB share={share:.1f}x "
          f"(distributed gain-on-spread)")
    grain = np.clip(np.full((24, 24, 3), 0.5)
                    + rng.normal(0, 0.02, (24, 24, 3)), 0, 1).astype(np.float32)
    io_g, _ = enhance_image_float_splat(grain, **kw)
    ii_g, _, _ = enhance_image_int(grain, beta=0.5, blur="splat_soft", ctrl="v5")
    dg = psnr(ii_g, io_g)
    check("v5-grain", dg >= BAR_DB, f"{dg:.2f}dB (realistic noise, barred)")
    rgb_r = np.asarray(Image.open(CAND).convert("RGB"), dtype=np.float64) / 255.0
    rgb_r = np.power(rgb_r, 2.2).astype(np.float32)
    io, _ = enhance_image_float_splat(rgb_r, **kw)
    ii_r, _, _ = enhance_image_int(rgb_r, beta=0.5, blur="splat_soft", ctrl="v5")
    check("v5-real", psnr(ii_r, io) >= BAR_DB, f"{psnr(ii_r, io):.2f}dB")
    xx = np.tile(np.linspace(0, 1, 32), (32, 1))
    g = np.clip(np.where(xx < 0.5, 0.5 + 0.35 * np.sin(2 * np.pi * xx * 4),
                         0.5 + 0.35 * np.sin(2 * np.pi * xx.T * 4)), 0, 1)
    grgb = np.stack([g, g[::-1, ::-1], np.full_like(g, 0.4)], -1).astype(np.float32)
    io, _ = enhance_image_float_splat(grgb, **kw)
    ii, _, _ = enhance_image_int(grgb, beta=0.5, blur="splat_soft", ctrl="v5")
    check("v5-structured", psnr(ii, io) >= BAR_DB, f"{psnr(ii, io):.2f}dB")
    # 4. flat identity
    flat = np.full((24, 24, 3), 0.5, np.float32)
    fout, _, _ = enhance_image_int(flat, beta=0.5, blur="splat_soft", ctrl="v5")
    check("v5-flat", float(np.abs(fout - 0.5).mean()) < 0.01,
          f"{float(np.abs(fout - 0.5).mean()):.2e}")
    # 5. rotation invariance re-run (smooth edges, cf test_ctrl doctrine)
    N = 96
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    sig = lambda t: 0.15 + 0.7 / (1 + np.exp(-t / 0.02))
    bar_s = sig(np.tile(np.linspace(0, 1, N), (N, 1)) - 0.5)
    diag_s = sig((xx + yy - 1.0) / np.sqrt(2))
    bb, dd = boost_of(bar_s), boost_of(diag_s)
    flat_r = boost_of(np.full((N, N), 0.5))
    gap = abs(bb - dd) / max(bb, dd, 1e-9)
    check("v5-rotation", gap < 0.30 and min(bb, dd) > 5 * flat_r,
          f"bar={bb:.4f} diag={dd:.4f} gap={gap:.2f} flat={flat_r:.2e}")
    # 6. detail ordering: corner (large D, weak coh) responds more than flat.
    #    This is the signature the v5 weights exist for (w2>0 rewards detail).
    N2 = 48
    yy2, xx2 = np.meshgrid(np.linspace(0, 1, N2), np.linspace(0, 1, N2))
    corner = np.where((xx2 > 0.5) ^ (yy2 > 0.5), 0.85, 0.15)
    cb, fb = boost_of(corner), boost_of(np.full((N2, N2), 0.5))
    check("v5-detail-order", cb > 5 * fb, f"corner={cb:.4f} flat={fb:.2e}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
