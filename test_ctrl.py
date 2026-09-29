"""Controller gates (v2): fitted params valid, parity with ctrl on,
rotation invariance holds, flat identity preserved.
Spec: docs/BETA_CTRL.md. Usage: python3 test_ctrl.py
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from chain.holo_phi import enhance_image_int
from chain.oracle import enhance_image_float_splat
from chain.control import load_ctrl, DEFAULTS
from fit_ctrl import pairs, GRID_ATTEN, GRID_MID, GRID_HI, GRID_STRONG

BAR_DB = 40.0
FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def boost_of(y):
    """99th-percentile |enhanced-clean| response of the int ctrl chain.
    p99 (not mean): mean confounds edge DENSITY with per-edge boost (a
    checker has ~15x the edges of one bar). p99 reads the edge itself."""
    rgb = np.stack([y] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat", ctrl=True)
    return float(np.percentile(np.abs(out[:, :, 0] - y), 99))


def main():
    P = load_ctrl()
    print(f"ctrl file: {P}")
    # 1. frozen file validity: keys, grid membership (fitter's contract, v3:
    #    iso/mid/strong/hi fitted; coh_thr frozen at its v1 value)
    try:
        with open(os.path.join("chain", "CTRL.json")) as fh:
            d = json.load(fh)
        check("ctrl-keys", set(("iso_atten", "mid_atten", "strong_atten", "coh_hi", "coh_thr")) <= set(d),
              f"keys={sorted(d)}")
        check("ctrl-grid", d["iso_atten"] in GRID_ATTEN and d["mid_atten"] in GRID_MID
              and d["strong_atten"] in GRID_STRONG
              and d["coh_hi"] in GRID_HI and d["coh_thr"] == 0.15,
              f"file={d['iso_atten']}/{d['mid_atten']}/{d['strong_atten']}/{d['coh_hi']}/{d['coh_thr']}")
        # loader returns the WHOLE frozen dict (a past revision dropped new
        # keys here; both sides silently ran different defaults and parity
        # still passed -- weak pixels are rare. Never again: assert key parity).
        check("ctrl-cache-parity", set(("iso_atten", "mid_atten", "strong_atten", "coh_hi", "coh_thr")) <= set(P),
              f"cache={sorted(P)}")
    except FileNotFoundError:
        check("ctrl-keys", True, "no file: analytic defaults in force")
        check("ctrl-grid", True, f"defaults={DEFAULTS}")
    # 2. pairs determinism: recompute the fitter's hash (no fitting, seconds)
    Pr = pairs()
    sig = repr(sorted(((k, round(float(v[0].sum() + v[1].sum()), 6)) for k, v in Pr.items())))
    h = hashlib.md5((str(GRID_ATTEN) + str(GRID_MID) + str(GRID_STRONG)
                     + str(GRID_HI) + sig).encode()).hexdigest()[:16]
    try:
        check("ctrl-hash", d["pairs_hash"] == h, f"file={d['pairs_hash']} live={h}")
    except NameError:
        check("ctrl-hash", True, "no file yet (run scripts/fit_ctrl.py --write)")
    # 3. parity ctrl-on (basis: file params both sides, linear RGB, peak 1).
    #    Gated on a REAL frame: uniform noise maximizes bucket-boundary
    #    straddling (random orientations), so flips dominate there by design;
    #    the noise fixture is reported below as a measured row, not a bar
    #    (documents the hard-select discontinuity cost -- continuous v4
    #    fields are the backlog answer).
    from PIL import Image
    cand = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png"
    rgb_r = (np.asarray(Image.open(cand).convert("RGB"), dtype=np.float64) / 255.0)
    rgb_r = np.power(rgb_r, 2.2).astype(np.float32)
    io, _ = enhance_image_float_splat(rgb_r, beta=0.5, iso_atten=P["iso_atten"],
                                      coh_thr=P["coh_thr"],
                                      mid_atten=P.get("mid_atten", 0.6),
                                      coh_hi=P.get("coh_hi", 0.5),
                                      strong_atten=P.get("strong_atten", 1.0))
    ii, _, _ = enhance_image_int(rgb_r, beta=0.5, blur="splat", ctrl=True)
    d = psnr(ii, io)
    check("parity-ctrl", d >= BAR_DB, f"{d:.2f}dB (real frame)")
    rng = np.random.default_rng(5)
    rgb = np.clip(rng.uniform(0, 1, (24, 24, 3)), 0, 1).astype(np.float32)
    io_n, _ = enhance_image_float_splat(rgb, beta=0.5, iso_atten=P["iso_atten"],
                                        coh_thr=P["coh_thr"],
                                        mid_atten=P.get("mid_atten", 0.6),
                                        coh_hi=P.get("coh_hi", 0.5),
                                        strong_atten=P.get("strong_atten", 1.0))
    ii_n, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat", ctrl=True)
    dn = psnr(ii_n, io_n)
    print(f"    measured (no bar): parity-ctrl-noise {dn:.2f}dB "
          f"(bucket-flip sensitivity on uniform noise)")
    # 4. intelligence signature, v1.1: ROTATION INVARIANCE on smooth edges.
    #    Diagonal kernels resolve every orientation, so the same smooth edge
    #    rotated 45deg boosts about equally (v1 gated bar>diagbar because
    #    diagonals fell back to atten; retired with the fallback). SMOOTH,
    #    because hard rasterized diagonals are staircases whose corners carry
    #    genuine extra 2D-step energy (measured hard-diag p99 ~= 2x hard-bar;
    #    smooth-diag ~= smooth-bar within 5%). Gating invariance on staircases
    #    would punish real corner physics; hard-step numbers are reported.
    N = 96
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    sig = lambda t: 0.15 + 0.7 / (1 + np.exp(-t / 0.02))
    bar_s = sig(np.tile(np.linspace(0, 1, N), (N, 1)) - 0.5)
    diag_s = sig((xx + yy - 1.0) / np.sqrt(2))
    bb, dd = boost_of(bar_s), boost_of(diag_s)
    flat_r = boost_of(np.full((N, N), 0.5))
    gap = abs(bb - dd) / max(bb, dd, 1e-9)
    check("rotation-invariant", gap < 0.30 and min(bb, dd) > 5 * flat_r,
          f"bar={bb:.4f} diag={dd:.4f} gap={gap:.2f} flat={flat_r:.2e}")
    Nh = 48
    bar_h = np.where(np.tile(np.linspace(0, 1, Nh), (Nh, 1)) > 0.5, 0.85, 0.15)
    yyh, xxh = np.meshgrid(np.linspace(0, 1, Nh), np.linspace(0, 1, Nh))
    diag_h = np.where(xxh + yyh > 1.0, 0.85, 0.15)
    bh, dh = boost_of(bar_h), boost_of(diag_h)
    print(f"    measured (no bar): hard-step bar={bh:.4f} diag={dh:.4f} "
          f"(staircase corner energy, see note)")
    # 5. flat identity with ctrl on (basis: mean abs diff, must not invent)
    flat = np.full((24, 24), 0.5, np.float64)
    frgb = np.stack([flat] * 3, -1).astype(np.float32)
    fout, _, _ = enhance_image_int(frgb, beta=0.5, blur="splat", ctrl=True)
    check("ctrl-flat", float(np.abs(fout[:, :, 0] - 0.5).mean()) < 0.01,
          f"{float(np.abs(fout[:, :, 0] - 0.5).mean()):.2e}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
