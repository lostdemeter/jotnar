"""Depth-composition gates (step 3a, L1): DAV2 prior modulates beta.

Bases declared per test. Spec: docs/COMPOSE.md. The prior is an offline
artifact (float allowed at the boundary, never in the hot loop); what runs
hot is integer (median mask at encode, select_mux + tmul in chain).
Usage: python3 test_depth.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain.holo_phi import enhance_image_int
from chain.oracle import enhance_image_float_splat
from chain.control import load_ctrl
from chain.depthprior import get_depth, near_mask, FAR_ATTEN

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


def main():
    P = load_ctrl()
    rgb01 = np.asarray(Image.open(CAND).convert("RGB"), dtype=np.float64) / 255.0
    rgb_lin = np.power(rgb01, 2.2).astype(np.float32)
    # 1. prior determinism: same tag+bytes -> same file, no recompute
    try:
        d1 = get_depth(rgb01, "f_012")
        d2 = get_depth(rgb01, "f_012")
    except (ImportError, OSError) as e:
        print(f"SKIP (needs DAV2 checkout + weights for cache miss: {e})")
        sys.exit(0)
    check("depth-cache", bool((d1 == d2).all()), f"shape={d1.shape}")
    check("depth-range", bool(np.isfinite(d1).all() and d1.max() > d1.min()),
          f"[{d1.min():.2f},{d1.max():.2f}]")
    near = near_mask(d1)
    print(f"    near_frac={near.mean():.3f}")
    y = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
         + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)
    # 2. parity depth-on (basis: file params + prior both sides, peak 1)
    io, _ = enhance_image_float_splat(rgb_lin, beta=0.5, iso_atten=P["iso_atten"],
                                      coh_thr=P["coh_thr"],
                                      mid_atten=P.get("mid_atten", 1.0),
                                      coh_hi=P.get("coh_hi", 0.5),
                                      depth=d1, far_atten=FAR_ATTEN)
    ii, _, info = enhance_image_int(rgb_lin, beta=0.5, blur="splat", ctrl=True,
                                    depth=d1)
    d = psnr(ii, io)
    check("parity-depth", d >= BAR_DB, f"{d:.2f}dB")
    # 3. ordering: near field boosts MORE than far field (the stated rule).
    #    Basis: p99 response over each mask on linear Y.
    r = np.abs(ii[:, :, 0].astype(np.float64) - rgb_lin[:, :, 0].astype(np.float64))
    # Y-channel response, not R (chroma gain confounds): recompute on Y
    from chain.holo_phi import enhance_luminance_int
    yl, _ = enhance_luminance_int(y, beta=0.5, blur="splat", ctrl=True, depth=d1)
    import phi_core.lattice as S
    yenh = np.clip(S.decode(yl[0], yl[1]) * (1 - yl[2].astype(np.float64)), 0, 1)
    ry = np.abs(yenh - y)
    pn, pf = float(np.percentile(ry[near], 99)), float(np.percentile(ry[~near], 99))
    check("near-gt-far", pn > pf, f"near={pn:.4f} far={pf:.4f}")
    # 4. toggle: depth=None path unaffected (basis: identical to pre-depth run)
    io0, _ = enhance_image_float_splat(rgb_lin, beta=0.5, iso_atten=P["iso_atten"],
                                       coh_thr=P["coh_thr"],
                                       mid_atten=P.get("mid_atten", 1.0),
                                       coh_hi=P.get("coh_hi", 0.5))
    ii0, _, _ = enhance_image_int(rgb_lin, beta=0.5, blur="splat", ctrl=True)
    check("toggle-clean", psnr(ii0, io0) >= BAR_DB, f"{psnr(ii0, io0):.2f}dB")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
