"""v4 continuity gates: fully-continuous selection clears the noise bar.

v3's hard selects cost 28-33dB on uniform noise (bucket flips at large local
kernel cost). v4 (sigmoid-blended beta + relu-blended blur) has no discrete
decision anywhere, so quantization cannot flip one: noise parity is BARRED
here (>=40dB), not merely reported. Structured + real + flat gates re-run
under v4; the v3 suites keep gating the hard path. Spec: docs/BETA_CTRL.md.
Usage: python3 tests/test_v4.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from chain.holo_phi import enhance_image_int
from chain.oracle import enhance_image_float_splat
from chain.control import load_ctrl

BAR_DB = 40.0
FAIL = []
_CAND_EXT = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "rife_reverse", "samples", "f_012.png")
CAND = _CAND_EXT if os.path.isfile(_CAND_EXT) else os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "samples", "input_example.png")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def v4_args(P):
    return dict(beta=0.5, iso_atten=P["iso_atten"], coh_thr=P["coh_thr"],
                mid_atten=P.get("mid_atten", 0.6), coh_hi=P.get("coh_hi", 0.5),
                strong_atten=P.get("strong_atten", 1.0), soft=True,
                soft_blur=True)


def main():
    P = load_ctrl()
    print(f"ctrl file: iso={P['iso_atten']} mid={P.get('mid_atten')} "
          f"strong={P.get('strong_atten')} hi={P.get('coh_hi')}")
    # 1. THE v4 proof: noise parity barred (was 28-33dB measured rows)
    rng = np.random.default_rng(5)
    rgb = np.clip(rng.uniform(0, 1, (24, 24, 3)), 0, 1).astype(np.float32)
    kw = v4_args(P)
    io, _ = enhance_image_float_splat(rgb, **kw)
    ii, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat_soft", ctrl="soft")
    dn = psnr(ii, io)
    check("v4-noise", dn >= BAR_DB, f"{dn:.2f}dB (was 32.76 pre-gate)")
    # 2. real frame
    rgb_r = np.asarray(Image.open(CAND).convert("RGB"), dtype=np.float64) / 255.0
    rgb_r = np.power(rgb_r, 2.2).astype(np.float32)
    io, _ = enhance_image_float_splat(rgb_r, **kw)
    ii_r, _, _ = enhance_image_int(rgb_r, beta=0.5, blur="splat_soft", ctrl="soft")
    check("v4-real", psnr(ii_r, io) >= BAR_DB, f"{psnr(ii_r, io):.2f}dB")
    # 3. structured pairing (gratings, cf test_splat doctrine)
    xx = np.tile(np.linspace(0, 1, 32), (32, 1))
    g = np.clip(np.where(xx < 0.5, 0.5 + 0.35 * np.sin(2 * np.pi * xx * 4),
                         0.5 + 0.35 * np.sin(2 * np.pi * xx.T * 4)), 0, 1)
    grgb = np.stack([g, g[::-1, ::-1], np.full_like(g, 0.4)], -1).astype(np.float32)
    io, _ = enhance_image_float_splat(grgb, **kw)
    ii, _, _ = enhance_image_int(grgb, beta=0.5, blur="splat_soft", ctrl="soft")
    check("v4-structured", psnr(ii, io) >= BAR_DB, f"{psnr(ii, io):.2f}dB")
    # 4. flat identity (continuous weights must not invent structure)
    flat = np.full((24, 24, 3), 0.5, np.float32)
    fout, _, _ = enhance_image_int(flat, beta=0.5, blur="splat_soft", ctrl="soft")
    check("v4-flat", float(np.abs(fout - 0.5).mean()) < 0.01,
          f"{float(np.abs(fout - 0.5).mean()):.2e}")
    # 5. rotation invariance on the SOFT blur path (added late: v4 shipped
    #    with rotation gated only on the hard-mux path, which is how the
    #    diagonal-weight swap survived -- parity mirrors bugs it can't see.
    #    Smooth fixtures, same doctrine as test_ctrl/test_v5.)
    N = 96
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    sig = lambda t: 0.15 + 0.7 / (1 + np.exp(-t / 0.02))
    bar_s = sig(np.tile(np.linspace(0, 1, N), (N, 1)) - 0.5)
    diag_s = sig((xx + yy - 1.0) / np.sqrt(2))

    def boost4(y):
        rgb = np.stack([y] * 3, -1).astype(np.float32)
        out, _, _ = enhance_image_int(rgb, beta=0.5, blur="splat_soft",
                                      ctrl="soft")
        return float(np.percentile(np.abs(out[:, :, 0] - y), 99))

    bb, dd = boost4(bar_s), boost4(diag_s)
    flat_r = boost4(np.full((N, N), 0.5))
    gap = abs(bb - dd) / max(bb, dd, 1e-9)
    check("v4-rotation", gap < 0.30 and min(bb, dd) > 5 * flat_r,
          f"bar={bb:.4f} diag={dd:.4f} gap={gap:.2f} flat={flat_r:.2e}")
    # 6. sanity: v4 sharpens the real frame (Laplacian variance up)
    from scipy.ndimage import laplace

    def sharp(im):
        y = (0.2126 * im[:, :, 0] + 0.7152 * im[:, :, 1]
             + 0.0722 * im[:, :, 2]).astype(np.float64)
        return float(np.var(y[2:, 1:-1] + y[:-2, 1:-1] + y[1:-1, 2:]
                            + y[1:-1, :-2] - 4 * y[1:-1, 1:-1]))

    check("v4-sharpens", sharp(ii_r) > sharp(rgb_r),
          f"{sharp(rgb_r):.2e} -> {sharp(ii_r):.2e} (linear-light units)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
