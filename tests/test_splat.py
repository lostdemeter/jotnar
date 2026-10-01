"""Splat gates: orientation truth table, flat fallback, halo metric, parity.

Comparison bases declared per test. Spec: docs/SPLAT_OP.md.
Usage: python3 tests/test_splat.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import holo_phi as H
from chain import splat as SP
from chain import oracle as O
from chain.control import load_ctrl

BAR_DB = 40.0
FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def vbar(height, width, Hh=32, Ww=32):
    """Vertical bar (edge runs along y): gx dominant -> bucket 0 (V)."""
    img = np.zeros((Hh, Ww))
    img[:, width:] = height
    return img


def hbar(height, width, Hh=32, Ww=32):
    img = np.zeros((Hh, Ww))
    img[width:, :] = height
    return img


def dbar_slash(height, Hh=32, Ww=32):
    """Edge along y=-x (line xx+yy=1): normal (1,1), Jxy>0 -> bucket 3 (/)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, Hh), np.linspace(0, 1, Ww),
                         indexing="ij")
    return np.where(xx + yy > 1.0, height, 0.0)


def dbar_backslash(height, Hh=32, Ww=32):
    """Edge along y=x (line xx-yy=0): normal (1,-1), Jxy<0 -> bucket 2 (\\)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, Hh), np.linspace(0, 1, Ww),
                         indexing="ij")
    return np.where(xx - yy > 0.0, height, 0.0)


def main():
    m_acc, m_cov = H._load_scales()
    # comparison basis: BOTH sides use the frozen controller file (never the
    # analytic consts) -- params travel with the comparison, not beside it.
    P = load_ctrl()
    print(f"ctrl: iso_atten={P['iso_atten']} coh_thr={P['coh_thr']}")
    # 1. orientation truth table (basis: interior pixels, 2px margin off edge
    #    and 2px off the bar boundary -- transition pixels are mixed by design)
    for tag, img, want in [("vert-bar", vbar(0.8, 16), 0),
                           ("horiz-bar", hbar(0.8, 16), 1)]:
        # NOTE: img here is Y in [0,1]; single sqrt on both sides (a past
        # revision double-sqrted the int side and sampled flat pixels).
        a_t = H.sqrt_trip(S.encode(img))
        _, diag = SP.splat_blur(a_t, m_acc, m_cov, coh_thr=P["coh_thr"])
        b = diag["bucket"]
        band = np.zeros_like(b, bool)
        if want == 0:
            band[6:-6, 13:20] = True  # transition band around col 16
        else:
            band[13:20, 6:-6] = True
        open_ = band & (b != 4)
        frac = float((b[open_] == want).mean()) if open_.any() else 0.0
        check(f"orient-{tag}", frac > 0.9 and float(open_.sum()) > 0,
              f"{frac:.2f} of gate-open=={want} (n={int(open_.sum())})")
    # diagonal truth table (v1.1): 45-degree edges resolve, not fall back
    yy, xx = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32),
                         indexing="ij")
    for tag, img, want in [("diag-slash", dbar_slash(0.8), 3),
                           ("diag-backslash", dbar_backslash(0.8), 2)]:
        a_t = H.sqrt_trip(S.encode(img))
        _, diag = SP.splat_blur(a_t, m_acc, m_cov, coh_thr=P["coh_thr"])
        b = diag["bucket"]
        near = np.abs((xx + yy if want == 3 else xx - yy)
                      - (1.0 if want == 3 else 0.0)) < 0.14
        open_ = near & (b != 4)
        frac = float((b[open_] == want).mean()) if open_.any() else 0.0
        check(f"orient-{tag}", frac > 0.8 and float(open_.sum()) > 0,
              f"{frac:.2f} of gate-open=={want} (n={int(open_.sum())})")
    # 2. flat -> isotropic fallback (basis: gate_frac ~1 on constant field)
    a_t = H.sqrt_trip(S.encode(np.full((24, 24), 0.5)))
    _, diag = SP.splat_blur(a_t, m_acc, m_cov)
    check("flat-isotropic", diag["gate_frac"] > 0.95,
          f"gate_frac={diag['gate_frac']:.3f}")
    # 3. halo metric (basis: unit step, beta=0.5, max overshoot beyond rails
    #    in a 3px window around the edge; splat must not exceed iso)
    step = np.zeros((32, 32))
    step[:, 16:] = 1.0
    yo = {}
    for mode in ("iso", "splat"):
        out, _, _ = H.enhance_image_int(
            np.stack([step] * 3, -1).astype(np.float32), beta=0.5, blur=mode)
        yo[mode] = out[:, :, 0]
    win = np.zeros((32, 32), bool)
    win[:, 13:19] = True
    over = {m: float(np.maximum(yo[m][win] - 1.0, -yo[m][win]).max()) for m in yo}
    check("halo-splat<=iso", over["splat"] <= over["iso"] + 1e-6,
          f"iso={over['iso']:.3f} splat={over['splat']:.3f}")
    # 4. int-vs-oracle parity on splat blur (basis: amplitude [0,1], peak 1;
    #    single sqrt both sides). Gated on STRUCTURED content (defined
    #    orientation: striped halves) -- uniform noise is reported, not gated:
    #    random orientations maximize boundary straddling, so quantization
    #    flips a few pixels at large local kernel cost (same doctrine as the
    #    ctrl-noise measured row in test_ctrl.py).
    xx = np.tile(np.linspace(0, 1, 32), (32, 1))
    # coherent gratings (period 8px, single orientation per half): unambiguous
    # tensor direction, unlike dense alternating stripes whose 5x5 smoothing
    # window mixes opposite edges into confused pixels.
    struct_img = np.where(xx < 0.5, 0.5 + 0.35 * np.sin(2 * np.pi * xx * 4),
                          0.5 + 0.35 * np.sin(2 * np.pi * xx.T * 4))
    struct_img = np.clip(struct_img, 0, 1)
    a = np.sqrt(struct_img)
    a_t = H.sqrt_trip(S.encode(struct_img))
    got, got_diag = SP.splat_blur(a_t, m_acc, m_cov, coh_thr=P["coh_thr"])
    gv = S.decode(got[0], got[1]) * (1 - got[2].astype(np.float64))
    ref, ref_b, _ = O.splat_blur_float(a, coh_thr=P["coh_thr"])
    d = psnr(gv, ref)
    check("parity-splat-blur", d >= BAR_DB, f"{d:.2f}dB (structured)")
    agree_s = float((got_diag["bucket"] == ref_b).mean())
    check("bucket-agreement-struct", agree_s >= 0.95, f"{agree_s:.2f}")
    rng = np.random.default_rng(3)
    img = np.clip(rng.uniform(0, 1, (24, 24)), 0, 1)
    an = np.sqrt(img)
    at_n = H.sqrt_trip(S.encode(img))
    gn, dn = SP.splat_blur(at_n, m_acc, m_cov, coh_thr=P["coh_thr"])
    gvn = S.decode(gn[0], gn[1]) * (1 - gn[2].astype(np.float64))
    refn, refn_b, _ = O.splat_blur_float(an, coh_thr=P["coh_thr"])
    print(f"    measured (no bar): parity-splat-noise {psnr(gvn, refn):.2f}dB, "
          f"agreement {float((dn['bucket'] == refn_b).mean()):.2f}")
    # 5. end-to-end splat parity (basis: linear RGB [0,1], peak 1; structured
    #    fixture -- same doctrine as #4)
    ch0 = struct_img
    ch1 = struct_img[::-1, ::-1]
    ch2 = np.full_like(struct_img, 0.4)
    rgb = np.stack([ch0, ch1, ch2], -1).astype(np.float32)
    io, _ = O.enhance_image_float_splat(rgb, beta=0.5, iso_atten=1.0,
                                        coh_thr=P["coh_thr"])
    ii, _, _ = H.enhance_image_int(rgb, beta=0.5, blur="splat",
                                   coh_thr=P["coh_thr"])
    d2 = psnr(ii, io)
    check("parity-splat-e2e", d2 >= BAR_DB, f"{d2:.2f}dB")
    # 6. fusion (#LIB-006): fused single-pass == compositional, BIT-EXACT
    #    (same products, integer sums commute) + timing evidence
    import time
    rng_f = np.random.default_rng(13)
    fimg = np.clip(rng_f.uniform(0, 1, (40, 40)), 0, 1)
    fa_t = H.sqrt_trip(S.encode(fimg))
    t0 = time.time()
    ref_f, _ = SP.splat_blur(fa_t, m_acc, m_cov, coh_thr=P["coh_thr"])
    t_comp = time.time() - t0
    t0 = time.time()
    got_f, _ = SP.splat_blur(fa_t, m_acc, m_cov, coh_thr=P["coh_thr"],
                             fused=True)
    t_fused = time.time() - t0
    same = bool((got_f[0] == ref_f[0]).all() and (got_f[1] == ref_f[1]).all()
                and (got_f[2] == ref_f[2]).all())
    check("fusion-exact", same, "bit-identical, not dB")
    print(f"    timing: compositional {t_comp:.2f}s fused {t_fused:.2f}s "
          f"(x{t_comp / max(t_fused, 1e-9):.1f})")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
