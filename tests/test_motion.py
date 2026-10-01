"""Motion side-channel gates (step: side channels #1).

The seam carries forward flow (pixels, frame-A geometry); the consensus rule
lets flow CONFIRM or VETO the tensor claim, never originate one (aperture
problem: panning along an edge breaks any perpendicular assumption).
Bases declared per test. Spec: chain/motion.py + docs/MOTION.md (when
written; until then the motion.py docstring is the spec).
Usage: python3 tests/test_motion.py
"""
import os
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import holo_phi as H
from chain import splat as SP
from chain import oracle as O
from chain.control import load_ctrl
from chain.motion import quantize, _perp_of, FLOW_REF, FLOW_ATTEN

BAR_DB = 40.0
FAIL = []
_CAND_EXT = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "rife_reverse", "samples", "f_012.png")
CAND = _CAND_EXT if os.path.isfile(_CAND_EXT) else os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "samples", "input_example.png")
N = 48


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def heavy_bar():
    """Translating-bar fixture with EXACT flow by construction: vertical bar
    moving +x (12px), heavy motion smear + noise. Tensor genuinely fails here
    (edge coh ~0.0, oriented 0.42); flow must rescue V without inventing."""
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                         indexing="ij")
    bar = np.where(xx > 0.5, 0.85, 0.15)
    rng = np.random.default_rng(23)
    heavy = np.clip(gaussian_filter(bar.astype(float), (0, 4.0))
                    + rng.normal(0, 0.08, bar.shape), 0, 1)
    fl = np.zeros((N, N, 2))
    fl[..., 1] = 12.0  # +x translation; perp must be V (0)
    band = np.zeros((N, N), bool)
    band[6:-6, 22:27] = True
    return heavy, fl, band


def main():
    m_acc, m_cov = H._load_scales()
    P = load_ctrl()
    print(f"ctrl: thr={P['coh_thr']} hi={P.get('coh_hi')} "
          f"flow_ref={FLOW_REF} flow_atten={FLOW_ATTEN}")

    # 0. compass: quantize agrees with the reference mapping on 8 points +
    #    octant boundaries (a swap here is parity-invisible -- this gate is
    #    load-bearing, cf #LIB-014 postscript).
    angs = [0, 45, 90, 135, 180, 225, 270, 315,
            22.4, 22.6, 67.4, 67.6, 112.4, 112.6]
    cok = True
    for a in angs:
        r = np.deg2rad(a)
        f = np.zeros((1, 1, 2))
        f[0, 0] = [np.sin(r), np.cos(r)]
        q, _, _ = quantize(f)
        if int(q[0, 0]) != _perp_of(a):
            cok = False
            print(f"    compass mismatch at {a}: {int(q[0, 0])} vs {_perp_of(a)}")
    check("compass", cok, f"{len(angs)} angles, twins agree")

    # 1. consensus rescues V on the heavy fixture (truth V=0)
    heavy, fl, band = heavy_bar()
    a_t = H.sqrt_trip(S.encode(heavy))
    j = SP.structure(a_t, m_acc, m_cov)
    _, b0, _ = SP.coherence_bucket(*j, m_cov, m_acc, coh_thr=P["coh_thr"])
    _, b1, d1 = SP.coherence_bucket(*j, m_cov, m_acc, coh_thr=P["coh_thr"],
                                    flow=quantize(fl))
    v0, v1 = float((b0[band] == 0).mean()), float((b1[band] == 0).mean())
    check("rescue-V", v1 > 0.7 and v1 > 2 * v0, f"V {v0:.2f} -> {v1:.2f}")
    wrong = float(((b1[band] == 2) | (b1[band] == 3)).mean())
    check("rescue-no-invent", wrong < 0.15, f"wrong-oriented {wrong:.2f}")

    # 2. zero flow is BIT-IDENTICAL to blind (not dB: static pixels take the
    #    direct path and motion=None skips the branch -- both exact).
    rng = np.random.default_rng(5)
    rgb = np.clip(rng.uniform(0, 1, (24, 24, 3)), 0, 1).astype(np.float32)
    a, _, _ = H.enhance_image_int(rgb, beta=0.5, blur="splat", ctrl=True)
    b, _, _ = H.enhance_image_int(rgb, beta=0.5, blur="splat", ctrl=True,
                                  motion=np.zeros((24, 24, 2)))
    check("zeroflow-exact", bool((a == b).all()), "identical bytes")
    f = np.clip(rng.uniform(0, 1, (24, 24)), 0, 1)
    f0, _ = H.enhance_luminance_int(f, beta=0.5, blur="splat", ctrl=True)
    f1, _ = H.enhance_luminance_int(f, beta=0.5, blur="splat", ctrl=True,
                                    motion=np.zeros((24, 24, 2)))
    q0 = S.to_fixed(f0[0], f0[1], f0[2], H._load_scales()[1])
    q1 = S.to_fixed(f1[0], f1[1], f1[2], H._load_scales()[1])
    check("zeroflow-lum-exact", bool((q0 == q1).all()), "fixed counts equal")

    # 3. panning-along-edge keeps tensor behavior (consensus never punishes
    #    confident local evidence: tensor-strong path preserves).
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                         indexing="ij")
    hbar = np.where(yy > 0.5, 0.85, 0.15)  # horizontal edge...
    pan = np.zeros((N, N, 2))
    pan[..., 1] = 10.0  # ...moving ALONG itself (flow says V, truth H)
    ah = H.sqrt_trip(S.encode(hbar))
    jh = SP.structure(ah, m_acc, m_cov)
    _, bh0, _ = SP.coherence_bucket(*jh, m_cov, m_acc, coh_thr=P["coh_thr"])
    _, bh1, _ = SP.coherence_bucket(*jh, m_cov, m_acc, coh_thr=P["coh_thr"],
                                    flow=quantize(pan))
    hband = np.zeros((N, N), bool)
    hband[22:27, 6:-6] = True
    h0, h1 = float((bh0[hband] == 1).mean()), float((bh1[hband] == 1).mean())
    check("pan-preserve", h1 >= 0.9 * h0 and h1 > 0.7,
          f"H {h0:.2f} -> {h1:.2f} (flow must not veto confidence)")

    # 4. parity motion-vs-motion (basis: file params + flow both sides,
    #    linear RGB [0,1], peak 1; uniform translation field).
    uni = np.ones((24, 24, 2)) * np.array([2.0, 6.0])
    io, _ = O.enhance_image_float_splat(rgb, beta=0.5, iso_atten=P["iso_atten"],
                                        coh_thr=P["coh_thr"],
                                        mid_atten=P.get("mid_atten", 0.6),
                                        coh_hi=P.get("coh_hi", 0.5),
                                        strong_atten=P.get("strong_atten", 1.0),
                                        motion=uni)
    ii, _, _ = H.enhance_image_int(rgb, beta=0.5, blur="splat", ctrl=True,
                                   motion=uni)
    d = psnr(ii, io)
    check("parity-motion", d >= BAR_DB, f"{d:.2f}dB")

    # 5. end-to-end panning preservation + rotation with flow present.
    #    Pan ALONG a sharp edge (flow claims V, truth H): confident tensor
    #    evidence must survive end-to-end (response within 20% of blind).
    #    Rotation: smooth bar vs smooth diag under zero flow must match blind
    #    gap (flow path adds nothing when static -- exactness, belted).
    N2 = 96
    Hh = 384
    yy2, xx2 = np.meshgrid(np.linspace(0, 1, Hh), np.linspace(0, 1, Hh))
    sig = lambda t: 0.15 + 0.7 / (1 + np.exp(-t / 0.02))
    bar_s = sig(np.tile(np.linspace(0, 1, Hh), (Hh, 1)) - 0.5)
    bar_s = bar_s.reshape(N2, 4, N2, 4).mean((1, 3))
    diag_s = sig((xx2 + yy2 - 1.0) / np.sqrt(2))
    diag_s = diag_s.reshape(N2, 4, N2, 4).mean((1, 3))

    def boost5(y, fl):
        r = np.stack([y] * 3, -1).astype(np.float32)
        out, _, _ = H.enhance_image_int(r, beta=0.5, blur="splat_soft",
                                        ctrl="v5", motion=fl)
        return float(np.percentile(np.abs(out[:, :, 0] - y), 99))

    pan_N = np.zeros((N, N, 2))
    pan_N[..., 1] = 10.0
    hrgb = np.stack([hbar] * 3, -1).astype(np.float32)
    out_b, _, _ = H.enhance_image_int(hrgb, beta=0.5, blur="splat_soft",
                                      ctrl="v5")
    out_p, _, _ = H.enhance_image_int(hrgb, beta=0.5, blur="splat_soft",
                                      ctrl="v5", motion=pan_N)
    rb = float(np.percentile(np.abs(out_b[:, :, 0] - hbar), 99))
    rp = float(np.percentile(np.abs(out_p[:, :, 0] - hbar), 99))
    # buckets preserved (pan-preserve) AND gain exactly halved: uniform 10px
    # flow -> norm=1 -> scale=0.5 everywhere. Ratio ~= FLOW_ATTEN proves the
    # ONLY end-to-end change is the stated caution (no bucket damage).
    ratio = rp / max(rb, 1e-12)
    check("pan-e2e", 0.4 < ratio < 0.6,
          f"blind={rb:.4f} pan={rp:.4f} ratio={ratio:.3f} (expect ~0.5)")
    zf = np.zeros((N2, N2, 2))
    bb, dd = boost5(bar_s, zf), boost5(diag_s, zf)
    gap = abs(bb - dd) / max(bb, dd, 1e-9)
    check("rotation-zeroflow", gap < 0.30,
          f"bar={bb:.4f} diag={dd:.4f} gap={gap:.2f}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
