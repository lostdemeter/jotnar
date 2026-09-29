"""splat_blur: structure-tensor anisotropic blur (splats-lite, step 1).

Replaces the blind isotropic Gaussian with content-aware selection among an
oriented bank, all-integer, no learning. Spec: docs/SPLAT_OP.md. Library
notes: docs/LIBRARY_NOTES.md.

v1.1 bank (5-way): 0=V (edge along y), 1=H (edge along x), 2=diag \\
(edge along y=x), 3=diag / (edge along y=-x), 4=isotropic fallback.
Sign convention: \\-edge has normal (1,-1) so Jxy<0; Jxy>=0 picks /.
sq==0 exactly -> 2 (deterministic tie-break, documented).
"""
import numpy as np

import sys
import os
sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phi_core.lattice as S
from chain import holo_phi as H
from chain.control import load_ctrl
from chain.holo_phi import (conv_trip, tmul, binop_fixed, sqrt_trip,
                            tdiv_trip, clip_fixed)

COH_THR = 0.25  # analytic init; live value comes from control (fitted)

# Sobel/8: normalized offline so |g| <= 0.5 on [0,1] input -- tensor domain
# stays inside m_cov coverage (#LIB-003). Calibration asserts this.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def aniso_kernel(sx, sy, radius=2):
    """Axis-aligned anisotropic Gaussian, offline deterministic build."""
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    gx = np.exp(-0.5 * (ax / sx) ** 2)
    gy = np.exp(-0.5 * (ax / sy) ** 2)
    k = np.outer(gy, gx)
    return k / k.sum()


def rotated_kernel(theta_deg, s_long=1.8, s_short=0.5, radius=2):
    """Anisotropic Gaussian elongated along edge direction theta_deg.
    theta=0 -> H, 90 -> V, 45 -> \\, -45 -> /. Offline deterministic."""
    th = np.deg2rad(theta_deg)
    ux, uy = np.cos(th), np.sin(th)  # along-edge unit vector
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    u = xx * ux + yy * uy
    v = -xx * uy + yy * ux
    k = np.exp(-0.5 * ((u / s_long) ** 2 + (v / s_short) ** 2))
    return k / k.sum()


def bank():
    """Frozen formulas: V/H/diag\\/diag//iso. Deterministic."""
    return [rotated_kernel(90),    # 0: V -- edge along y
            rotated_kernel(0),     # 1: H -- edge along x
            rotated_kernel(45),    # 2: diag \\ -- edge along y=x
            rotated_kernel(-45),   # 3: diag / -- edge along y=-x
            aniso_kernel(1.0, 1.0, radius=2)]  # 4: isotropic fallback


def tensor_smooth_kernel():
    from chain.holo_phi import gaussian_kernel
    return gaussian_kernel(radius=1, sigma=0.8)


def const_trip(v, shape):
    return S.encode(np.full(shape, float(v), dtype=np.float64))


def zero_trip(shape, m):
    return S.from_fixed(np.zeros(shape, dtype=np.int64), m)


def gradients(a_t, m_acc, m_cov):
    gx = conv_trip(a_t, SOBEL_X, m_acc, m_out=m_cov)
    gy = conv_trip(a_t, SOBEL_Y, m_acc, m_out=m_cov)
    return gx, gy


def structure(a_t, m_acc, m_cov):
    """Tensor components (smoothed) + raw gradients. All triples."""
    gx, gy = gradients(a_t, m_acc, m_cov)
    jxx = tmul(gx, gx)
    jyy = tmul(gy, gy)
    jxy = tmul(gx, gy)
    k = tensor_smooth_kernel()
    return (conv_trip(jxx, k, m_acc, m_out=m_cov),
            conv_trip(jyy, k, m_acc, m_out=m_cov),
            conv_trip(jxy, k, m_acc, m_out=m_cov), gx, gy)


def coherence_bucket(jxx, jyy, jxy, gx, gy, m_cov, m_acc, coh_thr=None):
    """coh in [0,1] triples + integer bucket map {0:V,1:H,2:\\,3:/,4:iso}.

    Orientation comes from the SMOOTHED tensor (never raw gradients).
    All compares integer in fixed domain, no atan:
      gate shut             -> 4 (iso)
      |2Jxy| > |Jxx-Jyy|    -> diagonal by sign(sq): sq<0 -> 2 (\\),
                               else 3 (/); sq==0 exactly -> 2 (tie-break)
      Jxx-Jyy > 0           -> 0 (V: x-gradient dominates)
      else                  -> 1 (H)
    NOTE on the diagonal sign: \\-edge (runs along y=x) has gradient normal
    (1,-1), so Jxy = gx*gy < 0 smoothed stays negative. Hence sq<0 -> \\ = 2,
    sq>=0 -> / = 3. The kernels match: bank[2] elongated along 45deg.
    coh = aniso/trace with 0/0 -> 0. gx/gy args kept for signature (unused)."""
    shape = jxx[0].shape
    # Discriminant squares are exact (tmul), but their SUM via binop at
    # m_cov underflows the fixed floor (~1.6e-5) on real pixels: disc lives
    # at amplitude^4 scale (measured 9e-6 with coh 0.49 -> lost to 0).
    # Summed at m_acc instead (floor ~1e-6, cap 0.26 covers disc<=0.25).
    # Everything downstream is triples (scale-free). (A normalize-first
    # variant was tried: 3 ratio quantizations beat the 1 sum quantization
    # badly -- agreement 0.65 vs 0.79. Sums of exact products win.)
    d = binop_fixed(jxx, jyy, m_cov, m_cov, op="sub")      # Jxx-Jyy
    d2 = tmul(d, d)
    xy2 = tmul(jxy, jxy)
    four = const_trip(4.0, shape)
    t4 = tmul(xy2, four)
    disc_acc = binop_fixed(d2, t4, m_acc, m_acc, op="add")
    dq_acc = S.to_fixed(disc_acc[0], disc_acc[1], disc_acc[2], m_acc)
    disc = S.from_fixed(H.rescale_(dq_acc, m_acc, m_cov), m_cov)
    aniso = sqrt_trip(disc)
    trace = binop_fixed(jxx, jyy, m_cov, m_cov, op="add")
    raw = tdiv_trip(aniso, trace)
    tz = trace[2].astype(bool)
    zs, ze, zz = zero_trip(shape, m_cov)
    coh0 = (np.where(tz, zs, raw[0]),
            np.where(tz, ze, raw[1]).astype(np.int32),
            np.where(tz, zz, raw[2]).astype(np.uint8))
    coh = clip_fixed(coh0, 0.0, 1.0, m_cov)
    dq = S.to_fixed(d[0], d[1], d[2], m_cov)
    s2 = binop_fixed(jxy, jxy, m_cov, m_cov, op="add")     # 2*Jxy, exact
    sq = S.to_fixed(s2[0], s2[1], s2[2], m_cov)
    coh_q = S.to_fixed(coh[0], coh[1], coh[2], m_cov)
    thr = COH_THR if coh_thr is None else float(coh_thr)
    tq = S.to_fixed(*const_trip(thr, (1,)), m_cov)[0]
    gate = coh_q >= tq
    # sq IS 2*Jxy already (binop add above): compare |sq| vs |dq| directly.
    # (A revision doubled it to 2*|sq|, silently halving the oriented set.)
    diag_dom = np.abs(sq) > np.abs(dq)
    vert = dq > 0
    diag_pos = sq >= 0  # Jxy>=0 -> /-edge (normal (1,1)); <0 -> \\-edge
    bucket = np.where(~gate, 4,
             np.where(~diag_dom, np.where(vert, 0, 1),
                      np.where(diag_pos, 3, 2))).astype(np.int8)
    return coh, bucket, {"gate_frac": float((bucket == 4).mean()),
                         "diag_frac": float(((bucket == 2) | (bucket == 3)).mean())}


def splat_blur(a_t, m_acc, m_cov, bank_k=None, coh_thr=None, fused=False):
    """Bank blur + exact mux. Returns (out_trip, diag).
    fused=True: single-pass per-pixel kernel gather (same products, same
    integer sums -> bit-identical to compositional; gated == in test_splat).
    Default False (compositional mirrors the C compositional lowering)."""
    if coh_thr is None:
        coh_thr = load_ctrl()["coh_thr"]
    jxx, jyy, jxy, gx, gy = structure(a_t, m_acc, m_cov)
    coh, bucket, audit = coherence_bucket(jxx, jyy, jxy, gx, gy, m_cov, m_acc,
                                          coh_thr=coh_thr)
    if bank_k is None:
        bank_k = bank()
    assert len(bank_k) == 5, "bank is 5-way in v1.1"
    if fused:
        out = _fused_blur(a_t, bank_k, bucket, m_acc, m_cov)
    else:
        outs = [conv_trip(a_t, k, m_acc, m_out=m_cov) for k in bank_k]
        out = H.select_mux(outs, bucket)  # #LIB-009: exact select
    diag = {"bucket": bucket, "gate_frac": audit["gate_frac"],
            "coh": S.decode(coh[0], coh[1]) * (1 - coh[2].astype(np.float64)),
            "coh_t": coh}  # triples for the v2 controller (integer path)
    return out, diag


def _fused_blur(a_t, bank_k, bucket, m_acc, m_cov):
    """Single-pass gathered blur (#LIB-006): per output pixel, accumulate
    taps of ONLY the selected kernel. Same products and same integer sums
    as compositional (integer addition commutes) -> bit-identical output.
    Cost: 25 tap-visits/px vs 8 convs x 25 (8x recovery on the blur stage)."""
    r = bank_k[0].shape[0] // 2
    Hh, Ww = a_t[0].shape

    def pad(x):
        return np.pad(x, ((r, r), (r, r)), mode="edge")
    ps, pe, pz = pad(a_t[0]), pad(a_t[1]), pad(a_t[2])
    # pre-encoded taps per bank entry (same values as conv_trip's encodes)
    banks = [H.kernel_triples(k) for k in bank_k]
    acc = np.zeros((Hh, Ww), dtype=np.int64)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            tap = (ps[dy:dy + Hh, dx:dx + Ww].astype(np.int8),
                   pe[dy:dy + Hh, dx:dx + Ww], pz[dy:dy + Hh, dx:dx + Ww])
            # per-pixel gathered weight triple: ONE tmul per tap (25 total),
            # vs one tmul per tap per bank entry (125) in compositional.
            ws = np.array([banks[b][0][dy, dx] for b in range(5)])[bucket]
            we = np.array([banks[b][1][dy, dx] for b in range(5)])[bucket]
            wz = np.array([banks[b][2][dy, dx] for b in range(5)])[bucket]
            prod = H.tmul(tap, (ws.astype(np.int8), we.astype(np.int32),
                                wz.astype(np.uint8)))
            acc = acc + S.to_fixed(prod[0], prod[1], prod[2], m_acc)
    assert int(np.abs(acc).max(initial=0)) < 2 ** 53
    return S.from_fixed(H.rescale_(acc, m_acc, m_cov), m_cov)
