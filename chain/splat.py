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
                            tdiv_trip, tdiv_pure, clip_fixed, neg_trip,
                            relu_trip)

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


def coherence_bucket(jxx, jyy, jxy, gx, gy, m_cov, m_acc, coh_thr=None,
                     flow=None):
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
    coh = aniso/trace with 0/0 -> 0. gx/gy args kept for signature (unused).

    flow (side-channel consensus, chain/motion.py): None (default, legacy
    behavior, bit-identical) or (perp int8 (H,W), norm float [0,1] (H,W),
    static bool (H,W)) from motion.quantize. Consensus rule -- flow CONFIRMS
    or VETOES the tensor claim, never originates one (aperture problem):
      eff = max(coh, norm); gate = eff >= thr   (motion explains weak coh)
      strong = coh >= hi                         (tensor's own testimony)
      agree = (tensor_bucket == perp)
      final = gate AND (strong OR agree)
    Static pixels take the direct path bit-exactly (integer select on the
    static mask -- the consensus roundtrip may 1-step-perturb near-zero
    coherence, and static pixels must never pay that).
    Returns (coh, final_bucket, audit) where audit also carries flow_scale
    triples (1-(1-FLOW_ATTEN)*norm: beta caution on fast regions, ones on
    static) and the raw tensor bucket (for gates). Zero flow fields reproduce
    blind EXACTLY (gated); motion=None skips the branch (also exact)."""
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
    # direction ALWAYS assigned (gate applied separately): consensus needs the
    # would-be direction of gate-shut pixels (a blind-iso pixel carries no
    # direction, so gating first would let flow only ever REMOVE orientation).
    # Reorder only -- blind path below reproduces the legacy expression exactly.
    direction = np.where(~diag_dom, np.where(vert, 0, 1),
                         np.where(diag_pos, 3, 2)).astype(np.int8)
    bucket = np.where(~gate, 4, direction).astype(np.int8)
    # soft-blend inputs (v4): the signed triples relu splits on. Returned via
    # audit (documented); the hard path ignores them.
    out_audit = {"gate_frac": float((bucket == 4).mean()),
                 "diag_frac": float(((bucket == 2) | (bucket == 3)).mean()),
                 "d": d, "s2": s2, "tensor_bucket": bucket.copy()}
    if flow is None:
        return coh, bucket, out_audit
    # ---- consensus branch (side-channel; flow is not None) ----
    from chain.control import load_ctrl as _lc
    from chain.motion import FLOW_ATTEN
    perp, norm_f, static_m = flow
    assert perp.shape == shape, f"flow geometry {perp.shape} vs {shape}"
    hi = _lc().get("coh_hi", 0.4)
    norm_t = S.encode(np.ascontiguousarray(norm_f, dtype=np.float64))
    # eff = max(coh, norm) in fixed domain (integer max, exact)
    q_coh = S.to_fixed(coh[0], coh[1], coh[2], m_cov)
    q_norm = S.to_fixed(norm_t[0], norm_t[1], norm_t[2], m_cov)
    eff = S.from_fixed(np.maximum(q_coh, q_norm), m_cov)
    eff_q = S.to_fixed(eff[0], eff[1], eff[2], m_cov)
    # flow only OPENS (blind gate OR-ed in): the eff roundtrip
    # (fixed max + rescale) may 1-step-perturb near-zero coherence, and must
    # never shut a blind-open pixel. Vetoes below are the design working
    # (weak tensor + disagreeing flow = genuine uncertainty -> iso), not noise.
    gate_f = gate | (eff_q >= tq)
    hq = S.to_fixed(*const_trip(hi, (1,)), m_cov)[0]
    strong = q_coh >= hq
    # agree against the DIRECTION (not the gated bucket): gate-shut pixels
    # have no direction in bucket (iso), so agreement must read direction.
    agree = (direction == np.ascontiguousarray(perp, dtype=np.int8))
    opened = np.where(~gate_f, 4,
                      np.where(strong | agree, direction, 4)).astype(np.int8)
    final = np.where(static_m, bucket, opened).astype(np.int8)
    # flow_scale = 1-(1-FLOW_ATTEN)*norm; exact ones on static (tmul by
    # encode(1.0) is the identity: exp-add of BIAS-BIAS, sign*1 -- exact)
    ka = S.encode(np.full(shape, 1.0 - FLOW_ATTEN, dtype=np.float64))
    one = S.encode(np.ones(shape, dtype=np.float64))
    dec = binop_fixed(one, tmul(ka, norm_t), m_cov, m_cov, op="sub")
    flow_scale = (np.where(static_m, one[0], dec[0]).astype(np.int8),
                  np.where(static_m, one[1], dec[1]).astype(np.int32),
                  np.where(static_m, one[2], dec[2]).astype(np.uint8))
    out_audit["gate_frac"] = float((final == 4).mean())
    out_audit["flow_scale"] = flow_scale
    out_audit["tensor_bucket"] = bucket
    return coh, final, out_audit


def splat_blur(a_t, m_acc, m_cov, bank_k=None, coh_thr=None, fused=False,
               soft=False, flow=None):
    """Bank blur + exact mux. Returns (out_trip, diag).
    fused=True: single-pass per-pixel kernel gather (bit-identical).
    soft=True (v4): relu-weighted BLEND of all 5 bank outputs instead of the
    mux -- continuous in the tensor (no bucket flips anywhere). buckets still
    computed for diag/audit + the v3 beta path. Default False (hard select
    mirrors the C lowering).
    flow: None (blind, legacy) or motion.quantize triple for side-channel
    consensus (buckets) -- diag carries flow_scale triples for beta caution
    (applied by the caller, like depth)."""
    if coh_thr is None:
        coh_thr = load_ctrl()["coh_thr"]
    jxx, jyy, jxy, gx, gy = structure(a_t, m_acc, m_cov)
    coh, bucket, audit = coherence_bucket(jxx, jyy, jxy, gx, gy, m_cov, m_acc,
                                          coh_thr=coh_thr, flow=flow)
    if bank_k is None:
        bank_k = bank()
    assert len(bank_k) == 5, "bank is 5-way in v1.1"
    if fused:
        out = _fused_blur(a_t, bank_k, bucket, m_acc, m_cov)
    elif soft:
        outs = [conv_trip(a_t, k, m_acc, m_out=m_cov) for k in bank_k]
        out = _soft_blend(outs, audit["d"], audit["s2"], m_cov)
    else:
        outs = [conv_trip(a_t, k, m_acc, m_out=m_cov) for k in bank_k]
        out = H.select_mux(outs, bucket)  # #LIB-009: exact select
    diag = {"bucket": bucket, "gate_frac": audit["gate_frac"],
            "coh": S.decode(coh[0], coh[1]) * (1 - coh[2].astype(np.float64)),
            "coh_t": coh}  # triples for the v2 controller (integer path)
    if "flow_scale" in audit:
        diag["flow_scale"] = audit["flow_scale"]
    if "tensor_bucket" in audit:
        diag["tensor_bucket"] = audit["tensor_bucket"]
    return out, diag


def _soft_blend(outs, d, s2, m_cov):
    """v4 relu-weighted blend (#LIB-014): orientation weights from the signed
    tensor triples, continuous everywhere (relu is continuous; the only exact
    select left is the true-flat guard, invisible since D~=0 there):
      rV=relu(d), rH=relu(-d), rD1=relu(s2), rD2=relu(-s2); w=r/sum;
      out = sum(w*blur). All triples/integers; division exact (exp-sub).
    Diagonal mapping (a swap here shipped once, caught by the rotation gate
    -- parity CANNOT catch it, both sides mirrored: rD1 is s2>0 = slash edge
    = outs[3] [rotated(-45)], rD2 is s2<0 = backslash = outs[2]. Cross-check
    against coherence_bucket's diag_pos rule, not against the oracle."""
    rV = relu_trip(d, m_cov)
    rH = relu_trip(neg_trip(d), m_cov)
    rD1 = relu_trip(s2, m_cov)
    rD2 = relu_trip(neg_trip(s2), m_cov)
    den = binop_fixed(binop_fixed(rV, rH, m_cov, m_cov, op="add"),
                      binop_fixed(rD1, rD2, m_cov, m_cov, op="add"),
                      m_cov, m_cov, op="add")
    wV = tdiv_pure(rV, den)
    wH = tdiv_pure(rH, den)
    wD1 = tdiv_pure(rD1, den)
    wD2 = tdiv_pure(rD2, den)
    acc = binop_fixed(tmul(wV, outs[0]), tmul(wH, outs[1]),
                      m_cov, m_cov, op="add")
    acc = binop_fixed(acc, tmul(wD1, outs[3]), m_cov, m_cov, op="add")
    acc = binop_fixed(acc, tmul(wD2, outs[2]), m_cov, m_cov, op="add")
    # true-flat guard (den z-flagged): exact iso, bit-clean flats
    dz = den[2].astype(bool)
    iso = outs[4]
    return (np.where(dz, iso[0], acc[0]).astype(np.int8),
            np.where(dz, iso[1], acc[1]).astype(np.int32),
            np.where(dz, iso[2], acc[2]).astype(np.uint8))


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
