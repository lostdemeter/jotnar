"""holo_phi: true-amplitude holographic enhancement on the phi lattice.

Philosophy fix vs old src/enhance.py:
  OLD: L_enh = L * (1 + boost*alpha*(L-L_blur)/(L_blur+eps))  (ratio unsharp-mask,
       no sqrt anywhere despite THEORY.md section 3 claiming A=sqrt(I)).
  NEW (this file): A = sqrt(Y) via exponent-halve (lattice-native, exact when
       even), blur in AMPLITUDE domain via integer conv, detail boost in
       amplitude, square back via exact tmul: I_enh = |A_enh|^2.

Pipeline (one direction, no cycles):
  sRGB uint8 --encode boundary--> linear Y float [0,1]
    -> triples (phi_core.encode, offline/float allowed ONLY here)
    -> integer datapath (triples/fixed, no FPU, order-free sums)
    -> triples out --decode boundary--> gain -> RGB linear -> sRGB uint8

IR op mapping (see phi-core IR.md):
  sqrt      : exponent halve (tmul-family, exact-or-1-step)
  conv      : per-product tmul + to_fixed @ m_acc + int64 accumulate
              (order-free) + from_fixed @ m_acc, then rescale_ to m_cov
  binop add/sub : to_fixed @ same m, int64 add/sub, from_fixed (tags asserted;
              caller must rescale_ first -- mismatch fails loud)
  tmul      : sign-XOR + exp-add, zero-or (exact)
  tdiv_trip : sign-XOR + exp-sub (exact, zero-guarded) -- gain computation
  clip      : integer compare in fixed domain (exact by construction)
  rescale_  : the ONLY scale changer (explicit, greppable). Mirrors
              c_core fq_rescale (triples roundtrip). Every call is a
              deliberate unit change, never an implicit cast.

Float appears ONLY at: sensor encode (sRGB->linear, RGB->Y), display decode
(linear->sRGB), LUT builds (offline, frozen to luts/). Runtime loop is
integers + gathers. See fpu_trap pattern in test_core.py.
"""
import json
import os
import numpy as np

import sys
sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
import phi_core.lattice as S

CHAIN_DIR = os.path.dirname(os.path.abspath(__file__))
M_PATH = os.path.join(CHAIN_DIR, "M.json")

AUDIT = {"gain_clip": 0, "gain_n": 0, "y_zero": 0}


def _load_m():
    try:
        with open(M_PATH) as fh:
            d = json.load(fh)
            return int(d.get("m", max(int(d["m_acc"]), int(d["m_cov"]))))
    except Exception:
        # default: covers pmax=1.0 with margin 512 (m_of(1.0)). Frozen by
        # calibrate.py on real frames; this fallback is identical formula.
        return S.BIAS + 512


def _load_scales():
    """Two-scale doctrine: m_acc covers max single PRODUCT (conv taps),
    m_cov covers max |value| wherever triples are decoded. Frozen by
    calibrate.py. Single-m callers must use max (never min -- min clips)."""
    try:
        with open(M_PATH) as fh:
            d = json.load(fh)
            return int(d["m_acc"]), int(d["m_cov"])
    except Exception:
        m = S.BIAS + 512
        return m, m


def rescale_(q, m1, m2):
    """The ONLY scale changer (IR `rescale`: fix@m1 -> fix@m2).
    Implemented as fixed@m1 -> triples -> fixed@m2 (mirrors c_core
    fq_rescale). Triples themselves are scale-free (untagged), so this
    operates on fixed arrays -- never re-encode triples at the small scale
    (that saturates values above its range; see Debt-1 fix log).
    Every unit change calls this; grep `rescale_` to audit all of them."""
    if m1 == m2:
        return q.copy()
    mid = S.from_fixed(q, m1)  # exact decode, no saturation (q < 2^53 gated)
    return S.to_fixed(mid[0], mid[1], mid[2], m2)


def tmul(a, b):
    """Exact triple multiply (sign-XOR + exp-add). Zero-aware."""
    s = (a[0].astype(np.int16) * b[0].astype(np.int16)).astype(np.int8)
    e = np.clip(a[1].astype(np.int64) + b[1].astype(np.int64) - S.BIAS,
                0, S.NLVL - 1).astype(np.int32)
    z = (a[2] | b[2]).astype(np.uint8)
    return s, e, z


def tdiv_pure(a, b):
    """Pure triple divide (sign-XOR + exp-sub, zero-or). No guards: 0/0 stays
    a zero triple (z=1). Callers add their own zero semantics explicitly
    (gain-guard 1.0 in tdiv_trip, flat-guard 0.0 in splat div0-site)."""
    s = (a[0].astype(np.int16) * b[0].astype(np.int16)).astype(np.int8)
    e = np.clip(a[1].astype(np.int64) - b[1].astype(np.int64) + S.BIAS,
                0, S.NLVL - 1).astype(np.int32)
    z = (a[2] | b[2]).astype(np.uint8)
    return s, e, z


def neg_trip(t):
    """Exact sign negate (NOT gate on the sign bit)."""
    return (-t[0].astype(np.int8), t[1].copy(), t[2].copy())


def relu_trip(t, m):
    """Exact relu via fixed-domain max(.,0): integer compare, no FP.
    Continuous (small input changes -> small output changes): the v4
    workhorse for discontinuity-free orientation weights."""
    q = S.to_fixed(t[0], t[1], t[2], m)
    return S.from_fixed(np.maximum(q, 0), m)


def sigmoid_trip(t):
    """IR `sigmoid` opcode (EXPACT gather + LUT, any input range, [0,1]
    output). Mirrors the rife integer sigmoid exactly, in numpy:
    triples -> 2^-14 via sigx gather -> sig LUT over +-16 (asymptotes exact)
    -> from_fixed. Transcendentals live in frozen tables, never in ALU."""
    X = S.sigx_lut()
    G = S.sig_lut()
    SPAN = S.SIG_SPAN
    e = np.clip(t[1].astype(np.int64), 0, 65535)
    x14 = np.where(t[2].astype(bool), 0, t[0].astype(np.int64) * X[e])
    idx = x14 + SPAN
    y14 = np.where(x14 < -SPAN, 0, np.where(
        x14 > SPAN, 16384, G[np.clip(idx, 0, len(G) - 1)]))
    return S.from_fixed(y14 * np.int64(16), S.BIAS)


def select_mux(trips, bucket):
    """#LIB-009 promoted helper: exact select among N triple-streams by an
    integer bucket mask (mirrors prelu_int sign-select). Exact by
    construction: np.where chains on integer masks, no arithmetic."""
    n = len(trips)
    assert n >= 2, "mux needs >= 2 streams"
    s, e, z = trips[-1]
    for i in range(n - 2, -1, -1):
        m = (bucket == i)
        s = np.where(m, trips[i][0], s).astype(np.int8)
        e = np.where(m, trips[i][1], e).astype(np.int32)
        z = np.where(m, trips[i][2], z).astype(np.uint8)
    return s, e, z


def kernel_triples(kernel):
    """Pre-encode every tap (identical values to conv_trip's inline encodes;
    shared with file-exchange so both sides use the same weight triples)."""
    kh, kw = kernel.shape
    ks = np.empty((kh, kw), np.int8)
    ke = np.empty((kh, kw), np.int32)
    kz = np.empty((kh, kw), np.uint8)
    for dy in range(kh):
        for dx in range(kw):
            ws, we, wz = S.encode(np.array([float(kernel[dy, dx])]))
            ks[dy, dx], ke[dy, dx], kz[dy, dx] = ws[0], we[0], wz[0]
    return ks, ke, kz


def tdiv_trip(a, b):
    """Exact triple divide (sign-XOR + exp-sub). Zero-guarded: where b is
    zero-ish (z flag or tiny), result forced to 1.0 triple (gain no-op)."""
    s = (a[0].astype(np.int16) * b[0].astype(np.int16)).astype(np.int8)
    e = np.clip(a[1].astype(np.int64) - b[1].astype(np.int64) + S.BIAS,
                0, S.NLVL - 1).astype(np.int32)
    z = (a[2] | b[2]).astype(np.uint8)
    # guard: divisor zero flag -> gain 1.0
    one = S.encode(np.ones(a[0].shape, dtype=np.float64))
    s = np.where(b[2].astype(bool), one[0], s)
    e = np.where(b[2].astype(bool), one[1], e).astype(np.int32)
    z = np.where(b[2].astype(bool), one[2], z)
    return s, e, z


def sqrt_trip(t):
    """A = sqrt(Y): lattice-native exponent halve. Y is non-negative so
    output sign forced +1. Exact when (e-BIAS) even, else 1-step rounding
    (~0.047% at K=512). THIS is the op the old code never implemented."""
    e = ((t[1].astype(np.int64) - S.BIAS) // 2 + S.BIAS).clip(0, S.NLVL - 1).astype(np.int32)
    s = np.ones(t[0].shape, dtype=np.int8)
    z = t[2].astype(np.uint8)
    return s, e, z


def binop_fixed(a, b, m_a, m_b=None, op="add"):
    """Fixed-domain add/sub with tag assert (both inputs @ same m).
    Pass m_a and m_b explicitly -- mismatch raises instead of silently
    mixing units. Caller must rescale_ first (greppable)."""
    if m_b is None:
        m_b = m_a
    assert m_a == m_b, f"unit confusion: {m_a} vs {m_b} -- rescale_ explicitly"
    qa = S.to_fixed(a[0], a[1], a[2], m_a)
    qb = S.to_fixed(b[0], b[1], b[2], m_b)
    q = qa + qb if op == "add" else qa - qb
    return S.from_fixed(q, m_a)


def clip_fixed(t, lo, hi, m):
    """Exact clip in fixed domain (integer compare, no float)."""
    q = S.to_fixed(t[0], t[1], t[2], m)
    qs, qe, qz = S.encode(np.array([lo, hi], dtype=np.float64))
    qlo = S.to_fixed(qs[0:1], qe[0:1], qz[0:1], m)[0]
    qhi = S.to_fixed(qs[1:2], qe[1:2], qz[1:2], m)[0]
    return S.from_fixed(np.clip(q, qlo, qhi), m)


def gaussian_kernel(radius=2, sigma=1.0):
    """Float kernel built OFFLINE (allowed). Quantized only via triples at
    runtime (each tap encoded, tmul exact). Normalized sum 1."""
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    k1 = np.exp(-0.5 * (ax / sigma) ** 2)
    k1 /= k1.sum()
    return np.outer(k1, k1)


def conv_trip(img_trip, kernel, m_acc, m_out=None):
    """IR conv: per-tap tmul + to_fixed @ m_acc + int64 accumulate
    (order-free) + from_fixed @ m_acc, then rescale_ to m_out.
    Edge: replicate (matches nihui-exact warp convention).
    m_acc covers max single product; m_out covers values (usually m_cov)."""
    if m_out is None:
        m_out = m_acc
    r = kernel.shape[0] // 2
    H, W = img_trip[0].shape
    # edge-replicate pad each triple plane
    def pad(x):
        return np.pad(x, ((r, r), (r, r)), mode="edge")
    ps, pe, pz = pad(img_trip[0]), pad(img_trip[1]), pad(img_trip[2])
    ks, ke, kz = kernel_triples(kernel)  # identical to inline encodes
    acc = None
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            if kernel[dy, dx] == 0:
                continue
            tap = (ps[dy:dy + H, dx:dx + W].astype(np.int8) * 1,
                   pe[dy:dy + H, dx:dx + W], pz[dy:dy + H, dx:dx + W])
            # broadcast scalar weight triple
            prod = tmul(tap, (np.full((H, W), ks[dy, dx], np.int8),
                              np.full((H, W), ke[dy, dx], np.int32),
                              np.full((H, W), kz[dy, dx], np.uint8)))
            q = S.to_fixed(prod[0], prod[1], prod[2], m_acc)
            acc = q if acc is None else acc + q  # int64, order-free
    assert acc is not None
    assert int(np.abs(acc).max(initial=0)) < 2 ** 53, "accum exceeds frexp-exact range"
    # explicit unit change: accum fixed@m_acc -> fixed@m_out (never implicit)
    return S.from_fixed(rescale_(acc, m_acc, m_out), m_out)


def alpha_lut():
    """DEPRECATED ablation only (Debt 2 verdict: REMOVE).
    Old-code parabola alpha(u)=clip(4u(1-u)+0.3,0.3,1) suppressed shadows and
    highlights to hide noise amplification and clipping -- but the amplitude
    domain already handles both honestly: detail D=A-As scales as
    D_Y/(2*sqrt(Y)) so first-order intensity enhancement is uniform, while
    shadows are protected by the integer gain clip [0.5,2.0] and highlights
    by the output clip [0,1] (both declared, both integer, both audited).
    Kept solely for parity ablation (use_alpha=True); default is False.
    Runtime use is integer gather only when enabled."""
    p = os.path.join(CHAIN_DIR, "..", "luts", "alpha_lut.npy")
    try:
        return np.load(p)
    except Exception:
        u = np.arange(256, dtype=np.float64) / 255.0
        a = np.clip(4 * u * (1 - u) + 0.3, 0.3, 1.0)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, a)
        return a


def alpha_gather(y_lin, lut=None):
    """Integer gather of per-pixel alpha triples (no float compute)."""
    lut = alpha_lut() if lut is None else lut
    idx = np.clip((y_lin * 255.0).astype(np.int64), 0, 255)
    a = lut[idx]  # gather
    return S.encode(a)


def enhance_luminance_int(y_lin, beta=0.5, kernel=None, m=None, use_alpha=False,
                          m_acc=None, m_cov=None, blur="iso", ctrl=False,
                          ctrl_atten=None, coh_thr=None, ctrl_mid=None,
                          coh_hi=None, depth=None, ctrl_strong=None):
    """Integer datapath: Y [0,1] float (boundary) -> Y_enh triples + audits.
    Math actually executed: A=sqrt(Y); As=blur(A); D=A-As;
    Aenh=A+beta_eff*D (beta_eff scalar, v2 controller field, and/or depth
    multiplier -- modulations compose multiplicatively); Ienh=Aenh^2.
    blur/ctrl/depth: docs/SPLAT_OP.md, docs/BETA_CTRL.md, docs/COMPOSE.md.
    depth=None off, else float depth map at Y geometry (offline prior).
    Overrides are search-only; scales as documented below."""
    if m_acc is None or m_cov is None:
        if m is not None:
            m_acc = m_cov = m
        else:
            m_acc, m_cov = _load_scales()
    if kernel is None:
        kernel = gaussian_kernel()
    H, W = y_lin.shape
    y_t = S.encode(np.ascontiguousarray(y_lin, dtype=np.float64))
    a_t = sqrt_trip(y_t)
    diag = None
    if blur in ("splat", "splat_soft"):
        from chain.splat import splat_blur
        from chain.control import load_ctrl
        cthr = coh_thr if coh_thr is not None else load_ctrl()["coh_thr"]
        as_t, diag = splat_blur(a_t, m_acc, m_cov, coh_thr=cthr,
                                soft=(blur == "splat_soft"))
    elif blur == "iso":
        as_t = conv_trip(a_t, kernel, m_acc, m_out=m_cov)
    else:
        raise ValueError(f"blur must be 'iso'|'splat'|'splat_soft', got {blur!r}")
    d_t = binop_fixed(a_t, as_t, m_cov, m_cov, op="sub")
    if ctrl == "soft":
        if blur not in ("splat", "splat_soft") or diag is None:
            raise ValueError("ctrl='soft' needs a splat blur (coh source)")
        from chain.control import beta_field_soft
        ds_t = tmul(d_t, beta_field_soft(beta, diag["coh_t"], m_cov))
    elif ctrl:
        if blur not in ("splat", "splat_soft") or diag is None:
            raise ValueError("ctrl=True needs a splat blur (bucket source)")
        from chain.control import beta_field
        ds_t = tmul(d_t, beta_field(beta, diag["bucket"], diag["coh_t"],
                                    m_cov, atten=ctrl_atten, mid=ctrl_mid,
                                    hi=coh_hi, strong=ctrl_strong))
    else:
        b_t = S.encode(np.full((H, W), float(beta), dtype=np.float64))
        ds_t = tmul(d_t, b_t)
    if use_alpha:
        al_t = alpha_gather(y_lin)  # DEPRECATED ablation only (see Debt 2)
        ds_t = tmul(ds_t, al_t)
    depth_frac = None
    if depth is not None:
        from chain.depthprior import near_mask, depth_mult
        depth = np.ascontiguousarray(depth, dtype=np.float64)
        assert depth.shape == (H, W), f"depth geometry {depth.shape} vs Y {(H, W)}"
        near = near_mask(depth)
        depth_frac = float(near.mean())
        ds_t = tmul(ds_t, depth_mult((H, W), near))
    aenh_t = binop_fixed(a_t, ds_t, m_cov, m_cov, op="add")
    ienh_t = tmul(aenh_t, aenh_t)  # I = |A|^2
    ienh_t = clip_fixed(ienh_t, 0.0, 1.0, m_cov)
    info = {"m_acc": m_acc, "m_cov": m_cov, "shape": (H, W),
            "alpha": bool(use_alpha), "blur": blur, "ctrl": bool(ctrl),
            "depth": depth is not None}
    if diag is not None:
        info["gate_frac"] = diag["gate_frac"]
    if depth_frac is not None:
        info["near_frac"] = depth_frac
    return ienh_t, info


def apply_gain_int(rgb_lin, y_lin, yenh_trip, m=None, m_cov=None):
    """Chroma-preserving recombine, integer: g = Yenh/Y (triple div),
    Cout = Cin * g (tmul exact). Gain clipped [0.5,2.0] integer. Returns
    float32 linear RGB (display-boundary decode).
    Boundary contract: rgb_lin/y_lin are linear-light [0,1] floats produced
    at the sensor-encode boundary (sRGB->linear + Rec.709 luma). Hue
    (R:G:B ratios) is preserved by construction -- only the amplitude
    (gain) changes, which is the holographic phase-preservation claim,
    gated by hue-preservation in test_parity.py."""
    if m_cov is None:
        m_cov = m if m is not None else _load_scales()[1]
    m = m_cov
    y_t = S.encode(np.ascontiguousarray(y_lin, dtype=np.float64))
    g_t = tdiv_trip(yenh_trip, y_t)
    g_t = clip_fixed(g_t, 0.5, 2.0, m)
    q = S.to_fixed(g_t[0], g_t[1], g_t[2], m)
    AUDIT["gain_n"] += int(q.size)
    lo, hi = S.to_fixed(*S.encode(np.array([0.5], np.float64)), m)[0], None
    # count clips via integer compare (no float)
    qs = S.to_fixed(*S.encode(np.array([0.5, 2.0])), m)
    AUDIT["gain_clip"] += int(((q <= qs[0]).sum() + (q >= qs[1]).sum()))
    AUDIT["y_zero"] += int((y_lin <= 1e-12).sum())
    H, W, _ = rgb_lin.shape
    out = np.empty_like(rgb_lin, dtype=np.float32)
    for c in range(3):
        ct = S.encode(np.ascontiguousarray(rgb_lin[:, :, c], dtype=np.float64))
        et = tmul(ct, g_t)
        out[:, :, c] = S.decode(et[0], et[1]).astype(np.float32) * (1 - et[2].astype(np.float32))
    return np.clip(out, 0, 1).astype(np.float32)


def enhance_image_int(rgb_lin, beta=0.5, sigma=1.0, radius=2, use_alpha=False,
                      m=None, m_acc=None, m_cov=None, blur="iso", ctrl=False,
                      ctrl_atten=None, coh_thr=None, ctrl_mid=None,
                      coh_hi=None, depth=None, ctrl_strong=None):
    """End-to-end integer chain on linear-light RGB float32 [0,1].
    Boundary float in/out; everything between is triples/fixed."""
    if m_acc is None or m_cov is None:
        if m is not None:
            m_acc = m_cov = m
        else:
            m_acc, m_cov = _load_scales()
    kernel = gaussian_kernel(radius=radius, sigma=sigma)
    # luminance at encode boundary (float allowed here; weights sum to 1.0,
    # fingerprinted in test_core test_luma_weights; Y in [0,1] asserted there)
    y = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
         + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)
    yenh_t, info = enhance_luminance_int(y, beta=beta, kernel=kernel,
                                         m_acc=m_acc, m_cov=m_cov,
                                         use_alpha=use_alpha, blur=blur,
                                         ctrl=ctrl, ctrl_atten=ctrl_atten,
                                         coh_thr=coh_thr, ctrl_mid=ctrl_mid,
                                         coh_hi=coh_hi, depth=depth,
                                         ctrl_strong=ctrl_strong)
    rgb_enh = apply_gain_int(rgb_lin.astype(np.float32), y, yenh_t, m_cov=m_cov)
    yenh = S.decode(yenh_t[0], yenh_t[1]) * (1 - yenh_t[2].astype(np.float64))
    return rgb_enh, np.clip(yenh, 0, 1), info
