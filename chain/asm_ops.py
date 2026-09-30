"""Assembly ops registry (stage 1): mnemonics -> chain functions.

Each op is a thin, faithful wrapper: no new math, no hidden state. Scales
come from frozen files, beta from CONFIG. v1 covers exactly the flagship
(splat_soft + v5); new structures add mnemonics here (stage-2 micro-AI will
reuse these plus a few new ones -- hopefully ZERO new ones, reusing only
existing mnemonics in a novel listing).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phi_core.lattice as S
from chain import holo_phi as H
from chain import splat as SP
from chain import control as C


def _scales():
    return H._load_scales()


def _beta(config):
    return float(config.get("beta", 0.5))


def op_srgb_decode(vals, config, feeds):
    (x,) = vals
    return np.power(np.clip(np.asarray(x, dtype=np.float64) / 255.0, 0, 1),
                    2.2).astype(np.float32)


def op_luma(vals, config, feeds):
    (lin,) = vals
    lin = np.asarray(lin, dtype=np.float64)
    return (0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1]
            + 0.0722 * lin[:, :, 2])


def op_sqrt(vals, config, feeds):
    (y,) = vals
    return H.sqrt_trip(S.encode(np.ascontiguousarray(y, dtype=np.float64)))


def op_splat_blur(vals, config, feeds):
    """Soft blend (flagship default; hard/fused variants are backlog
    mnemonics, not silent flags). Returns (AS triples, COH triples)."""
    (a,) = vals
    m_acc, m_cov = _scales()
    coh_thr = C.load_ctrl()["coh_thr"]
    out, diag = SP.splat_blur(a, m_acc, m_cov, coh_thr=coh_thr, soft=True)
    return out, diag["coh_t"]


def _need_triples(t, op, name):
    """Triple-ness assertion for the arithmetic core: triple-ops receiving
    float arrays silently compute garbage (caught by gate). Layout-kind
    checks cover DECLARED streams; this covers UNANNOTATED ones (gradual
    typing must still refuse to compute nonsense). Systematic coverage of
    every triple-op is backlog; core first, stated."""
    import numpy as _np
    ok = (isinstance(t, tuple) and len(t) == 3
          and all(isinstance(x, _np.ndarray) for x in t))
    if not ok:
        raise ValueError(f"{op}: stream '{name}' must be triples, got {type(t)}")


def op_sub(vals, config, feeds):
    a, b = vals
    _need_triples(a, "SUB", "a")
    _need_triples(b, "SUB", "b")
    _, m_cov = _scales()
    return H.binop_fixed(a, b, m_cov, m_cov, op="sub")


def op_beta_v5(vals, config, feeds):
    d, coh = vals
    _, m_cov = _scales()
    return C.beta_field_v5(_beta(config), coh, d, m_cov)


def op_mul(vals, config, feeds):
    _need_triples(vals[0], "MUL", "a")
    _need_triples(vals[1], "MUL", "b")
    return H.tmul(vals[0], vals[1])


def op_add(vals, config, feeds):
    _need_triples(vals[0], "ADD", "a")
    _need_triples(vals[1], "ADD", "b")
    _, m_cov = _scales()
    return H.binop_fixed(vals[0], vals[1], m_cov, m_cov, op="add")


def op_square(vals, config, feeds):
    (a,) = vals
    _need_triples(a, "SQUARE", "a")
    _, m_cov = _scales()
    return H.clip_fixed(H.tmul(a, a), 0.0, 1.0, m_cov)


def op_gain(vals, config, feeds):
    lin, y, yenh = vals
    return H.apply_gain_int(np.ascontiguousarray(lin, dtype=np.float32),
                            np.ascontiguousarray(y, dtype=np.float64), yenh)


def op_beta(vals, config, feeds):
    """Beta triples from CONFIG at the reference stream's shape (keeps the
    value frozen-named, never a magic literal in listings). Takes the
    reference triples as input purely for geometry."""
    (ref,) = vals
    import phi_core.lattice as S
    return S.encode(np.full(ref[0].shape, _beta(config), dtype=np.float64))


def op_srgb_encode(vals, config, feeds):
    (lin,) = vals
    return np.clip(np.power(np.clip(lin, 0, 1), 1.0 / 2.2) * 255.0,
                   0, 255).astype(np.uint8)


def op_iso_blur(vals, config, feeds):
    """Wide gaussian structure extraction (denoising wants smoothing over
    orientation analysis; no tensor, no buckets)."""
    (a,) = vals
    m_acc, m_cov = _scales()
    return H.conv_trip(a, H.gaussian_kernel(), m_acc, m_out=m_cov)


def op_gauss(vals, config, feeds):
    """Gaussian blur with radius/sigma structural literals (covers smoothing
    needs beyond the fixed ISO kernel -- e.g. tensor regularization r1s0.8).
    Normalized kernel: preserves value range (the estimator relies on this)."""
    from chain.holo_phi import gaussian_kernel as _gk, conv_trip as _ct
    (a,) = vals[:1]
    r = _int_arg(vals[1], "GAUSS radius")
    s = float(vals[2])
    m_acc, m_cov = _scales()
    return _ct(a, _gk(radius=r, sigma=s), m_acc, m_out=m_cov)


def op_warp(vals, config, feeds):
    """Warp detail triples by float flow. dprev=None passes through as None
    (feed convention: no history on frame 0; MIXDYAD handles None)."""
    from chain.temporal import warp_trips
    dprev, flow = vals
    if dprev is None:
        return None
    _, m_cov = _scales()
    return warp_trips(dprev, np.ascontiguousarray(flow, dtype=np.float64),
                      m_cov)


def op_static(vals, config, feeds):
    """Flow -> exact static mask (verdict: |flow| == 0)."""
    from chain.verdict import verdict_mask
    (flow,) = vals
    flow = np.ascontiguousarray(flow, dtype=np.float64)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    return verdict_mask(mag, 0.0, "==")


def _phi_ops():
    """phi-core numpy ops (lazy import: keeps holo import light)."""
    import sys as _sys
    _sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
    from phi_core import numpy_ops as _N
    from phi_core import lattice as _S
    return _N, _S


def op_matmul(vals, config, feeds):
    """Batched triples matmul (phi-core matmul_int, 0-diff: wrapper adds
    nothing). m_acc from frozen scales."""
    N, _S = _phi_ops()
    m_acc, _ = _scales()
    return N.matmul_int(vals[0], vals[1], m_acc)


def op_softmax(vals, config, feeds):
    """N-way softmax to probability triples. phi-core softmaxN gives
    (num 2^24, den); the num/den -> triples normalization (tdiv to 2^-18
    counts + from_fixed @ BIAS, overflow-asserted) is authored here (mini-bar:
    gated vs float softmax below) -- the one non-mechanical step in Batch 1,
    stated not hidden."""
    N, S = _phi_ops()
    from phi_core.numpy_ops import _assert_bound
    (t,) = vals
    num, den = N.softmaxN_triples(t)
    _assert_bound("asm-softmax:num", num)
    assert int(np.asarray(num).max(initial=0)) < (1 << 40), "softmax num overflows 2^18 shift"
    # den broadcasts over the LAST axis (one denominator per row): reshape
    # explicitly -- relying on numpy trailing broadcast here divided rows by
    # columns (shipped once, rotation-style scramble caught by the gate below).
    den_b = np.asarray(den).reshape(np.asarray(den).shape + (1,) * (np.asarray(num).ndim - np.asarray(den).ndim))
    c18 = S.tdiv(np.asarray(num) * np.int64(1 << 18), np.where(den_b == 0, 1, den_b))
    return S.from_fixed(c18, S.BIAS)


def op_rmsnorm(vals, config, feeds):
    """Per-row RMSNorm+weight (phi-core rmsnorm_int, 0-diff). eps_c frozen via
    CONFIG eps_rms_c (default 4514, the promoted-test convention @ m_of(3.0)
    regime) -- scale-regime-dependent by nature; per-model eps calibration is
    backlog, stated here not hidden."""
    N, _S = _phi_ops()
    x, w = vals
    _, m_cov = _scales()
    eps_c = int(config.get("eps_rms_c", 4514))
    return N.rmsnorm_int(x[0], x[1], x[2], w, m_cov, eps_c)


def op_silu(vals, config, feeds):
    """SiLU x*sigmoid(x) (phi-core silu_int, 0-diff)."""
    N, _S = _phi_ops()
    (t,) = vals
    return N.silu_int(t)


def op_gelu(vals, config, feeds):
    """GELU exact-form x*Phi(x) (phi-core gelu_erf_int: EXPACT + PHI LUT,
    any input range, 0-diff). v1.0 Gate 5 drill mnemonic: first stranger-
    supplied structure (phi-core ASM_HANDOFF.md), thin wrapper, no new math.
    GELU_SPAN asymptotes (x<=-16 -> 0, x>=16 -> x) are exact by construction."""
    N, _S = _phi_ops()
    (t,) = vals
    return N.gelu_erf_int(t)


_rope_cache = {}


def rope_tables(max_pos, dim, base=10000.0):
    """Frozen sin/cos tables (max_pos, dim/2), deterministic build (same
    load-or-freeze discipline as kernels: formula is the source of truth).
    theta_i = base^(-2i/dim), angle = pos * theta_i."""
    key = (int(max_pos), int(dim), float(base))
    if key not in _rope_cache:
        i = np.arange(dim // 2, dtype=np.float64)
        theta = np.power(float(base), -2.0 * i / dim)
        pos = np.arange(int(max_pos), dtype=np.float64)[:, None]
        ang = pos * theta[None, :]
        _rope_cache[key] = (np.cos(ang), np.sin(ang))
    return _rope_cache[key]


def op_rotary(vals, config, feeds):
    """RoPE rotation (first genuinely-new structure at assembly level):
    frozen sin/cos tables + 4 tmuls + 2 binops per pair -- all existing ops,
    no new C needed (composition). X triples (..., D) with even D; POS int
    array broadcastable to X.shape[:-1] (positions are structural metadata,
    carried as integers until TYPED STREAMS formalizes them).
    out[2i] = x[2i]*c - x[2i+1]*s; out[2i+1] = x[2i]*s + x[2i+1]*c.
    base frozen via CONFIG rope_base (default 10000.0)."""
    from chain.holo_phi import tmul, binop_fixed
    x, pos = vals
    _, m_cov = _scales()
    base = float(config.get("rope_base", 10000.0))
    D = x[0].shape[-1]
    assert D % 2 == 0, f"head dim must be even, got {D}"
    P = int(np.ascontiguousarray(pos, dtype=np.int64).max(initial=0)) + 1
    cos_t, sin_t = rope_tables(P, D, base)
    pidx = np.ascontiguousarray(pos, dtype=np.int64)
    c = S.encode(cos_t[pidx])
    s = S.encode(sin_t[pidx])
    x0 = (x[0][..., 0::2], x[1][..., 0::2], x[2][..., 0::2])
    x1 = (x[0][..., 1::2], x[1][..., 1::2], x[2][..., 1::2])
    y0 = binop_fixed(tmul(x0, c), tmul(x1, s), m_cov, m_cov, op="sub")
    y1 = binop_fixed(tmul(x0, s), tmul(x1, c), m_cov, m_cov, op="add")
    out_s = np.empty_like(x[0])
    out_e = np.empty_like(x[1])
    out_z = np.empty_like(x[2])
    out_s[..., 0::2], out_s[..., 1::2] = y0[0], y1[0]
    out_e[..., 0::2], out_e[..., 1::2] = y0[1], y1[1]
    out_z[..., 0::2], out_z[..., 1::2] = y0[2], y1[2]
    return out_s.astype(np.int8), out_e.astype(np.int32), out_z.astype(np.uint8)


def op_batch_matmul(vals, config, feeds):
    """Batched triples matmul (phi-core matmul_int handles batch dims +
    B-broadcast natively -- verified shapes (2,3,5); 0-diff: wrapper adds
    nothing). m_acc from frozen scales."""
    N, _S = _phi_ops()
    m_acc, _ = _scales()
    return N.matmul_int(vals[0], vals[1], m_acc)


def op_transpose(vals, config, feeds):
    """Last-two-axes swap (IR move family). Exact: no arithmetic, only layout.
    Needed wherever scores need K^T (attention) -- the probe demanded it."""
    (t,) = vals
    return (np.swapaxes(t[0], -1, -2).astype(np.int8, copy=False),
            np.swapaxes(t[1], -1, -2).astype(np.int32, copy=False),
            np.swapaxes(t[2], -1, -2).astype(np.uint8, copy=False))


def _int_arg(v, what):
    """Structural literal: must be integral (fail loud on 1.5, never silent
    truncation -- shape bugs must not hide behind float casting)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: shape literal must be integral, got {v!r}")
    if not f.is_integer():
        raise ValueError(f"{what}: shape literal must be integral, got {v!r}")
    return int(f)


def op_reshape2(vals, config, feeds):
    """Reshape any stream to 2D (d0, d1) literals. Exact move; element count
    must match (fail loud). Covers flatten + head-merge; 4D+ needs RESHAPEN
    (backlog, stated -- no demand yet)."""
    (t,) = vals[:1]
    d0, d1 = (_int_arg(vals[1], "RESHAPE2 d0"), _int_arg(vals[2], "RESHAPE2 d1"))
    n = t[0].size
    if d0 * d1 != n:
        raise ValueError(f"RESHAPE2: {d0}x{d1}={d0*d1} != {n} elements")
    return (t[0].reshape(d0, d1), t[1].reshape(d0, d1), t[2].reshape(d0, d1))


def op_reshape3(vals, config, feeds):
    """Reshape any stream to 3D (d0, d1, d2) literals. Exact move; count must
    match. Covers unflatten-to-heads (S,H,Dh)."""
    (t,) = vals[:1]
    d = [_int_arg(v, f"RESHAPE3 d{i}") for i, v in enumerate(vals[1:4])]
    n = t[0].size
    if d[0] * d[1] * d[2] != n:
        raise ValueError(f"RESHAPE3: {d} != {n} elements")
    return (t[0].reshape(*d), t[1].reshape(*d), t[2].reshape(*d))


def op_permute3(vals, config, feeds):
    """Reorder axes of a 3D stream (o0, o1, o2) literals, a permutation of
    (0,1,2). Exact move; covers seq<->heads layout swaps."""
    (t,) = vals[:1]
    o = [_int_arg(v, f"PERMUTE3 o{i}") for i, v in enumerate(vals[1:4])]
    if sorted(o) != [0, 1, 2] or t[0].ndim != 3:
        raise ValueError(f"PERMUTE3: need a permutation of (0,1,2) on 3D input, got {o} ndim={t[0].ndim}")
    return (np.transpose(t[0], o), np.transpose(t[1], o), np.transpose(t[2], o))


def op_select(vals, config, feeds):
    """General verdict-gated branch: per-element pick of A/B by bool MASK.
    Exact (np.where chains on integer masks, no arithmetic) -- generalizes
    MIXDYAD's hardcoded pattern and select_mux to arbitrary streams.
    Layouts: mask I anything, both branches same $A."""
    m, a, b = vals
    m = np.ascontiguousarray(m, dtype=bool)
    if not (m.shape == a[0].shape == b[0].shape):
        raise ValueError(f"SELECT: shape mismatch {m.shape} vs {a[0].shape} vs {b[0].shape}")
    return (np.where(m, a[0], b[0]).astype(np.int8),
            np.where(m, a[1], b[1]).astype(np.int32),
            np.where(m, a[2], b[2]).astype(np.uint8))


def op_concat(vals, config, feeds):
    """Concatenate two same-kind streams along an axis literal (append-only
    growth primitive for KV-style caches). Exact move; all dims except the
    axis must match (fail loud). Triples concatenate plane-wise (the three
    planes stay aligned by construction -- gated); plain arrays concatenate
    directly."""
    a, b, ax = vals
    axis = _int_arg(ax, "CONCAT axis")

    def cat(x, y):
        if not (0 <= axis < x.ndim == y.ndim):
            raise ValueError(f"CONCAT: bad axis {axis} for ndim {x.ndim}/{y.ndim}")
        dx, dy = list(x.shape), list(y.shape)
        if [d for i, d in enumerate(dx) if i != axis] != [d for i, d in enumerate(dy) if i != axis]:
            raise ValueError(f"CONCAT: non-axis dims differ {dx} vs {dy}")
        return np.concatenate([x, y], axis=axis)

    if isinstance(a, tuple) and isinstance(b, tuple):
        return (cat(a[0], b[0]), cat(a[1], b[1]), cat(a[2], b[2]))
    return cat(np.ascontiguousarray(a), np.ascontiguousarray(b))


def op_argmax(vals, config, feeds):
    """Argmax over an axis literal: exact lattice ordering WITHOUT decoding
    (the only new math in Pile A, mini-bar: class rank pos>zero>neg, then
    exponent -- larger e wins for BOTH signs (pos: bigger value; neg: more
    negative = smaller... careful: among negatives the SMALLEST value has
    the LARGEST e; argmax wants the largest value = positives first, then
    zero, then negatives closest to zero = SMALLEST e). Ties -> first index
    (deterministic, stated). Returns int64 indices (I streams)."""
    (t,) = vals[:1]
    axis = _int_arg(vals[1], "ARGMAX axis") if len(vals) > 1 else -1
    s = np.ascontiguousarray(t[0])
    e = np.ascontiguousarray(t[1]).astype(np.int64)
    z = np.ascontiguousarray(t[2]).astype(bool)
    pos = (~z) & (s > 0)
    neg = (~z) & (s < 0)
    ax = axis % s.ndim
    # move target axis last for uniform handling
    ps = np.moveaxis(pos.astype(np.int64), ax, -1)
    ng = np.moveaxis(neg.astype(np.int64), ax, -1)
    ee = np.moveaxis(e, ax, -1)
    zz = np.moveaxis(z, ax, -1)
    n = ee.shape[-1]
    idx = np.arange(n)
    # rank key: class (pos 2 > zero 1 > neg 0) primary; within pos: max e;
    # within neg: min e (closest to zero); ties: first index. lexsort takes
    # keys ascending with LAST primary: (rev-index, key2, class).
    cls = np.where(ps > 0, 2, np.where(zz, 1, 0))
    key2 = np.where(ps > 0, ee, np.where(zz, 0, -ee))
    order = np.lexsort((np.broadcast_to(-idx, ee.shape), key2, cls), axis=-1)
    return np.ascontiguousarray(order[..., -1]).astype(np.int64)


def op_slice(vals, config, feeds):
    """Window a stream: SLICE(X, AXIS, START, END) integer literals
    (half-open [start,end), numpy semantics). Exact move; bounds-checked
    (fail loud on out-of-range -- silent clamping would hide shape bugs).
    Layout preserved approximately (documented: dims unchecked v1)."""
    (t,) = vals[:1]
    axis = _int_arg(vals[1], "SLICE axis")
    start = _int_arg(vals[2], "SLICE start")
    end = _int_arg(vals[3], "SLICE end")
    base = t[0] if isinstance(t, tuple) else np.ascontiguousarray(t)
    nd = base.ndim
    if not (0 <= axis < nd):
        raise ValueError(f"SLICE: bad axis {axis} for ndim {nd}")
    if not (0 <= start <= end <= base.shape[axis]):
        raise ValueError(f"SLICE: [{start},{end}) out of range dim {axis}={base.shape[axis]}")
    sl = [slice(None)] * nd
    sl[axis] = slice(start, end)
    sl = tuple(sl)
    if isinstance(t, tuple):
        return (np.ascontiguousarray(t[0][sl]), np.ascontiguousarray(t[1][sl]),
                np.ascontiguousarray(t[2][sl]))
    return np.ascontiguousarray(np.ascontiguousarray(t)[sl])


def op_clip(vals, config, feeds):
    """Clip triples to [LO,HI] float literals (holo clip_fixed). Exact
    lattice compare, no float arithmetic on values."""
    t, lo, hi = vals
    _, m_cov = _scales()
    return H.clip_fixed(t, float(lo), float(hi), m_cov)


def op_div(vals, config, feeds):
    """Triple divide (holo tdiv_pure: sign-XOR + exp-sub, zero-or). No
    guards: 0/0 stays a zero triple (callers add zero semantics explicitly)."""
    return H.tdiv_pure(vals[0], vals[1])


def op_sigmoid(vals, config, feeds):
    """Sigmoid via EXPACT+LUT (holo sigmoid_trip). Any input range, [0,1]."""
    return H.sigmoid_trip(vals[0])


def op_rescale(vals, config, feeds):
    """The ONLY scale changer, in-language: triples -> fixed@m_cov ->
    rescale_ to M2 (int literal scale) -> triples. Value-preserving up to
    lattice quantum (gated). Needed wherever listings cross scales (MATMUL
    m_acc vs surrounding m_cov)."""
    t, m2v = vals
    m2 = _int_arg(m2v, "RESCALE m2")
    if not (0 <= m2 < 65536):
        raise ValueError(f"RESCALE: scale out of range: {m2}")
    _, m_cov = _scales()
    q = S.to_fixed(t[0], t[1], t[2], m_cov)
    return S.from_fixed(H.rescale_(q, m_cov, m2), m2)


def op_gather(vals, config, feeds):
    """Exact row-gather: table (V,C) triples + int ids -> rows (phi-core
    gather_int, 0-diff). Reindex family (structurally gated upstream)."""
    N, _S = _phi_ops()
    w, ids = vals
    return N.gather_int(w, np.ascontiguousarray(ids))


def op_prelu(vals, config, feeds):
    """PReLU exact (phi-core prelu_int: sign-bit select + triple slope).
    0-diff: wrapper adds nothing."""
    N, _S = _phi_ops()
    return N.prelu_int(vals[0], vals[1])


def op_poolavg(vals, config, feeds):
    """Global average pool to (1,1,C) (phi-core avgpool_int). HWC triples
    only (asserted -- SEQ layouts fail loud, not silently mis-averaged).
    Mean is truncating (documented 1-LSB-class approx upstream)."""
    N, _S = _phi_ops()
    (t,) = vals
    if t[0].ndim != 3:
        raise ValueError(f"POOLAVG: HWC triples only, got ndim={t[0].ndim}")
    _, m_cov = _scales()
    return N.avgpool_int(t[0], t[1], t[2], m_cov)


def op_deconv(vals, config, feeds):
    """Transpose-conv (phi-core deconv_int, torch-verified upstream).
    W as triples (O,kh,kh,Cin) + int stride/pad literals; no bias v1
    (documented -- bias-drop lesson lives upstream; add Wb stream when
    demanded). 0-diff vs phi-core fn with same args."""
    N, _S = _phi_ops()
    t, w = vals[0], vals[1]
    stride = _int_arg(vals[2], "DECONV stride")
    pad = _int_arg(vals[3], "DECONV pad")
    _, m_cov = _scales()
    Wd = {"s": w[0], "e": w[1], "z": w[2]}
    return N.deconv_int(t[0], t[1], t[2], Wd, m_cov, stride=stride, pad=pad)


def _is_dyadic(v):
    """Dyadic scale check via frexp (exact): significand must be 0.5."""
    import math
    f = float(v)
    if f <= 0:
        return False
    return math.frexp(f)[0] == 0.5


def op_interp(vals, config, feeds):
    """Bilinear resample (phi-core interp_fixed) with dyadic enforcement:
    non-dyadic scales are a LOWERING ERROR per IR doctrine (fail loud here,
    never silently approximate). Bridge triples->fixed@m_cov and back
    inside (stated roundtrip)."""
    N, S = _phi_ops()
    t, syv, sxv = vals
    sy, sx = float(syv), float(sxv)
    if not (_is_dyadic(sy) and _is_dyadic(sx)):
        raise ValueError(f"INTERP: non-dyadic scale ({sy},{sx}) is a LOWERING ERROR")
    _, m_cov = _scales()
    q = S.to_fixed(t[0], t[1], t[2], m_cov)
    H, W = q.shape[0], q.shape[1]
    C = q.shape[2] if q.ndim == 3 else 1
    Ho, Wo = max(int(round(H * sy)), 1), max(int(round(W * sx)), 1)
    o = S.from_fixed(N.interp_fixed(q.reshape(H, W, C), sy, sx).reshape(-1), m_cov)
    return (o[0].reshape(Ho, Wo, C), o[1].reshape(Ho, Wo, C), o[2].reshape(Ho, Wo, C))


def op_conv(vals, config, feeds):
    """General-kernel conv (holo conv_trip: per-tap tmul + order-free
    accumulate). K as FLOAT kernel array stream (encoded per-tap inside,
    same as conv paths -- kernel_triples-identical). m_acc/m_out from
    frozen scales. 0-diff vs conv_trip trivially (same fn, proves wiring)."""
    t, k = vals
    m_acc, m_cov = _scales()
    return H.conv_trip(t, np.ascontiguousarray(k, dtype=np.float64),
                       m_acc, m_out=m_cov)


def op_mixdyad(vals, config, feeds):
    """Motion-gated dyadic memory: static ? (D+3W)/4 : D. W=None (no
    history) -> D directly (feed convention, same doctrine as temporal
    step's state-None rule). Adds + trunc-halve only: dyadic by
    construction, no multiplier. Moving trust is bit-clean (no arithmetic
    on the False branch)."""
    from chain.holo_phi import binop_fixed
    d, w, s = vals
    _, m_cov = _scales()
    if w is None:
        return d
    import phi_core.lattice as S
    acc = binop_fixed(d, w, m_cov, m_cov, op="add")
    acc = binop_fixed(acc, w, m_cov, m_cov, op="add")
    acc = binop_fixed(acc, w, m_cov, m_cov, op="add")
    mixed = S.from_fixed(S.tdiv(
        S.to_fixed(acc[0], acc[1], acc[2], m_cov), 4), m_cov)
    m = np.ascontiguousarray(s, dtype=bool)
    assert m.shape == d[0].shape, f"mask geometry {m.shape}"
    return (np.where(m, mixed[0], d[0]).astype(np.int8),
            np.where(m, mixed[1], d[1]).astype(np.int32),
            np.where(m, mixed[2], d[2]).astype(np.uint8))


REGISTRY = {
    "SRGB_DECODE": (op_srgb_decode, 1, 1),
    "LUMA": (op_luma, 1, 1),
    "SQRT": (op_sqrt, 1, 1),
    "SPLAT_BLUR": (op_splat_blur, 1, 2),
    "SUB": (op_sub, 2, 1),
    "BETA_V5": (op_beta_v5, 2, 1),
    "MUL": (op_mul, 2, 1),
    "ADD": (op_add, 2, 1),
    "SQUARE": (op_square, 1, 1),
    "GAIN": (op_gain, 3, 1),
    "SRGB_ENCODE": (op_srgb_encode, 1, 1),
    "BETA": (op_beta, 1, 1),
    "ISO_BLUR": (op_iso_blur, 1, 1),
    "WARP": (op_warp, 2, 1),
    "STATIC": (op_static, 1, 1),
    "MIXDYAD": (op_mixdyad, 3, 1),
    "MATMUL": (op_matmul, 2, 1),
    "SOFTMAX": (op_softmax, 1, 1),
    "RMSNORM": (op_rmsnorm, 2, 1),
    "SILU": (op_silu, 1, 1),
    "GELU": (op_gelu, 1, 1),
    "ROTARY": (op_rotary, 2, 1),
    "BATCH_MATMUL": (op_batch_matmul, 2, 1),
    "TRANSPOSE": (op_transpose, 1, 1),
    "GAUSS": (op_gauss, 3, 1),
    "RESHAPE2": (op_reshape2, 3, 1),
    "RESHAPE3": (op_reshape3, 4, 1),
    "PERMUTE3": (op_permute3, 4, 1),
    "SELECT": (op_select, 3, 1),
    "CONCAT": (op_concat, 3, 1),
    "ARGMAX": (op_argmax, 2, 1),
    "SLICE": (op_slice, 4, 1),
    "CLIP": (op_clip, 3, 1),
    "DIV": (op_div, 2, 1),
    "SIGMOID": (op_sigmoid, 1, 1),
    "RESCALE": (op_rescale, 2, 1),
    "GATHER": (op_gather, 2, 1),
    "PRELU": (op_prelu, 2, 1),
    "POOLAVG": (op_poolavg, 1, 1),
    "DECONV": (op_deconv, 4, 1),
    "INTERP": (op_interp, 3, 1),
    "CONV": (op_conv, 2, 1),
}

# Layout signatures (TYPED STREAMS v1): (in_layouts, out_layouts) per
# mnemonic. "$VAR" unifies whole layout strings; "$VAR^T" derives;
# "*" matches anything, binds nothing; missing entry = all-wildcard.
# Layouts "KIND:GEOM" (F float / T triples / I integer-exact / U8 bytes).
# v1 limits (stated): no rank arithmetic (MATMUL wildcard; phi-core asserts
# inside fail loud), POS/metadata carried as "*" (structural, pre-formal).
SIGS = {
    "SRGB_DECODE": (["U8:HWC"], ["F:HWC"]),
    "LUMA": (["F:HWC"], ["F:HW"]),
    "SQRT": (["F:HW"], ["T:HW"]),
    "SPLAT_BLUR": (["T:HW"], ["T:HW", "T:HW"]),
    "SUB": (["$A", "$A"], ["$A"]),
    "BETA_V5": (["T:HW", "T:HW"], ["T:HW"]),
    "BETA": (["$A"], ["$A"]),
    "MUL": (["$A", "$A"], ["$A"]),
    "ADD": (["$A", "$A"], ["$A"]),
    "SQUARE": (["$A"], ["$A"]),
    "GAIN": (["F:HWC", "F:HW", "T:HW"], ["F:HWC"]),
    "SRGB_ENCODE": (["F:HWC"], ["U8:HWC"]),
    "ISO_BLUR": (["T:HW"], ["T:HW"]),
    "WARP": (["T:HW", "F:HW2"], ["T:HW"]),
    "STATIC": (["F:HW2"], ["I:HW"]),
    "MIXDYAD": (["T:HW", "T:HW", "I:HW"], ["T:HW"]),
    "MATMUL": (["*", "*"], ["*"]),
    "SOFTMAX": (["$A"], ["$A"]),
    "RMSNORM": (["$X", "$W"], ["$X"]),
    "SILU": (["$A"], ["$A"]),
    "GELU": (["$A"], ["$A"]),
    "ROTARY": (["$X", "*"], ["$X"]),
    "BATCH_MATMUL": (["*", "*"], ["*"]),
    "TRANSPOSE": (["$A"], ["$A^T"]),
    "GAUSS": (["$A", "F:SCALAR", "F:SCALAR"], ["$A"]),
    "RESHAPE2": (["*", "F:SCALAR", "F:SCALAR"], ["*"]),
    "RESHAPE3": (["*", "F:SCALAR", "F:SCALAR", "F:SCALAR"], ["*"]),
    "PERMUTE3": (["*", "F:SCALAR", "F:SCALAR", "F:SCALAR"], ["*"]),
    "SELECT": (["I:*", "$A", "$A"], ["$A"]),
    "CONCAT": (["$A", "$A", "F:SCALAR"], ["$A"]),
    "ARGMAX": (["$A", "F:SCALAR"], ["I:*"]),
    "SLICE": (["$A", "F:SCALAR", "F:SCALAR", "F:SCALAR"], ["$A"]),
    "CLIP": (["$A", "F:SCALAR", "F:SCALAR"], ["$A"]),
    "DIV": (["$A", "$A"], ["$A"]),
    "SIGMOID": (["$A"], ["$A"]),
    "RESCALE": (["$A", "F:SCALAR"], ["$A"]),
    "GATHER": (["*", "I:*"], ["*"]),
    "PRELU": (["$A", "$A"], ["$A"]),
    "POOLAVG": (["*"], ["*"]),
    "DECONV": (["*", "*", "F:SCALAR", "F:SCALAR"], ["*"]),
    "INTERP": (["*", "F:SCALAR", "F:SCALAR"], ["*"]),
    "CONV": (["*", "*"], ["*"]),
}
