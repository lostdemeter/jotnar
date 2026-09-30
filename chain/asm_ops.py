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


def op_sub(vals, config, feeds):
    a, b = vals
    _, m_cov = _scales()
    return H.binop_fixed(a, b, m_cov, m_cov, op="sub")


def op_beta_v5(vals, config, feeds):
    d, coh = vals
    _, m_cov = _scales()
    return C.beta_field_v5(_beta(config), coh, d, m_cov)


def op_mul(vals, config, feeds):
    return H.tmul(vals[0], vals[1])


def op_add(vals, config, feeds):
    _, m_cov = _scales()
    return H.binop_fixed(vals[0], vals[1], m_cov, m_cov, op="add")


def op_square(vals, config, feeds):
    (a,) = vals
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
}
