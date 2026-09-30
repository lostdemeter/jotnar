"""Beta-field controller (step 2, v5): per-pixel effective beta, fitted offline.

Rule v5: v4 soft field times a learned detail gate --
  beff = beff_v4 * scale, scale = 0.5 + sigmoid(w0 + w1*coh + w2*dhat),
  dhat = clip(|D|*4, 0, 1).
Zero weights give gate=0.5, scale=1.0, i.e. exactly v4: the fitter must beat
v4 to rewrite (same must-beat doctrine as every round). Rotation symmetry
kept (features are rotation-invariant: coherence magnitude, detail
magnitude -- no orientation enters). One 1x1 layer + sigmoid: the tiny net,
executed as tmul/binop/sigmoid_trip (all already-lowered IR ops, so v5 needs
no new C -- composition only). Stated reasons in docs/BETA_CTRL.md. Fitting
lives in scripts/fit_v5.py; this module loads the frozen result.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
import phi_core.lattice as S

CHAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CTRL_PATH = os.path.join(CHAIN_DIR, "CTRL.json")

DEFAULTS = {"iso_atten": 0.5, "coh_thr": 0.25,
            "mid_atten": 0.6, "coh_hi": 0.5, "strong_atten": 1.0,
            "v5_w0": 0.0, "v5_w1": 0.0, "v5_w2": 0.0}

_cache = None


def load_ctrl():
    """Frozen controller params (whole validated dict, not a subset: every
    key the fitter freezes must reach the runtime -- a past revision dropped
    mid_atten/coh_hi here and both sides silently ran different defaults)."""
    global _cache
    if _cache is None:
        try:
            with open(CTRL_PATH) as fh:
                d = json.load(fh)
            m = dict(DEFAULTS)
            for k in ("iso_atten", "coh_thr", "mid_atten", "coh_hi",
                        "strong_atten", "v5_w0", "v5_w1", "v5_w2"):
                if k in d:
                    m[k] = float(d[k])
            _cache = m
        except Exception:
            _cache = dict(DEFAULTS)
    return _cache


def reset_cache():
    global _cache
    _cache = None


def beta_field(beta, bucket, coh_t, m_cov, atten=None, mid=None, hi=None,
               strong=None):
    """v3 per-pixel beta triples: iso fallback gets atten; oriented pixels
    split by coherence magnitude (strong -> strong_atten, weak -> mid).
    All selects exact on integer masks (fixed compares for the split).
    Overrides are search-only; runtime leaves them None. Missing file keys
    fall back to v2 behavior (strong 1.0)."""
    p = load_ctrl()
    att = float(atten) if atten is not None else p.get("iso_atten", 0.5)
    md = float(mid) if mid is not None else p.get("mid_atten", 0.6)
    hiv = float(hi) if hi is not None else p.get("coh_hi", 0.5)
    st = float(strong) if strong is not None else p.get("strong_atten", 1.0)
    full = S.encode(np.full(bucket.shape, float(beta) * st, np.float64))
    iso_t = S.encode(np.full(bucket.shape, float(beta) * att, np.float64))
    mid_t = S.encode(np.full(bucket.shape, float(beta) * md, np.float64))
    # coherence split in fixed domain (integer compare, no FP)
    cq = S.to_fixed(coh_t[0], coh_t[1], coh_t[2], m_cov)
    hq = S.to_fixed(*S.encode(np.array([hiv], dtype=np.float64)), m_cov)[0]
    strong_m = cq >= hq
    # index: iso -> 2, oriented-strong -> 0, oriented-weak -> 1
    idx = np.where(bucket == 4, 2, np.where(strong_m, 0, 1)).astype(np.int8)
    # #LIB-009: exact select via the shared helper.
    from chain.holo_phi import select_mux
    return select_mux([full, mid_t, iso_t], idx)


def beta_field_soft(beta, coh_t, m_cov, k=30.0):
    """v4 continuous beta triples: no bucket, no hard select. Weight blends
    the three fitted levels by coherence through integer sigmoids:
      w = iso + (mid-iso)*s(k*(coh-thr)) + (strong-mid)*s(k*(coh-hi))
      beff = beta * w   (tmul exact; sums via same-scale binop)
    All transcendental work in frozen LUTs (sigmoid_trip); ALU sees integers
    + gathers only. k=30 fixed (transition ~0.27 wide); levels from file."""
    from chain.holo_phi import sigmoid_trip, tmul, binop_fixed
    p = load_ctrl()
    att, md = p.get("iso_atten", 0.5), p.get("mid_atten", 0.6)
    st = p.get("strong_atten", 1.0)
    thr, hi = p.get("coh_thr", 0.15), p.get("coh_hi", 0.4)
    shape = coh_t[0].shape
    kt = S.encode(np.full(shape, float(k), dtype=np.float64))

    def sig_at(level):
        lt = S.encode(np.full(shape, float(level), dtype=np.float64))
        d = binop_fixed(coh_t, lt, m_cov, m_cov, op="sub")
        return sigmoid_trip(tmul(d, kt))

    s1 = sig_at(thr)
    s2 = sig_at(hi)
    iso = S.encode(np.full(shape, float(att), dtype=np.float64))
    dm = S.encode(np.full(shape, float(md - att), dtype=np.float64))
    ds = S.encode(np.full(shape, float(st - md), dtype=np.float64))
    w = binop_fixed(iso, tmul(dm, s1), m_cov, m_cov, op="add")
    w = binop_fixed(w, tmul(ds, s2), m_cov, m_cov, op="add")
    b = S.encode(np.full(shape, float(beta), dtype=np.float64))
    return tmul(b, w)


def beta_field_v5(beta, coh_t, d_t, m_cov, w0=None, w1=None, w2=None):
    """v5 learned detail gate: beff_v4 * scale, scale in [0.5, 1.5].
    logit = w0 + w1*coh + w2*dhat (tmul scalars + same-scale binop adds);
    gate = sigmoid_trip(logit); scale = 0.5 + gate. dhat = clip(|D|*4,0,1):
    |D| via exact abs_trip, x4 via exact tmul, clip via clip_fixed.
    Overrides search-only; missing file keys fall back to zeros (= v4).
    Continuous (sigmoid of triples): noise parity stays barred."""
    from chain.holo_phi import (sigmoid_trip, tmul, binop_fixed, abs_trip,
                                clip_fixed)
    p = load_ctrl()
    a0 = float(w0) if w0 is not None else p.get("v5_w0", 0.0)
    a1 = float(w1) if w1 is not None else p.get("v5_w1", 0.0)
    a2 = float(w2) if w2 is not None else p.get("v5_w2", 0.0)
    shape = coh_t[0].shape
    base = beta_field_soft(beta, coh_t, m_cov)
    # detail feature in [0,1]
    four = S.encode(np.full(shape, 4.0, dtype=np.float64))
    dhat = clip_fixed(tmul(abs_trip(d_t), four), 0.0, 1.0, m_cov)
    # logit terms (scalar triples exact via encode)
    t0 = S.encode(np.full(shape, a0, dtype=np.float64))
    t1 = tmul(S.encode(np.full(shape, a1, dtype=np.float64)), coh_t)
    t2 = tmul(S.encode(np.full(shape, a2, dtype=np.float64)), dhat)
    logit = binop_fixed(binop_fixed(t0, t1, m_cov, m_cov, op="add"),
                        t2, m_cov, m_cov, op="add")
    gate = sigmoid_trip(logit)
    half = S.encode(np.full(shape, 0.5, dtype=np.float64))
    scale = binop_fixed(half, gate, m_cov, m_cov, op="add")
    return tmul(base, scale)
