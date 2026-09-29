"""Beta-field controller (step 2, v3): per-pixel effective beta, fitted offline.

Rule v3: three fitted levels selected by (bucket, coherence band) -- iso
fallback gets iso_atten; oriented pixels split by coh_hi into strong
(strong_atten) and mid (mid_atten). Rotation symmetry is the inductive
bias: all four orientations share levels (no per-direction params). Read as
a 2-layer decision net (threshold routing + table); executed as exact
integer selects + tmul. Stated reasons in docs/BETA_CTRL.md. Fitting lives
in scripts/fit_ctrl.py; this module loads the frozen result.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
import phi_core.lattice as S

CHAIN_DIR = os.path.dirname(os.path.abspath(__file__))
CTRL_PATH = os.path.join(CHAIN_DIR, "CTRL.json")

DEFAULTS = {"iso_atten": 0.5, "coh_thr": 0.25,
            "mid_atten": 0.6, "coh_hi": 0.5, "strong_atten": 1.0}

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
                        "strong_atten"):
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
