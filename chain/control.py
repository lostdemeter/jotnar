"""Beta-field controller (step 2, v2): per-pixel effective beta, fitted offline.

Rule v2: iso fallback gets iso_atten; oriented pixels split by coherence
magnitude (strong -> full beta, weak -> mid_atten). Stated reasons in
docs/BETA_CTRL.md. Runtime is exact selects + tmul (integers + gather only).
Fitting lives in scripts/fit_ctrl.py; this module loads the frozen result.
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
            "mid_atten": 0.6, "coh_hi": 0.5}

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
            for k in ("iso_atten", "coh_thr", "mid_atten", "coh_hi"):
                if k in d:
                    m[k] = float(d[k])
            _cache = m
        except Exception:
            _cache = dict(DEFAULTS)
    return _cache


def reset_cache():
    global _cache
    _cache = None


def beta_field(beta, bucket, coh_t, m_cov, atten=None, mid=None, hi=None):
    """v2 per-pixel beta triples: iso fallback gets atten; oriented pixels
    split by coherence magnitude (strong -> full beta, weak -> mid).
    All selects exact on integer masks (fixed compares for the split).
    atten/mid/hi override the frozen file (search only; runtime None)."""
    p = load_ctrl()
    att = float(atten) if atten is not None else p.get("iso_atten", 0.5)
    md = float(mid) if mid is not None else p.get("mid_atten", 0.6)
    hiv = float(hi) if hi is not None else p.get("coh_hi", 0.5)
    full = S.encode(np.full(bucket.shape, float(beta), np.float64))
    iso_t = S.encode(np.full(bucket.shape, float(beta) * att, np.float64))
    mid_t = S.encode(np.full(bucket.shape, float(beta) * md, np.float64))
    # coherence split in fixed domain (integer compare, no FP)
    cq = S.to_fixed(coh_t[0], coh_t[1], coh_t[2], m_cov)
    hq = S.to_fixed(*S.encode(np.array([hiv], dtype=np.float64)), m_cov)[0]
    strong = cq >= hq
    # index: iso -> 2, oriented-strong -> 0, oriented-weak -> 1
    idx = np.where(bucket == 4, 2, np.where(strong, 0, 1)).astype(np.int8)
    # #LIB-009: exact select via the shared helper.
    from chain.holo_phi import select_mux
    return select_mux([full, mid_t, iso_t], idx)
