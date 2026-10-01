"""Vendored wide bridge (BACKPORT, delete on merge).

Mirrors phi-core ai/t-transform-bridge's to_fixed_wide + frac_hi LUT
bit-exactly (gated 0-diff in test_tshift.py) so DDColor-scale work proceeds
without waiting on owner review. Provenance: copied verbatim from the
branch (numpy_ops.py), path-adjusted to chain/luts/ (own LUT file, same
bytes on build). DELETE THIS FILE when the branch merges (the gate will
fail loudly on drift before then -- mirrors drift, cf LIB-014).
Covers |v| to ~2200 at any m; beyond keeps the fold convention.
"""
import os

import numpy as np

import phi_core.lattice as S

WIDE_CAP = 8192
LUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "luts", "frac_hi_lut.npy")
_FRAC_HI = None


def frac_hi_lut():
    global _FRAC_HI
    if _FRAC_HI is None:
        try:
            _FRAC_HI = np.load(LUT_PATH)
        except Exception:
            _FRAC_HI = np.round(np.power(
                S.PHI, np.arange(1, WIDE_CAP + 1, dtype=np.float64) / S.K
                ) * (1 << 18)).astype(np.int64)
            try:
                os.makedirs(os.path.dirname(LUT_PATH), exist_ok=True)
                np.save(LUT_PATH, _FRAC_HI)
            except Exception:
                pass
    return _FRAC_HI


def to_fixed_wide(s, e, z, m):
    """See module docstring (backport)."""
    s = np.ascontiguousarray(s)
    e = np.ascontiguousarray(e)
    z = np.ascontiguousarray(z)
    d = np.asarray(m, dtype=np.int64) - e.astype(np.int64)
    F = S.L_FRAC()
    base = F[0]
    G = frac_hi_lut()
    q_lo = np.where(d < -WIDE_CAP, s.astype(np.int64) * base,
                    s.astype(np.int64) * G[np.clip(-d, 1, WIDE_CAP) - 1])
    out = np.where(z.astype(bool), 0, np.where(
        d < 0, q_lo, np.where(
            d > S.FRAC_CAP, 0,
            s.astype(np.int64) * F[np.clip(d, 0, S.FRAC_CAP)])))
    return out.astype(np.int64)
