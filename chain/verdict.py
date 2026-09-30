"""Verdicts (#LIB-019 promotion): decide float at the seam, consume exact.

The pattern, third use reached (motion static mask, depth median split,
router mode verdict): a boundary float comparison producing an exact boolean
verdict that downstream integer selects consume. Both substrates call the
SAME conversion, so verdicts agree structurally -- never re-derive the
comparison downstream (a second mapping is a mirror waiting to drift,
cf #LIB-014 postscript).

Contract: verdict_mask(x, thr, op) with op in {"==","<=",">="} returns an
exact bool array. Inclusivity is STATED per op (">=" engages at equality --
the router boundary gate pins this). NaN input is a caller bug: NaN
comparisons are False under every op, which would silently verdict False;
callers with untrusted inputs assert finiteness first (all three current
call sites operate on finite fields by construction: magnitudes, normalized
depth, finite means).
"""
import numpy as np

_OPS = {"==": np.equal, "<=": np.less_equal, ">=": np.greater_equal}


def verdict_mask(x, thr, op):
    """Float array + scalar threshold + direction -> exact bool mask."""
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}, got {op!r}")
    x = np.ascontiguousarray(x, dtype=np.float64)
    return np.asarray(_OPS[op](x, float(thr)))
