"""Assembler-side install synthesis (host math, doctrine-clean).

The language stays geometric; the COMPILER may use any math (like
assembler-side SVD and the predictor cost model). This module solves
the install as a QP: min ||v|| s.t. A v >= d, where rows of A are
(w_target - w_competitor) in the decision head and d the logit
deficits (+ margin). Solved via the dual (concave quadratic over
lambda >= 0, projected gradient ascent): v* = A'lam/2.
Replaces dose ladders with one solve: specify target ranking, emit
value + dose with predicted receipt. Non-geometric paths are
exploration/tests only; anything synthesized here must still
install through listings (the gate decides, this proposes).
"""
import numpy as np


def qpsolve(A, d, iters=20000, tol=1e-9):
    """min ||v|| s.t. A v >= d. Returns (v, req=||v||).
    Raises ValueError if infeasible within iters (dual gap check:
    primal-dual agree at optimum; disagreement fails loud)."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    d = np.ascontiguousarray(d, dtype=np.float64).ravel()
    m, n = A.shape
    lam = np.zeros(m)
    step = 0.5 / (float(np.linalg.norm(A, ord=2)) ** 2 + 1e-12)
    for _ in range(iters):
        v = (A.T @ lam) / 2.0
        grad = d - A @ v
        lam = np.maximum(lam + step * grad, 0.0)
    v = (A.T @ lam) / 2.0
    viol = d - A @ v
    if float(np.max(viol)) > 1e-3 * max(1.0, float(np.abs(d).max())):
        raise ValueError(f"qpsolve: infeasible/unconverged "
                         f"(max viol {float(np.max(viol)):.3g})")
    return v, float(np.linalg.norm(v))


def synth_value(W, base_logits, target, comps, margin=0.5, mass=1.0):
    """Optimal install value direction+norm for target vs comps.
    W: (D,V) decision head; base_logits (V,); comps [ids]; margin: logit
    headroom beyond top (engineering margin: optimal == fragile);
    mass: expected routing weight (blend dilutes: norm scales by 1/mass).
    Returns (v, req, info dict with per-comp gains)."""
    wt = np.ascontiguousarray(W[:, target])
    A = np.stack([wt - np.ascontiguousarray(W[:, c]) for c in comps], axis=0)
    d = np.array([float(base_logits[c] - base_logits[target]) + margin
                  for c in comps])
    pos = d > 0
    if not np.any(pos):
        return np.zeros(W.shape[0]), 0.0, {"note": "already top-1"}
    v, req = qpsolve(A[pos], d[pos])
    req /= max(mass, 1e-9)
    v = v / (np.linalg.norm(v) + 1e-12) * req
    return v, req, {"t_unit": None, "n_comp": int(pos.sum()),
                    "max_deficit": float(d[pos].max())}
