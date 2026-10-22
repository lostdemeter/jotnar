"""Hopfield capacity math (analysis round, CPU-only, no runs).

Frozen formulas from Ramsauer et al., arXiv:2008.02282 (see
docs/HOPFIELD.md constitution + honesty note + K-correction).
Pure functions (importable — the gate recomputes from them);
main() prints the capacity table and banks
samples/marigold/hopfield_capacity.json.
Usage: python3 scripts/hopfield_capacity.py
"""
import json
import math
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DIMS = (64, 128, 256, 512, 1024)
K_PATTERNS = 20       # 5 observed basins x safety factor 4
P_FAIL = 0.001
BETA = 1.0
K_RADIUS = 1.0        # patterns on sphere radius sqrt(d-1) (K-correction)
CLASSICAL_RATIO = 0.138


def lambert_w0(z):
    """Principal Lambert W by Newton iteration (checked vs W(e)=1,
    W(1)=Omega, W(0)=0 in-gate)."""
    if z == 0.0:
        return 0.0
    if z < 0.0:
        raise ValueError("W0 needs z >= 0 here")
    w = math.log1p(z)
    for _ in range(100):
        ew = math.exp(w)
        dw = (w * ew - z) / (ew * (w + 1.0))
        w -= dw
        if abs(dw) < 1e-15 * max(1.0, abs(w)):
            break
    return w


def radius(d):
    return K_RADIUS * math.sqrt(d - 1)


def capacity_exp(d, beta=BETA, p=P_FAIL):
    """Ramsauer lower bound: N >= sqrt(p) * c^((d-1)/4)."""
    if d <= 1:
        raise ValueError("need d > 1")
    K = K_RADIUS
    a = (2.0 / (d - 1)) * (1.0 + math.log(2 * beta * K * K * p * (d - 1)))
    b = 2 * K * K * beta / 5.0
    c = b / lambert_w0(math.exp(a + math.log(b)))
    return math.sqrt(p) * c ** ((d - 1) / 4.0), c


def separation_required(N, beta=BETA, M=None, d=64):
    """Theorem-1 storability floor: min dot-separation for N patterns."""
    if N < 2:
        return 0.0
    M = radius(d) if M is None else M
    return 2.0 / (beta * N) + (1.0 / beta) * math.log(
        2 * (N - 1) * N * beta * M * M)


def q_tail(x):
    """Upper Gaussian tail P(Z > x)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def capacity_sep(d, beta=BETA, p=P_FAIL):
    """Separation-limited capacity: max N with P(all pairs carry the
    Theorem-1 separation) >= 1-p. Random patterns on the radius-M
    sphere: pair dots ~ N(0, M^4/d); need maxdot <= M^2 - delta
    (union bound over pairs). Returns (N, delta_req). The scan caps
    at N=1e6: beyond that separation never binds inside any regime
    this round judges (verdicts live at 20/80), reported unbounded.
    (Rationale: Gaussian tails underflow to 0.0 exactly at high z,
    so an unbounded scan would spin to 1e9.)"""
    M = radius(d)
    N = 1
    while True:
        N += 1
        if N > 10 ** 6:
            return float("inf"), separation_required(N, beta, M)
        delta = separation_required(N, beta, M)
        z = (M * M - delta) / (M * M / math.sqrt(d))
        if z <= 0:
            return N - 1, separation_required(N - 1, beta, M)
        if (N * (N - 1) / 2.0) * q_tail(z) > p:
            return N - 1, separation_required(N - 1, beta, M)


def capacity_classical(d):
    return CLASSICAL_RATIO * d


def verdict(C, k=K_PATTERNS):
    if C >= 4 * k:
        return "sufficient"
    if C >= k:
        return "marginal"
    return "insufficient"


def table():
    rows = {}
    for d in DIMS:
        n_r, c = capacity_exp(d)
        n_s, delta = capacity_sep(d)
        unb = math.isinf(n_s)
        C = n_r if unb else min(n_r, n_s)
        n_r20 = math.sqrt(P_FAIL) * (0.8 * c) ** ((d - 1) / 4.0)
        rows[d] = {"N_exp": n_r, "c": c,
                   "N_sep": (None if unb else float(n_s)),
                   "N_sep_unbounded": unb,
                   "delta_req": delta, "classical": capacity_classical(d),
                   "capacity": C, "verdict": verdict(C),
                   "robust_minus20": verdict(
                       n_r20 if unb else min(n_r20, n_s))}
    return rows


def main():
    rows = table()
    print(f"{'dim':>5} {'N_exp':>12} {'N_sep':>12} {'classical':>10} "
          f"{'capacity':>12} verdict")
    for d in DIMS:
        r = rows[d]
        ns = "inf" if r["N_sep"] is None else f"{r['N_sep']:12.3g}"
        print(f"{d:>5} {r['N_exp']:12.3g} {ns:>12} "
              f"{r['classical']:10.3g} {r['capacity']:12.3g} "
              f"{r['verdict']}")
    rep = {"dims": list(DIMS), "K": K_PATTERNS, "p": P_FAIL,
           "beta": BETA, "K_radius": K_RADIUS,
           "rows": {str(d): v for d, v in rows.items()}}
    outp = os.path.join(ROOT, "samples", "marigold",
                        "hopfield_capacity.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
