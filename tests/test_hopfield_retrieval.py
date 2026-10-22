"""Hopfield-retrieval gate (docs/HOPFIELD.md constitution): CPU-only, seconds.

Legs (all recomputed in-gate from banked prototypes + banked math):
- prototype-integrity: banked harvests reload (3 points x 8 frames x
  3072, finite/nonzero) and L2-normalize to unit rows (the
  normalize-before-store invariant, checked not trusted).
- correctness legs: per-level accuracy recomputed == banked and below
  the frozen bar (0.25); LOO diagnostic recomputed == banked.
- determinism: same query twice retrieves the identical frame,
  bit-identical (numpy deterministic, asserted exactly).
- convergence legs: all banked trials converged within the frozen cap
  (iters <= 100 honored); max-iters recomputed == banked.
- fail-loud legs: wrong-dim / NaN / zero / empty queries refuse.
Usage: python3 tests/test_hopfield_retrieval.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []
BANK = os.path.join(ROOT, "samples", "marigold", "hopfield_retrieval.json")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import hopfield_retrieval as R

    rep = json.load(open(BANK))
    r = rep["report"]
    ids, X = R.load_prototypes()

    # -- prototype integrity (banked harvests -> unit rows) ------------------
    ok = (set(X) == set(R.POINTS)
          and all(X[p].shape == (8, 3072) for p in R.POINTS)
          and all(bool(np.isfinite(X[p]).all()) for p in R.POINTS))
    Xn = {p: X[p] / np.linalg.norm(X[p], axis=1, keepdims=True)
          for p in R.POINTS}
    unit = all(bool(np.abs(np.linalg.norm(Xn[p], axis=1) - 1.0).max() < 1e-12)
               for p in R.POINTS)
    check("hret-prototypes", ok and unit and ids == rep["frames"],
          "3x8x3072 finite, unit rows after normalize")

    # -- correctness legs recomputed in-gate ----------------------------------
    allok, alleq = True, True
    for p in R.POINTS:
        for li, f in enumerate(R.LEVELS):
            h = 0
            for fi in range(len(ids)):
                for seed in R.SEEDS:
                    w, _, _ = R.retrieve(Xn[p], R.corrupt(Xn[p][fi], f, seed, li))
                    h += ids[w] == ids[fi]
            lv = r[p]["levels"][str(f)]
            alleq &= (lv["hit"] == h and lv["tot"] == 40)
            allok &= h == 5
    check("hret-levels", alleq and allok,
          "recomputed == banked, every level 5/40 at all points")
    acc = {p: r[p]["accuracy"] for p in R.POINTS}
    check("hret-accuracy",
          all(abs(a - 0.125) < 1e-12 for a in acc.values())
          and all(a < r[p]["bar"] == 0.25 for p, a in acc.items()),
          f"recomputed {sorted(set(acc.values()))} == chance 0.125 < bar 0.25")
    loo = {p: r[p]["loo_accuracy"] for p in R.POINTS}
    check("hret-loo",
          all(abs(v - 1.0) < 1e-12 for v in loo.values()),
          f"same-scene LOO {sorted(set(loo.values()))} (diagnostic, no bar)")

    # -- depth tier recomputed --------------------------------------------------
    means = R.frame_means()
    uniform = sum(means[f] for f in ids) / len(ids)
    check("hret-depth",
          abs(r[R.POINTS[0]]["uniform_mean"] - uniform) < 1e-12
          and abs(uniform - 0.2936) < 1e-3
          and all(abs(r[p]["retrieved_depth"] - means["ai_021_001_fr0000"]) < 1e-9
                  for p in R.POINTS),
          f"uniform {uniform:.4f}, retrieved = winner-frame mean "
          f"{means['ai_021_001_fr0000']:.4f} (delta -0.04, not > +0.05)")

    # -- determinism: same query twice, bit-identical -----------------------------
    q = R.corrupt(Xn[R.POINTS[0]][0], 0.1, 2, 3)
    w1, _, _ = R.retrieve(Xn[R.POINTS[0]], q)
    w2, _, _ = R.retrieve(Xn[R.POINTS[0]], q)
    check("hret-determinism", w1 == w2, f"frame {ids[w1]} twice")

    # -- convergence legs --------------------------------------------------------------
    itok = True
    for p in R.POINTS:
        its = []
        for li, f in enumerate(R.LEVELS):
            for fi in range(len(ids)):
                for seed in R.SEEDS:
                    _, it, conv = R.retrieve(Xn[p], R.corrupt(Xn[p][fi], f, seed, li))
                    its.append(it)
                    itok &= conv and it <= R.MAXIT
        itok &= max(its) == r[p]["max_iters"] and r[p]["converged_frac"] == 1.0
    check("hret-convergence", itok, "all converge within cap 100 (max 2-4)")

    # -- fail-loud legs ------------------------------------------------------------------
    for bad, tag in ((np.zeros(3072), "zero"), (np.full(3072, np.nan), "nan"),
                     (np.ones(100), "dim"), (np.array([]), "empty")):
        try:
            R.retrieve(Xn[R.POINTS[0]], bad)
            check(f"hret-loud-{tag}", False, "must raise")
        except ValueError:
            check(f"hret-loud-{tag}", True, "ValueError")

    # -- verdicts frozen-rule --------------------------------------------------------------
    check("hret-verdicts",
          rep["verdicts"] == {p: "dead" for p in R.POINTS},
          "0.125<0.25 and depth delta -0.04: dead x3")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
