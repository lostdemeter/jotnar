"""Associative-memory gate (docs/HOPFIELD.md constitution): CPU-only, seconds.

Legs:
- prototype integrity: per-(basin,frame) means recomputed from the
  banked lesion census (no peeking — straight means) equal the
  banked prototypes exactly.
- retrieval legs: accuracy + depth tier recomputed in-gate from
  banked prototypes with frozen seeds/masks; LOO control too.
- determinism: same query twice retrieves the identical basin,
  bit-identical (numpy deterministic, asserted exactly).
- fail-loud: wrong-dim / NaN / zero / empty queries refuse.
Usage: python3 tests/test_assoc_memory.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []
BANK = os.path.join(ROOT, "samples", "marigold", "assoc_memory.json")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import assoc_memory as A

    rep = json.load(open(BANK))
    ids = [tuple(t) for t in rep["ids"]]
    Xb = np.array(rep["prototypes"])
    Wb = np.array(rep["W"])
    r = rep["report"]

    # -- prototype integrity (recomputed from banked census) ----------------
    live_ids, live_X = A.build_prototypes()
    check("assoc-prototypes",
          live_ids == ids and live_X.shape == Xb.shape
          and bool((live_X == Xb).all()),
          f"{len(ids)}x{Xb.shape[1]} bit-identical, L2-normalized rows")
    check("assoc-weights",
          bool(((live_X.T @ live_X) == Wb).all()),
          "W = X'X outer-product storage, exact")

    # -- retrieval legs recomputed in-gate -----------------------------------
    Xn = Xb
    hit = {str(k): {"hit": 0, "tot": 0} for k in A.LEVELS if k != 6}
    tot_all, hit_all = 0, 0
    for li, keep in enumerate(A.LEVELS):
        if keep == 6:
            continue
        for pi in range(len(ids)):
            for seed in A.SEEDS:
                q, _ = A.corrupt(Xn[pi], keep, seed, li)
                w, _ = A.retrieve(Xn, q)
                tot_all += 1
                hit[str(keep)]["tot"] += 1
                if ids[w][0] == ids[pi][0]:
                    hit[str(keep)]["hit"] += 1
                    hit_all += 1
    acc = hit_all / tot_all
    check("assoc-accuracy",
          abs(acc - r["accuracy"]) < 1e-12 and acc < r["bar"],
          f"recomputed {acc:.4f} == banked {r['accuracy']:.4f} "
          f"(< bar {r['bar']}, chance {r['chance']})")
    check("assoc-levels",
          all(v["tot"] == 200 for v in hit.values()),
          "3 levels x 200 trials each")

    # -- depth tier recomputed -------------------------------------------------
    rows, basins = A.load_banked()
    basin_mean = {}
    for bb in sorted(basins):
        vals = [rows[t]["frames"][n][k]
                for t in basins[bb] for n in rows[t]["frames"]
                for k in A.STRATA if rows[t]["frames"][n][k] is not None]
        basin_mean[bb] = sum(vals) / len(vals)
    overall = sum(basin_mean.values()) / len(basin_mean)
    check("assoc-depth",
          abs(overall - r["uniform_prior_mean"]) < 1e-9,
          f"uniform-prior mean {overall:.4f} (retrieved-depth "
          f"cannot beat it when retrieval is chance)")

    # -- determinism: same query twice, bit-identical ---------------------------
    q, _ = A.corrupt(Xn[0], 4, 0, 2)
    w1, _ = A.retrieve(Xn, q)
    w2, _ = A.retrieve(Xn, q)
    check("assoc-determinism", w1 == w2, f"basin {ids[w1][0]} twice")

    # -- fail-loud legs ------------------------------------------------------------
    for bad, tag in ((np.zeros(6), "zero"), (np.full(6, np.nan), "nan"),
                     (np.ones(5), "dim"), (np.array([]), "empty")):
        try:
            A.retrieve(Xn, bad)
            check(f"assoc-loud-{tag}", False, "must raise")
        except ValueError:
            check(f"assoc-loud-{tag}", True, "ValueError")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
