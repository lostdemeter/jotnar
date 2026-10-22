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

    # -- separability key (hopfield-separable round; existing legs above
    #    untouched) ---------------------------------------------------------------
    import hopfield_separable as S
    sep = json.load(open(os.path.join(
        ROOT, "samples", "marigold", "hopfield_separable.json")))
    sids, sX = S.load_normed()
    sall = np.array([sX[p] for p in S.POINTS]).reshape(-1, 3072)
    sW = S.fit_zca(sall)
    sM = S.construct(sids, sX, sW)
    seq = True
    for key, Mp in sM.items():
        mx, mn = S.sepscore(Mp)
        b = sep["separability"][key]
        seq &= (abs(mx - b["maxcos"]) < 1e-9
                and abs(mn - b["meancos"]) < 1e-9
                and b["clears"] == (mx < 0.9 and mn < 0.5))
    check("assoc-sep-integrity", seq,
          "C1/C2/C3 recomputed from banked harvests == banked table")
    check("assoc-sep-bar",
          sep["full_clear"] == []
          and sorted(sep["separability"]) == sorted(sM)
          and [k for k, v in sep["separability"].items() if v["clears"]]
          == ["transformer_blocks.59:c2", "transformer_blocks.29:c3",
              "transformer_blocks.59:c3"],
          "no construction clears all 3 points (nearest miss C3 2/3)")
    rok = sep["reruns"] == {}
    if sep["reruns"]:
        import hopfield_retrieval as _R
        for key, rr in sep["reruns"].items():
            rok &= rr["accuracy"] >= rr["bar"]
    check("assoc-sep-reruns", rok,
          "none earned (vacuous); any clearing construction must beat 0.25")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
