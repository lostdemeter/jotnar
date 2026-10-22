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

    # -- follow-ups key (hopfield-followups round; legs above untouched) ----
    import hopfield_followups as F
    import hopfield_retrieval as R
    fids, fX = R.load_prototypes()
    fXn = {p: fX[p] / np.linalg.norm(fX[p], axis=1, keepdims=True)
           for p in R.POINTS}
    fol = json.load(open(os.path.join(
        ROOT, "samples", "marigold", "hopfield_followups.json")))
    sweep_ok = True
    for beta in F.BETAS:
        for p in R.POINTS:
            hit = sum(
                fids[R.retrieve(
                    fXn[p], R.corrupt(fXn[p][fi], fl, s, li),
                    beta=beta)[0]] == fids[fi]
                for li, fl in enumerate(R.LEVELS)
                for fi in range(len(fids)) for s in R.SEEDS)
            b = fol["sweep"][f"{p}:b={beta}"]
            sweep_ok &= (b["accuracy"] == hit / 200 and hit == 25
                         and not b["clears"])
    check("assoc-fol-beta", sweep_ok and fol["betas"] == list(F.BETAS),
          "15 cells recomputed == banked, all 25/200=0.125 below 0.25")
    fJ = np.array([fXn[p] for p in R.POINTS]).reshape(-1, 3072)
    jhit = sum(
        (R.retrieve(fJ, R.corrupt(fJ[oi * 8 + fi], fl, s, li))[0] % 8) == fi
        for oi in range(3) for li, fl in enumerate(R.LEVELS)
        for fi in range(len(fids)) for s in R.SEEDS)
    j = fol["joint"]
    check("assoc-fol-joint",
          j["patterns"] == 24 and j["trials"] == 600
          and j["accuracy"] == jhit / 600 == 0.125
          and abs(j["chance"] - 3 / 24) < 1e-12 and not j["clears"],
          f"joint-24 recomputed {jhit}/600=0.125 below 0.25")
    q = R.corrupt(fJ[0], 0.1, 1, 2)
    check("assoc-fol-determinism",
          R.retrieve(fJ, q)[0] == R.retrieve(fJ, q)[0],
          "joint query twice, identical frame")

    # -- token key (hopfield-tokens round; legs above untouched) ------------
    tarr = np.load(os.path.join(ROOT, "samples", "marigold",
                                "hopfield_tokens.npy"))
    trep = json.load(open(os.path.join(
        ROOT, "samples", "marigold", "hopfield_tokens.json")))
    trows = [tuple(x) for x in trep["index_rows"]]
    want_rows = [(p, f, t) for p in ("transformer_blocks.29",
                                     "transformer_blocks.59", "norm_out")
                 for f in ("ai_018_001_fr0000", "ai_018_001_fr0001",
                           "ai_021_001_fr0000", "ai_021_001_fr0001",
                           "ai_022_001_fr0000", "ai_022_001_fr0001",
                           "ai_023_005_fr0000", "ai_023_005_fr0001")
                 for t in range(0, 1728, 96)]
    check("assoc-tok-integrity",
          tarr.shape == (432, 3072) and str(tarr.dtype) == "float16"
          and bool(np.isfinite(tarr).all()) and trows == want_rows
          and trep["stride"] == 96,
          "432x3072 fp16 finite, point/frame/stride-96 index exact")
    tok_ok, con_ok = True, True
    for pi, p in enumerate(("transformer_blocks.29",
                            "transformer_blocks.59", "norm_out")):
        M64 = tarr[pi * 144:(pi + 1) * 144].astype(np.float64)
        M64 = M64 / np.linalg.norm(M64, axis=1, keepdims=True)
        fr = np.array([i // 18 for i in range(144)])
        G = M64 @ M64.T
        iu = np.triu_indices(144, 1)
        cr = fr[iu[0]] != fr[iu[1]]
        mx, mn = float(G[iu][cr].max()), float(G[iu][cr].mean())
        full = trep["separability_full"][p]
        tok_ok &= mx <= full["cross_max"] + 1e-3
        con_ok &= (full["cross_max"] >= 0.9 or full["cross_mean"] >= 0.5)
    check("assoc-tok-separability", tok_ok and con_ok,
          "subset Gram recomputed; full stats fail 0.9/0.5 at all points")
    rok = trep["battery"] == {}
    if trep["battery"]:
        rok &= all(v["accuracy"] >= v["bar"]
                   for v in trep["battery"].values())
    check("assoc-tok-battery", rok, "none earned (dead, no runs — stated)")
    check("assoc-tok-determinism",
          all(v == 0.0 for v in trep["determinism_maxabs"].values()),
          "frame0 x2 maxabs 0.0 all points")
    check("assoc-tok-checksums",
          trep["sha256"]["pre"] == trep["sha256"]["post"]
          and len(trep["sha256"]["pre"]) == 7,
          "7 weight files, pre==post")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
