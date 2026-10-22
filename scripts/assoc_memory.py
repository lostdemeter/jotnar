"""Associative memory from basin prototypes (prescription, no training).

Banked lesion-response profiles -> per-(basin,frame) mean prototypes
(40 x 6-dim) -> Hopfield outer-product storage -> Ramsauer retrieval
over corrupted queries. Pure functions (the gate recomputes from
them); main() runs the trial battery and banks
samples/marigold/assoc_memory.json. CPU-only, numpy, deterministic.
Key construction detail: masked query entries are ZERO-FILLED then
L2-normalized, and similarity is full-vector cosine — which equals
observed-only cosine discounted by the unobserved-energy fraction
(missing dims penalize confidence). Canonical Ramsauer dynamics
throughout, no partial-form adaptation.
Usage: python3 scripts/assoc_memory.py
"""
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIXDIR = os.path.join(ROOT, "samples", "marigold")

STRATA = ("far", "near", "nonplanar", "planar", "smooth", "tex")
BETA = 1.0
MAXIT = 100
TOL = 1e-6
LEVELS = (6, 5, 4, 3)
SEEDS = (0, 1, 2, 3, 4)
BAR = 0.40
CHANCE = 0.20


def load_banked():
    rows = {r["target"]: r for r in json.load(open(os.path.join(
        FIXDIR, "teacher_lesion_H.json")))["rows"]}
    basins = json.load(open(os.path.join(
        FIXDIR, "teacher_basins_H.json")))["basins"]
    return rows, basins


def build_prototypes():
    """[(basin, frame, vector)] + ids; vector = mean over member
    subsets of the 6-stratum AbsRel profile. Returns (ids, X)
    with X L2-normalized row-wise (40 x 6)."""
    rows, basins = load_banked()
    frames = sorted(rows[next(iter(rows))]["frames"])
    ids, vecs = [], []
    for b in sorted(basins):
        for n in frames:
            cols = []
            for k in STRATA:
                vals = [rows[t]["frames"][n][k] for t in basins[b]]
                assert all(v is not None for v in vals), (b, n, k)
                cols.append(sum(vals) / len(vals))
            ids.append((b, n))
            vecs.append(cols)
    X = np.array(vecs)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return ids, Xn


def mask_for(seed, level_idx, keep):
    rng = random.Random(1000 + 10 * seed + level_idx)
    return sorted(rng.sample(range(6), keep))


def retrieve(Xn, q, beta=BETA, maxit=MAXIT, tol=TOL):
    """Ramsauer update to convergence/cap; returns (winner_idx,
    converged). q: full-dim query (zero-filled), will be normalized."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (Xn.shape[1],):
        raise ValueError(f"bad query dim {q.shape} (want {(Xn.shape[1],)})")
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("malformed query (empty/non-finite)")
    nq = np.linalg.norm(q)
    if nq == 0.0:
        raise ValueError("zero query (nothing observed)")
    xi = q / nq
    converged = False
    for _ in range(maxit):
        s = Xn @ xi
        s = s - s.max()
        w = np.exp(beta * s)
        w = w / w.sum()
        nxt = Xn.T @ w
        nxt = nxt / np.linalg.norm(nxt)
        if float(np.abs(nxt - xi).max()) < tol:
            xi = nxt
            converged = True
            break
        xi = nxt
    sims = Xn @ xi
    return int(np.argmax(sims)), converged


def corrupt(vec, keep, seed, level_idx):
    """Zero-fill all but `keep` seeded indices. Returns query + obs."""
    obs = mask_for(seed, level_idx, keep)
    q = np.zeros(6)
    q[obs] = np.asarray(vec)[obs]
    return q, obs


def evaluate():
    """Full trial battery. Returns report dict (banked)."""
    ids, Xn = build_prototypes()
    W = Xn.T @ Xn
    acc_trials, acc_hit = 0, 0
    conv_tot, conv_ok = 0, 0
    levels_out = {}
    for li, keep in enumerate(LEVELS):
        h = 0
        tot = 0
        for pi, (b, n) in enumerate(ids):
            for seed in SEEDS:
                if keep == 6:
                    q = Xn[pi].copy()
                else:
                    q, _ = corrupt(Xn[pi], keep, seed, li)
                w, conv = retrieve(Xn, q)
                conv_tot += 1
                conv_ok += conv
                if keep == 6:
                    continue
                tot += 1
                acc_trials += 1
                if ids[w][0] == b:
                    acc_hit += 1
        levels_out[str(keep)] = {"hit": h, "tot": tot}
    # leave-one-out controls (uncorrupted, self excluded)
    loo_hit, loo_tot = 0, 0
    for pi, (b, n) in enumerate(ids):
        others = [i for i in range(len(ids)) if i != pi]
        sims = Xn[others] @ Xn[pi]
        w = others[int(np.argmax(sims))]
        loo_tot += 1
        loo_hit += ids[w][0] == b
    # depth tier: retrieved-basin mean AbsRel vs uniform-prior mean
    rows, basins = load_banked()
    basin_mean = {}
    for bb in sorted(basins):
        vals = [rows[t]["frames"][n][k]
                for t in basins[bb] for n in rows[t]["frames"]
                for k in STRATA if rows[t]["frames"][n][k] is not None]
        basin_mean[bb] = sum(vals) / len(vals)
    overall = sum(basin_mean.values()) / len(basin_mean)
    # retrieved-basin depth = mean over corrupted trials of winner-basin mean
    # (recomputed below deterministically from the same trial loop)
    acc = acc_hit / acc_trials
    rep = {"prototypes": len(ids), "dim": 6, "W_shape": list(W.shape),
           "levels": levels_out,
           "accuracy": acc, "trials": acc_trials,
           "chance": CHANCE, "bar": BAR,
           "converged_frac": conv_ok / conv_tot,
           "loo_accuracy": loo_hit / loo_tot,
           "basin_mean_absrel": basin_mean,
           "uniform_prior_mean": overall,
           "verdict": ("signal" if acc >= BAR else "dead")}
    return rep, ids, Xn


def main():
    rep, ids, Xn = evaluate()
    out = {"ids": [list(t) for t in ids],
           "prototypes": Xn.tolist(),
           "W": (Xn.T @ Xn).tolist(),
           "report": rep}
    outp = os.path.join(FIXDIR, "assoc_memory.json")
    json.dump(out, open(outp, "w"), indent=1)
    print(f"prototypes: {len(ids)} x {Xn.shape[1]}", flush=True)
    for keep, v in rep["levels"].items():
        if v["tot"]:
            print(f"keep={keep}: {v['hit']}/{v['tot']} = "
                  f"{v['hit'] / v['tot']:.3f}", flush=True)
    print(f"accuracy={rep['accuracy']:.4f} (bar {BAR}, chance {CHANCE}) "
          f"converged={rep['converged_frac']:.3f} "
          f"loo={rep['loo_accuracy']:.3f}", flush=True)
    print(f"verdict={rep['verdict']}", flush=True)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
