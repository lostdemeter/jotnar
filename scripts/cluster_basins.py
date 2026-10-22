"""Phase-A basin ground truth: exhaustive k-medoids on lesion responses.

8 Hypersim frames x (15 lesion targets x 6 strata AbsRel) = 90-dim
vectors, cosine distance, exhaustive medoid search per k in 3..7,
winner = max mean silhouette. Deterministic (no seed dependence).
Banks samples/marigold/basin_ground_truth.json. CPU-only, numpy.
Usage: python3 scripts/cluster_basins.py
"""
import itertools
import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIXDIR = os.path.join(ROOT, "samples", "marigold")

STRATA = ("far", "near", "nonplanar", "planar", "smooth", "tex")
KS = (3, 4, 5, 6, 7)
SEED = 7


def load_vectors():
    rep = json.load(open(os.path.join(FIXDIR, "teacher_lesion_H.json")))
    rows = rep["rows"]
    assert len(rows) == 15, len(rows)
    frames = list(rows[0]["frames"])
    assert len(frames) == 8, frames
    V = []
    for f in frames:
        V.append([rows[t]["frames"][f][s] for t in range(15) for s in STRATA])
    V = np.array(V, dtype=np.float64)
    assert np.isfinite(V).all(), "lesion vectors must be finite"
    return frames, V


def cosdist(A):
    An = A / np.linalg.norm(A, axis=1, keepdims=True)
    return np.clip(1.0 - An @ An.T, 0.0, 2.0)


def medoids_exhaustive(D, k):
    best, bestcost = None, np.inf
    for med in itertools.combinations(range(len(D)), k):
        cost = D[:, med].min(axis=1).sum()
        if cost < bestcost - 1e-12:
            bestcost, best = float(cost), med
    labels = np.argmin(D[:, best], axis=1)
    order = {m: i for i, m in enumerate(sorted(best))}
    return (tuple(sorted(best)),
            np.array([order[best[l]] for l in labels]), bestcost)


def silhouette(D, labels):
    s = []
    for i in range(len(D)):
        same = [j for j in range(len(D)) if labels[j] == labels[i] and j != i]
        a = float(np.mean([D[i, j] for j in same])) if same else 0.0
        others = {int(v) for v in labels if v != labels[i]}
        b = min(float(np.mean([D[i, j] for j in range(len(D))
                               if labels[j] == o])) for o in others)
        s.append(0.0 if not same else (b - a) / max(a, b))
    return float(np.mean(s))


def main():
    frames, V = load_vectors()
    D = cosdist(V)
    fit = {}
    for k in KS:
        med, labels, cost = medoids_exhaustive(D, k)
        sil = silhouette(D, labels)
        fit[str(k)] = {"medoids": list(med),
                       "labels": {f: int(l) for f, l in zip(frames, labels)},
                       "cost": cost, "silhouette": sil}
        print(f"k={k}: silhouette={sil:.4f} cost={cost:.4f} medoids={list(med)}",
              flush=True)
        print(f"   labels: {fit[str(k)]['labels']}", flush=True)
    winner = max(KS, key=lambda k: fit[str(k)]["silhouette"])
    print(f"winner k={winner} (silhouette={fit[str(winner)]['silhouette']:.4f})",
          flush=True)
    json.dump({"method": "exhaustive-k-medoids", "distance": "cosine",
               "dims": "15 targets x 6 strata = 90, raw AbsRel, no baseline",
               "seed": SEED, "k_fit": fit, "winner_k": winner,
               "labels": fit[str(winner)]["labels"]},
              open(os.path.join(FIXDIR, "basin_ground_truth.json"), "w"),
              indent=1)
    print("wrote samples/marigold/basin_ground_truth.json", flush=True)


if __name__ == "__main__":
    main()
