"""Separable prototypes (analysis + prescription, no training).

Three frozen constructions from banked L2-normalized harvests
(constitution in docs/HOPFIELD.md — no fourth construction exists):
  C1 ridge-ZCA (shared fit on all 24, eps=1e-4) applied per point.
  C2 per-point mean-centering.
  C3 sibling-subtraction (scene-contrastive, alpha=1 exact).
Separability audit per (construction, point); clearing constructions
earn the frozen 600-trial retrieval battery (same judge, bar 0.25).
Pure functions (gate recomputes from them); main() banks
samples/marigold/hopfield_separable.json. CPU-only, numpy, deterministic.
Usage: python3 scripts/hopfield_separable.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")

POINTS = ("transformer_blocks.29", "transformer_blocks.59", "norm_out")
EPS = 1e-4
MAXCOS_BAR = 0.9
MEANCOS_BAR = 0.5


def load_normed():
    rep = json.load(open(os.path.join(FIXDIR, "hopfield_prototypes.json")))
    ids = list(rep["frames"])
    out = {}
    for p in POINTS:
        X = np.array([rep["prototypes"][p][f] for f in ids])
        out[p] = X / np.linalg.norm(X, axis=1, keepdims=True)
    return ids, out


def norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if bool((n == 0.0).any()):
        raise ValueError("zero row after construction (void, fail loud)")
    return X / n


def fit_zca(allX, eps=EPS):
    C = np.cov(allX, rowvar=False)
    w, V = np.linalg.eigh(C)
    return (V * (1.0 / np.sqrt(w + eps))) @ V.T


def construct(ids, X, W=None):
    out = {}
    for p in POINTS:
        out[p + ":raw"] = X[p]
    if W is not None:
        for p in POINTS:
            out[p + ":c1"] = norm_rows(X[p] @ W)
    for p in POINTS:
        out[p + ":c2"] = norm_rows(X[p] - X[p].mean(axis=0, keepdims=True))
    for p in POINTS:
        sib = []
        for f in ids:
            scene = f.rsplit("_fr", 1)[0]
            other = [g for g in ids if g != f and g.rsplit("_fr", 1)[0] == scene]
            assert len(other) == 1, (f, other)
            sib.append(X[p][ids.index(other[0])])
        out[p + ":c3"] = norm_rows(X[p] - np.array(sib))
    return out


def sepscore(M):
    G = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return float(G[iu].max()), float(G[iu].mean())


def main():
    import hopfield_retrieval as R
    ids, X = load_normed()
    allX = np.array([X[p] for p in POINTS]).reshape(-1, X[POINTS[0]].shape[1])
    W = fit_zca(allX)
    M = construct(ids, X, W)
    print(f"whitener: shared fit on {len(allX)} patterns, eps={EPS}", flush=True)

    seps, cleared = {}, []
    for key, Mp in M.items():
        mx, mn = sepscore(Mp)
        ok = mx < MAXCOS_BAR and mn < MEANCOS_BAR
        seps[key] = {"maxcos": mx, "meancos": mn, "clears": ok}
        print(f"{key}: maxcos={mx:.4f} meancos={mn:+.4f} "
              f"{'CLEARS' if ok else 'fails'}", flush=True)
        if key.endswith(":c1") or key.endswith(":c2") or key.endswith(":c3"):
            if ok:
                cleared.append(key)

    constr_cleared = sorted({k.rsplit(":", 1)[1] for k in cleared})
    full_clear = [c for c in ("c1", "c2", "c3")
                  if all(seps[f"{p}:{c}"]["clears"] for p in POINTS)]
    print(f"constructions clearing all 3 points: {full_clear}", flush=True)

    reruns = {}
    for c in full_clear:
        for p in POINTS:
            Mp = M[f"{p}:{c}"]
            hit_all, tot_all, wdepth, wtot = 0, 0, 0.0, 0
            iters, convs = [], 0
            for li, f in enumerate(R.LEVELS):
                h = 0
                for fi in range(len(ids)):
                    for seed in R.SEEDS:
                        w, it, conv = R.retrieve(
                            Mp, R.corrupt(Mp[fi], f, seed, li))
                        iters.append(it)
                        convs += conv
                        h += ids[w] == ids[fi]
                hit_all += h
                tot_all += 40
            means = R.frame_means()
            for li, f in enumerate(R.LEVELS):
                for fi in range(len(ids)):
                    for seed in R.SEEDS:
                        w, _, _ = R.retrieve(
                            Mp, R.corrupt(Mp[fi], f, seed, li))
                        wdepth += means[ids[w]]
                        wtot += 1
            loo_h = sum(
                R.scene_of(ids[([i for i in range(8) if i != fi][
                    int(np.argmax(Mp[[i for i in range(8) if i != fi]]
                                    @ Mp[fi]))])]) == R.scene_of(ids[fi])
                for fi in range(8))
            uniform = sum(means[f] for f in ids) / len(ids)
            acc = hit_all / tot_all
            reruns[f"{p}:{c}"] = {
                "accuracy": acc, "bar": R.BAR, "chance": R.CHANCE,
                "loo_accuracy": loo_h / 8, "converged_frac": convs / len(iters),
                "max_iters": max(iters), "retrieved_depth": wdepth / wtot,
                "uniform_mean": uniform,
                "verdict": ("alive" if acc >= R.BAR else "dead")}
            print(f"rerun {p}:{c}: acc={acc:.4f} (bar {R.BAR}) "
                  f"loo={loo_h / 8:.3f} rdepth={wdepth / wtot:.4f} "
                  f"verdict={reruns[f'{p}:{c}']['verdict']}", flush=True)

    json.dump({"eps": EPS, "bar": [MAXCOS_BAR, MEANCOS_BAR],
               "separability": seps, "full_clear": full_clear,
               "reruns": reruns},
              open(os.path.join(FIXDIR, "hopfield_separable.json"), "w"),
              indent=1)
    print("wrote samples/marigold/hopfield_separable.json", flush=True)


if __name__ == "__main__":
    main()
