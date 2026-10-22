"""Hopfield retrieval dynamics over harvested prototypes (no training).

Banked 3072-dim prototypes -> L2-normalize -> three independent
8-pattern Ramsauer memories (one per harvest point) -> corrupted +
held-out queries. Pure functions (the gate recomputes from them);
main() runs the trial battery and banks
samples/marigold/hopfield_retrieval.json. CPU-only, numpy,
deterministic. Teacher/Qwen never loaded (banked data only).
Usage: python3 scripts/hopfield_retrieval.py
"""
import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIXDIR = os.path.join(ROOT, "samples", "marigold")

POINTS = ("transformer_blocks.29", "transformer_blocks.59", "norm_out")
LEVELS = (0.0, 0.01, 0.05, 0.1, 0.2)
SEEDS = (0, 1, 2, 3, 4)
BETA = 1.0
MAXIT = 100
TOL = 1e-6
CHANCE = 1.0 / 8.0
BAR = 0.25
LOO_CHANCE = 1.0 / 7.0


def load_prototypes():
    rep = json.load(open(os.path.join(FIXDIR, "hopfield_prototypes.json")))
    ids = list(rep["frames"])
    X = {p: np.array([rep["prototypes"][p][f] for f in ids]) for p in POINTS}
    return ids, X


def frame_means():
    base = json.load(open(os.path.join(FIXDIR, "teacher_baseline_H.json")))
    out = {}
    for f, d in base["frames"].items():
        vs = [v for v in d["strata"].values() if v is not None]
        out[f] = sum(vs) / len(vs)
    return out


def retrieve(Xn, q, beta=BETA, maxit=MAXIT, tol=TOL):
    """Ramsauer update to convergence/cap. Returns (winner, iters, conv)."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (Xn.shape[1],):
        raise ValueError(f"bad query dim {q.shape} (want {(Xn.shape[1],)})")
    if q.size == 0 or not np.all(np.isfinite(q)):
        raise ValueError("malformed query (empty/non-finite)")
    nq = np.linalg.norm(q)
    if nq == 0.0:
        raise ValueError("zero query (nothing observed)")
    xi = q / nq
    conv, it = False, 0
    for it in range(1, maxit + 1):
        s = Xn @ xi
        s = s - s.max()
        w = np.exp(beta * s)
        w = w / w.sum()
        nxt = Xn.T @ w
        nxt = nxt / np.linalg.norm(nxt)
        if float(np.abs(nxt - xi).max()) < tol:
            xi = nxt
            conv = True
            break
        xi = nxt
    return int(np.argmax(Xn @ xi)), it, conv


def corrupt(vec, f, seed, level_idx):
    rng = np.random.RandomState(1000 + 10 * seed + level_idx)
    if f == 0.0:
        return np.asarray(vec, dtype=np.float64).copy()
    rms = float(np.sqrt(np.mean(np.asarray(vec) ** 2)))
    return np.asarray(vec, dtype=np.float64) + rng.randn(vec.shape[0]) * f * rms


def scene_of(frame):
    return frame.rsplit("_fr", 1)[0]


def evaluate():
    ids, X = load_prototypes()
    means = frame_means()
    uniform = sum(means[f] for f in ids) / len(ids)
    out, verdicts = {}, {}
    for p in POINTS:
        Xn = X[p] / np.linalg.norm(X[p], axis=1, keepdims=True)
        lv, hit_all, tot_all = {}, 0, 0
        iters, convs = [], 0
        wdepth, wtot = 0.0, 0
        for li, f in enumerate(LEVELS):
            h = 0
            for fi in range(len(ids)):
                for seed in SEEDS:
                    q = corrupt(Xn[fi], f, seed, li)
                    w, it, conv = retrieve(Xn, q)
                    iters.append(it)
                    convs += conv
                    if ids[w] == ids[fi]:
                        h += 1
                    wdepth += means[ids[w]]
                    wtot += 1
            lv[str(f)] = {"hit": h, "tot": len(ids) * len(SEEDS)}
            hit_all += h
            tot_all += len(ids) * len(SEEDS)
        loo_h, loo_t = 0, 0
        for fi in range(len(ids)):
            others = [i for i in range(len(ids)) if i != fi]
            sims = Xn[others] @ Xn[fi]
            w = others[int(np.argmax(sims))]
            loo_t += 1
            loo_h += scene_of(ids[w]) == scene_of(ids[fi])
        acc = hit_all / tot_all
        rdepth = wdepth / wtot
        out[p] = {"levels": lv, "accuracy": acc, "trials": tot_all,
                  "chance": CHANCE, "bar": BAR,
                  "loo_accuracy": loo_h / loo_t, "loo_chance": LOO_CHANCE,
                  "converged_frac": convs / len(iters),
                  "max_iters": max(iters),
                  "retrieved_depth": rdepth,
                  "uniform_mean": uniform,
                  "depth_delta": uniform - rdepth}
        verdicts[p] = ("alive" if (acc >= BAR and uniform - rdepth > 0.05)
                       else "dead")
    return out, verdicts, ids


def main():
    out, verdicts, ids = evaluate()
    json.dump({"points": list(POINTS), "frames": ids, "beta": BETA,
               "maxit": MAXIT, "tol": TOL, "report": out,
               "verdicts": verdicts},
              open(os.path.join(FIXDIR, "hopfield_retrieval.json"), "w"),
              indent=1)
    for p in POINTS:
        r = out[p]
        print(f"{p}: " + " ".join(
            f"f={f}:{r['levels'][f]['hit']}/{r['levels'][f]['tot']}"
            for f in r["levels"]), flush=True)
        print(f"  acc={r['accuracy']:.4f} (bar {BAR}, chance {CHANCE}) "
              f"loo={r['loo_accuracy']:.3f} (chance {LOO_CHANCE:.3f}) "
              f"conv={r['converged_frac']:.3f} maxit={r['max_iters']} "
              f"rdepth={r['retrieved_depth']:.4f} uniform={r['uniform_mean']:.4f} "
              f"delta={r['depth_delta']:+.4f} verdict={verdicts[p]}", flush=True)


if __name__ == "__main__":
    main()
