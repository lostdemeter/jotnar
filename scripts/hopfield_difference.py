"""Difference-preserving prototypes + manifold diagnostics (no training).

D1 max-deviation exemplars / D2 variance-weighted pooling / D3
farthest-point sets — all from banked means + banked 432-token subset
(full captures gone, stated). NN-distance + intrinsic-dimension
diagnostics characterize (never gate). Conditional battery per clearing
construction. Pure functions (gate recomputes from them); main() banks
samples/marigold/hopfield_difference.json. CPU-only, numpy,
deterministic. Teacher/Qwen never loaded.
Usage: python3 scripts/hopfield_difference.py
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


def load_all():
    import hopfield_separable as S
    ids, X = S.load_normed()
    trep = json.load(open(os.path.join(FIXDIR, "hopfield_tokens.json")))
    T = np.load(os.path.join(FIXDIR, "hopfield_tokens.npy")).astype(np.float64)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    rows = [tuple(x) for x in trep["index_rows"]]
    sub = {}
    for p in POINTS:
        for fi, f in enumerate(ids):
            idx = [i for i, r in enumerate(rows) if r[0] == p and r[1] == f]
            assert len(idx) == 18, (p, f, len(idx))
            sub[(p, f)] = T[idx]
    return ids, X, sub


def norm_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    if bool((n == 0.0).any()):
        raise ValueError("zero row in construction (void, fail loud)")
    return M / n


def construct(ids, X, sub):
    out = {}
    for p in POINTS:
        d1, d2, d3pool = [], [], []
        for f in ids:
            m, T18 = X[p][ids.index(f)], sub[(p, f)]
            d = np.linalg.norm(T18 - m[None, :], axis=1)
            d1.append(T18[int(np.argmax(d))])
            w = d ** 2
            if float(w.sum()) == 0.0:
                raise ValueError(f"zero weights {(p, f)} (void)")
            d2.append((w[:, None] * T18).sum(axis=0) / float(w.sum()))
            d3pool.append(T18)
        out[p + ":d1"] = norm_rows(np.array(d1))
        out[p + ":d2"] = norm_rows(np.array(d2))
        A = np.array(d3pool).reshape(-1, 3072)
        mu = A.mean(axis=0)
        sel = [int(np.argmax(np.linalg.norm(A - mu[None, :], axis=1)))]
        for _ in range(7):
            dd = np.min([np.linalg.norm(A - A[s][None, :], axis=1)
                         for s in sel], axis=0)
            dd[sel] = -1.0
            sel.append(int(np.argmax(dd)))
        out[p + ":d3"] = norm_rows(A[sel])
        out[p + ":d3_idx"] = np.array(sel)
    return out


def sepscore(M):
    G = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return float(G[iu].max()), float(G[iu].mean())


def diagnostics(ids, X):
    A = np.array([X[p] for p in POINTS]).reshape(-1, 3072)
    G = A @ A.T
    np.fill_diagonal(G, -2.0)
    nn = 1.0 - G.max(axis=1)
    rep = {"nn_min": float(nn.min()), "nn_median": float(np.median(nn)),
           "nn_max": float(nn.max())}
    w = np.linalg.eigvalsh(np.cov(A, rowvar=False))
    w = w[w > 0]
    rep["PR_24"] = float(w.sum() ** 2 / (w ** 2).sum())
    rep["eigfrac_24"] = [float((w >= 0.01 * w.max()).sum()), int(len(w))]
    rep["per_point"] = {}
    for p in POINTS:
        v = np.linalg.eigvalsh(np.cov(X[p], rowvar=False))
        v = v[v > 0]
        rep["per_point"][p] = {
            "PR": float(v.sum() ** 2 / (v ** 2).sum()),
            "eigfrac": [float((v >= 0.01 * v.max()).sum()), int(len(v))]}
    rep["thin_confirmed"] = bool(rep["PR_24"] <= 5
                                 and rep["nn_median"] < 0.05)
    return rep


def main():
    import hopfield_retrieval as R
    ids, X, sub = load_all()
    M = construct(ids, X, sub)
    seps, full = {}, []
    for key in [k for k in M if not k.endswith("idx")]:
        mx, mn = sepscore(M[key])
        ok = mx < 0.9 and mn < 0.5
        seps[key] = {"maxcos": mx, "meancos": mn, "clears": ok}
        print(f"{key}: maxcos={mx:.4f} meancos={mn:+.4f} "
              f"{'CLEARS' if ok else 'fails'}", flush=True)
    for c in ("d1", "d2", "d3"):
        if all(seps[f"{p}:{c}"]["clears"] for p in POINTS):
            full.append(c)
    print(f"constructions clearing all 3 points: {full}", flush=True)
    diag = diagnostics(ids, X)
    print(f"manifold: nn min/med/max = {diag['nn_min']:.2e} / "
          f"{diag['nn_median']:.2e} / {diag['nn_max']:.2e}, "
          f"PR_24 = {diag['PR_24']:.2f}, "
          f"thin_confirmed = {diag['thin_confirmed']}", flush=True)
    battery = {}
    for c in full:
        for p in POINTS:
            Mp = M[f"{p}:{c}"]
            hit = tot = 0
            lv = {}
            for li, f in enumerate(R.LEVELS):
                h = 0
                for fi in range(8):
                    for s in R.SEEDS:
                        w, _, _ = R.retrieve(
                            Mp, R.corrupt(Mp[fi], f, s, li))
                        h += ids[w] == ids[fi]
                        tot += 1
                lv[str(f)] = {"hit": h, "tot": 40}
                hit += h
            acc = hit / tot
            battery[f"{p}:{c}"] = {
                "levels": lv, "accuracy": acc, "trials": tot,
                "chance": R.CHANCE, "bar": R.BAR,
                "verdict": ("alive" if acc >= R.BAR else "dead")}
            print(f"battery {p}:{c}: acc={acc:.4f} "
                  f"verdict={battery[f'{p}:{c}']['verdict']}", flush=True)
    if not full:
        print("empty: NO battery runs (stated)", flush=True)
    json.dump({"constructions": ["d1", "d2", "d3"],
               "separability": seps, "full_clear": full,
               "diagnostics": diag, "battery": battery,
               "d3_idx": {p: M[f"{p}:d3_idx"].tolist() for p in POINTS}},
              open(os.path.join(FIXDIR, "hopfield_difference.json"), "w"),
              indent=1)
    print("wrote samples/marigold/hopfield_difference.json", flush=True)


if __name__ == "__main__":
    main()
