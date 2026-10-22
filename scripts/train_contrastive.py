"""Contrastive separation trainer (analysis, numpy CPU, no emission).

ONE shared linear P: 3072->512 (no bias/activation), triplet margin on
cosines over the 24 banked harvest patterns. All math is MATMUL / MUL /
ADD / SUB / host-side scalar max (see constitution for per-op cites).
P + state live OUT-OF-TREE (~/student_runs/contrastive/); metrics +
projected codes banked in samples/marigold/. Deterministic (seed 7).
Usage: python3 scripts/train_contrastive.py
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTDIR = os.path.expanduser("~/student_runs/contrastive")

SEED = 7
DIM_OUT = 512
MARGIN = 0.5
LR = 0.1
STEPS = 5000
POINTS = ("transformer_blocks.29", "transformer_blocks.59", "norm_out")


def load24():
    import hopfield_separable as S
    ids, X = S.load_normed()
    order = [(p, f) for p in POINTS for f in ids]
    return ids, np.array([X[p][ids.index(f)] for p, f in order]), order


def triplets(order):
    T = []
    for a, (p, f) in enumerate(order):
        pos = [i for i, (pp, ff) in enumerate(order) if ff == f and pp != p]
        neg = [i for i, (pp, ff) in enumerate(order) if ff != f]
        assert len(pos) == 2 and len(neg) == 21, (a, len(pos), len(neg))
        for q in pos:
            for n in neg:
                T.append((a, q, n))
    return np.array(T)


def train(X, T, seed=SEED, steps=STEPS, lr=LR, margin=MARGIN):
    rng = np.random.RandomState(seed)
    P = rng.randn(X.shape[1], DIM_OUT) / np.sqrt(X.shape[1])
    nT = len(T)
    curve = {}
    for it in range(1, steps + 1):
        Z = X @ P
        zn = np.linalg.norm(Z, axis=1, keepdims=True)
        Zn = Z / zn
        S = Zn @ Zn.T
        d = margin + S[T[:, 0], T[:, 2]] - S[T[:, 0], T[:, 1]]
        viol = d > 0
        loss = float(d[viol].sum() / nT)
        dS = np.zeros_like(S)
        np.add.at(dS, (T[viol, 0], T[viol, 2]), 1.0 / nT)
        np.add.at(dS, (T[viol, 0], T[viol, 1]), -1.0 / nT)
        dZn = (dS + dS.T) @ Zn
        dZ = (dZn - Zn * (dZn * Zn).sum(axis=1, keepdims=True)) / zn
        P = P - lr * (X.T @ dZ)
        if it in (1, 500, 1000, 2000, 3000, 4000, 5000):
            curve[str(it)] = loss
            print(f"step {it}: loss={loss:.4f} viol={viol.mean():.3f}",
                  flush=True)
    return P, curve


def sepscore(M):
    G = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return float(G[iu].max()), float(G[iu].mean())


def main():
    import hopfield_retrieval as R
    t0 = time.time()
    ids, X, order = load24()
    T = triplets(order)
    print(f"patterns={len(X)} triplets={len(T)} "
          f"(24 anchors x 2 pos x 21 neg)", flush=True)
    P, curve = train(X, T)
    os.makedirs(OUTDIR, exist_ok=True)
    pp = os.path.join(OUTDIR, "P.npy")
    np.save(pp, P)
    sha = hashlib.sha256(open(pp, "rb").read()).hexdigest()
    Z = X @ P
    C = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    seps, clears = {}, []
    for pi, p in enumerate(POINTS):
        M8 = C[pi * 8:(pi + 1) * 8]
        mx, mn = sepscore(M8)
        ok = mx < 0.9 and mn < 0.5
        seps[p] = {"maxcos": mx, "meancos": mn, "clears": ok}
        print(f"{p}: maxcos={mx:.4f} meancos={mn:+.4f} "
              f"{'CLEARS' if ok else 'fails'}", flush=True)
        clears.append(ok)
    np.save(os.path.join(FIXDIR, "contrastive_codes.npy"),
            C.astype(np.float16))
    battery = {}
    if all(clears):
        for pi, p in enumerate(POINTS):
            Mp = C[pi * 8:(pi + 1) * 8]
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
            battery[p] = {"levels": lv, "accuracy": acc, "trials": tot,
                          "chance": R.CHANCE, "bar": R.BAR,
                          "verdict": ("alive" if acc >= R.BAR else "dead")}
            print(f"battery {p}: acc={acc:.4f} "
                  f"verdict={battery[p]['verdict']}", flush=True)
    else:
        print("separability fails: NO battery runs (stated)", flush=True)
    json.dump({"seed": SEED, "dim_out": DIM_OUT, "margin": MARGIN,
               "lr": LR, "steps": STEPS, "wall_s": round(time.time() - t0, 1),
               "loss_curve": curve, "P_path": pp, "P_sha256": sha,
               "P_shape": list(P.shape),
               "separability": seps, "battery": battery},
              open(os.path.join(FIXDIR, "contrastive_separation.json"), "w"),
              indent=1)
    print(f"wall {time.time() - t0:.1f}s; P out-of-tree ({pp}); "
          f"metrics banked", flush=True)


if __name__ == "__main__":
    main()
