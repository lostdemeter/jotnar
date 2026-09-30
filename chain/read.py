"""Model read instrument (v1.3): directional readout tables + searches.

A read of a weight matrix is a TABLE, one row per singular direction:
sval, global ablation dB, and the per-token delta vector (the functional
fingerprint). Searches over the table: dead_shelves (removable storage),
movers (which directions move token t), selectivity (broadcasters vs
specialists). Fingerprints are POSITIONAL on the read input, not semantic
labels -- a direction moving position 3 here moves whatever sits at
position 3. Whether fingerprints stabilize across contexts is the labeling
loop's question, not this instrument's claim.
Numpy-only: takes base output + run_fn(ablated W) -> output; no ASM inside.
"""
import numpy as np


def direction_readout(base, run_fn, W, idx=None):
    """Full readout over sampled directions. run_fn(W_ablated) -> output
    array (..., D). Returns dict(idx, sval, gdb, tokdb): tokdb[i] is the
    per-token delta-dB vector of direction idx[i] (peak=1.0 basis)."""
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    if idx is None:
        idx = list(range(len(s)))
    base = np.ascontiguousarray(base, dtype=np.float64)
    rows = []
    for i in idx:
        Wi = W - np.outer(U[:, i] * s[i], Vt[i])
        got = np.ascontiguousarray(run_fn(Wi), dtype=np.float64)
        mse_tok = ((got - base) ** 2).mean(-1)
        tokdb = np.array([float("inf") if v == 0 else 10 * np.log10(1.0 / v)
                          for v in mse_tok])
        mse = float(((got - base) ** 2).mean())
        rows.append((s[i], float("inf") if mse == 0 else 10 * np.log10(1.0 / mse),
                     tokdb))
    return {"idx": np.array(idx), "sval": np.array([r[0] for r in rows]),
            "gdb": np.array([r[1] for r in rows]),
            "tokdb": np.stack([r[2] for r in rows])}


def dead_shelves(ro, thresh=55.0):
    """Directions removable above thresh dB, most-removable first."""
    o = np.argsort(-ro["gdb"])
    return [(int(ro["idx"][i]), float(ro["gdb"][i]), float(ro["sval"][i]))
            for i in o if ro["gdb"][i] > thresh]


def movers(ro, token, k=5):
    """Top-k directions moving token t (lowest token-dB first)."""
    o = np.argsort(ro["tokdb"][:, token])[:k]
    return [(int(ro["idx"][i]), round(float(ro["tokdb"][i, token]), 1))
            for i in o]


def selectivity(ro):
    """Per direction: (spread max-min, argmin token). High spread with a
    low global dB = specialist; low spread = broadcaster (or dead)."""
    t = ro["tokdb"]
    return [(int(idx), float(v.max() - v.min()), int(np.argmin(v)))
            for idx, v in zip(ro["idx"], t)]
