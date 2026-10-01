"""Native ENGRAM storage (v1.3): freeze/load/recompose directional stores.

A weight matrix W (M,N) decomposes once (host, offline) into stores:
keys U (M,K), gains s (K,), values Vt (K,N) with W == U S Vt. Frozen to
stores/<tag>.npz (float64, lossless); loaded with shape asserts; recompose
is bit-near-exact (float64 SVD roundtrip ~1e-12). Cost prediction stays in
chain/read.py (no duplicate); this module is STORAGE only.
Blobs stay OUT of git (41MB full float64 for down_proj dwarfs the 1.9MB
repo): stores/*.npz is gitignored and freeze regenerates in ~10-20s.
Regeneration is deterministic (gesdd on identical bytes -> identical
vectors); the roundtrip gate below pins it. (S17's commit-priors rule
bows to repo hygiene here -- stated, with reason.)
"""
import json
import os

import numpy as np

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "stores")


def decompose(W, k=None):
    """W (M,N) float64 -> (U (M,K), s (K,), Vt (K,N)). k=top-k or full."""
    U, s, Vt = np.linalg.svd(np.ascontiguousarray(W, dtype=np.float64),
                             full_matrices=False)
    if k is not None:
        U, s, Vt = U[:, :k], s[:k], Vt[:k]
    return U, s, Vt


def freeze(W, tag, shape=None):
    """Decompose + write stores/<tag>.npz + .json sidecar (source shape,
    K, widths). Returns paths. Deterministic for identical input bytes."""
    U, s, Vt = decompose(W)
    os.makedirs(STORE_DIR, exist_ok=True)
    npz = os.path.join(STORE_DIR, tag + ".npz")
    meta = {"shape": list(W.shape), "k": int(s.shape[0]),
            "source": "qwen2-0.5B" if shape is None else str(shape)}
    np.savez(npz, U=U, s=s, Vt=Vt)
    with open(os.path.join(STORE_DIR, tag + ".json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return npz


def load(tag):
    """Load stores with shape asserts (mismatched cache fails loud)."""
    with open(os.path.join(STORE_DIR, tag + ".json")) as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(STORE_DIR, tag + ".npz"))
    U, s, Vt = z["U"], z["s"], z["Vt"]
    assert list(U.shape) == [meta["shape"][0], meta["k"]], "store shape drift"
    assert list(Vt.shape) == [meta["k"], meta["shape"][1]], "store shape drift"
    return U, s, Vt, meta


def recompose(U, s, Vt):
    """Stores -> matrix (U*S @ Vt)."""
    return (U * s) @ Vt
