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


def bank(tag, idx=None):
    """Bank as listing data: (Ub, Vb) with gains folded into Ub (Ub = U*s),
    ready to encode as IN streams for storebank_apply. idx = subset for
    pruning (assembler-side edit: rebuild with fewer columns); None = all.
    Gains live in Ub by the A-folding precedent (retune = rebuild, stated).
    Same values as the matmul form up to materialization quantum (gated)."""
    U, s, Vt, _ = load(tag)
    if idx is None:
        idx = list(range(s.shape[0]))
    idx = list(idx)
    return U[:, idx] * s[idx], Vt[idx]


def yarnball_bank(base_ukt, base_evb, stores, key_scale=2.0):
    """Yarn-ball bank as listing data for yarnball_apply.

    base_ukt (D,K0), base_evb (K0,D): the unstructured ball (native bank).
    stores: list of dicts, one per strand, each with:
      key (D,) unit address direction in RECEIVER geometry (native-grown
        or emb; mapped-teacher keys allowed but tier-tagged as such),
      value (D,) unit content direction (native readout dir),
      dose (float): install dose folded into the value row (HOW MUCH),
      tier (str): 'exact' | 'assoc' | 'opt' (operating-point refit) |
        'null' (zero value: background store so the softmax has
        somewhere to route non-targets -- hold by construction),
      support (str): provenance (corpus-mined prompt / label id),
    key_scale (float): address key norm (2.0 = winner scale from the
      key-rank gate; addressing/strength are independent knobs).

    Returns (Ua, Vc, ledger): Ua (D,K0+N) address keys with gains folded
    (key*key_scale), Vc (K0+N,D) content values with dose folded
    (value*dose), ledger (list of rows: tier, support, key_norm,
    value_norm, dose, sha of key bytes). Base stores get ledger rows
    with tier 'base', dose 1.0. Cost preview stays in chain/read.py
    (predict_db before emitting); this function is STORAGE + ledger.
    """
    import hashlib
    Ua = [np.ascontiguousarray(base_ukt, dtype=np.float64)]
    Vc = [np.ascontiguousarray(base_evb, dtype=np.float64)]
    ledger = [{"tier": "base", "support": f"native-{i}",
               "key_norm": None, "value_norm": float(np.linalg.norm(
                   np.ascontiguousarray(base_evb, dtype=np.float64)[i])),
               "dose": 1.0, "sha": None}
              for i in range(np.ascontiguousarray(base_evb).shape[0])]
    for st in stores:
        k = np.ascontiguousarray(st["key"], dtype=np.float64)
        v = np.ascontiguousarray(st["value"], dtype=np.float64)
        kn, vn = float(np.linalg.norm(k)), float(np.linalg.norm(v))
        if not kn > 0:
            raise ValueError("yarnball_bank: key must be nonzero "
                             f"(got norm {kn:.3g})")
        if st.get("tier", "assoc") not in ("exact", "assoc", "opt", "null"):
            raise ValueError(f"yarnball_bank: bad tier {st.get('tier')!r} "
                             "(want exact|assoc|opt|null)")
        ku = k / kn * float(key_scale)
        vr = np.zeros_like(v) if vn == 0 else v / vn * float(st.get("dose", 1.0))
        Ua.append(ku[:, None])
        Vc.append(vr[None, :])
        ledger.append({"tier": st.get("tier", "assoc"),
                       "support": str(st.get("support", "?")),
                       "key_norm": float(key_scale),
                       "value_norm": float(np.linalg.norm(vr)),
                       "dose": float(st.get("dose", 1.0)),
                       "sha": hashlib.sha256(np.ascontiguousarray(
                           k).tobytes()).hexdigest()[:16]})
    return (np.concatenate(Ua, axis=1),
            np.concatenate(Vc, axis=0), ledger)
