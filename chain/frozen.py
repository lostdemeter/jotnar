"""Content-addressed frozen triples (lossless consolidation).

Two redundancies, both exact:
  1. constant planes (all-equal arrays -- zero planes above all): stored
     ONCE, referenced by manifest. The 7B L0 MLP freeze wastes 136MB on
     three identical zero planes; this removes it with zero bit change.
  2. exact-duplicate arrays across artifacts (47 groups repo-wide, mostly
     shared bank metadata): same mechanism, same guarantee.

Format: the .npz holds one blob per unique sha (key `b_<sha16>`);
the manifest maps logical names (wupT_s, ...) to blob keys + shapes +
dtypes. load() reconstructs the logical dict. save() computes sharing
automatically. Nothing here is lossy: identical bytes in, identical
bytes out (gated).
"""
import hashlib
import json
import os

import numpy as np

__all__ = ["save_shared", "load_shared", "share_report"]


def _blob_key(a):
    return "b_" + hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def save_shared(path, logical, manifest_path=None, _warn_mb=1.0):
    """logical: {name: ndarray}. Writes deduped .npz + manifest dict.

    Returns (manifest, report). Constant planes (single unique value)
    are recorded inline as {"const": value} -- zero bytes on disk."""
    blobs, refs, saved = {}, {}, {"blobs": 0, "shared": 0, "const": 0}
    for name, arr in logical.items():
        a = np.ascontiguousarray(arr)
        u = np.unique(a)
        if len(u) == 1:
            refs[name] = {"const": u.reshape(-1)[0].item(),
                          "shape": list(a.shape), "dtype": str(a.dtype)}
            saved["const"] += a.nbytes
            continue
        key = _blob_key(a)
        if key not in blobs:
            blobs[key] = a
            saved["blobs"] += a.nbytes
        else:
            saved["shared"] += a.nbytes
        refs[name] = {"blob": key, "shape": list(a.shape),
                      "dtype": str(a.dtype)}
    np.savez(path, **blobs)
    manifest = {"refs": refs, "saved_bytes": saved,
                "note": "lossless content-addressed triples (chain/frozen.py)"}
    if manifest_path:
        json.dump(manifest, open(manifest_path, "w"), indent=1)
    return manifest, saved


def load_shared(path, manifest):
    """Reconstruct the logical {name: ndarray} dict. Bit-exact."""
    if isinstance(manifest, str):
        manifest = json.load(open(manifest))
    z = np.load(path, allow_pickle=False)
    out = {}
    for name, ref in manifest["refs"].items():
        if "const" in ref:
            out[name] = np.full(ref["shape"], ref["const"],
                                dtype=np.dtype(ref["dtype"]))
        else:
            b = np.ascontiguousarray(z[ref["blob"]])
            out[name] = b.reshape(ref["shape"]).astype(
                np.dtype(ref["dtype"]), copy=False)
    return out


def share_report(manifest):
    s = manifest["saved_bytes"]
    tot = s["blobs"] + s["shared"] + s["const"]
    dup = s["shared"] + s["const"]
    return (f"unique {s['blobs'] / 1e6:.1f}MB, shared-away "
            f"{dup / 1e6:.1f}MB ({100 * dup / max(tot, 1):.1f}%)")
