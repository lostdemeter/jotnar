"""frozen.py gate: lossless consolidation roundtrips bit-exact.

Covers: const planes (zeros + nonzero consts), exact-duplicate arrays
across logical names, mixed dtypes/shapes, manifest refs resolve.
Usage: python3 tests/test_frozen.py (fast, /tmp only).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from chain.frozen import load_shared, save_shared, share_report

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(0)
    work = "/tmp/frozen_gate"
    os.makedirs(work, exist_ok=True)
    zplane = np.zeros((64, 128), np.uint8)
    logical = {
        "a_s": rng.integers(-5, 6, size=(32, 64)).astype(np.int8),
        "a_e": rng.integers(32000, 33500, size=(32, 64)).astype(np.int32),
        "a_s2": None,  # exact dupe of a_s (filled below)
        "a_z": zplane.copy(),
        "b_z": zplane.copy(),  # exact dupe of a_z
        "c_z": zplane.copy(),  # x3
        "w": np.full((16,), 7, np.int64),  # nonzero const
        "v": rng.normal(size=(8, 8)),
    }
    logical["a_s2"] = logical["a_s"].copy()
    man, saved = save_shared(os.path.join(work, "t.npz"), logical,
                             os.path.join(work, "t_manifest.json"))
    print("share:", share_report(man))
    check("frozen-saves-dedupe", saved["shared"] > 0 and saved["const"] > 0,
          str(saved))
    back = load_shared(os.path.join(work, "t.npz"),
                       os.path.join(work, "t_manifest.json"))
    ok = all(bool((np.ascontiguousarray(back[k])
                   == np.ascontiguousarray(v)).all())
             and back[k].dtype == v.dtype and back[k].shape == v.shape
             for k, v in logical.items())
    check("frozen-roundtrip-exact", ok, f"{len(logical)} arrays bit-identical")
    # on-disk footprint smaller than logical bytes
    disk = os.path.getsize(os.path.join(work, "t.npz"))
    logic = sum(v.nbytes for v in logical.values())
    check("frozen-smaller", disk < logic, f"{disk} < {logic}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
