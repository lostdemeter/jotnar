"""Edge-store gate: relations frozen, recall exact (path C first step).

Frozen data/edge_keys.npz (55 Echion edges -> 68 keys incl. 8 twin
second-orders, D=64) + records.jsonl + manifest. Gates: provenance,
recall bands via programs/assoc_mem.asm (demo2 bars: 100% @0/8,
>=80% @16), twin keys retrieve the SAME value (two orders, one
content -- as data), determinism bit-exact.
Usage: python3 tests/test_edge_store.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    man = json.load(open(os.path.join(root, "data", "edge_manifest.json")))
    check("edge-manifest", man["n_edges"] == 55 and man["n_keys"] == 68
          and man["twin_edges"] == 8,
          f"55 edges, 68 keys, 8 twins (sha {man['keys_sha']})")
    recs = [json.loads(l) for l in
            open(os.path.join(root, "data", "edge_records.jsonl"))]
    check("edge-records", recs[0]["format"] == "echion-store/1"
          and recs[0]["n_records"] == 55 and len(recs) == 56,
          "envelope + 55 records (own reader, no import)")
    z = np.load(os.path.join(root, "data", "edge_keys.npz"))
    keys, values = z["keys"], z["values"]
    text = open(os.path.join(sdir, "assoc_mem.asm")).read()

    def recall(cue):
        f = ASM.run_text(text, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()), "values": enc(values)},
                         sigs=SIGS, basedir=sdir)
        return np.asarray(f["OUT"][0]).reshape(-1) > 0

    def rate(nflip, trials=4, seed=1000 + 0):
        rng = np.random.default_rng(seed + nflip)
        ok = tot = 0
        for i in range(len(keys)):
            for _ in range(trials):
                cue = keys[i].copy()
                if nflip:
                    cue[rng.choice(64, size=nflip, replace=False)] *= -1
                got = np.where(recall(cue), 1.0, -1.0)
                ok += bool((got == values[i]).all())
                tot += 1
        return ok / tot

    r0 = rate(0)
    check("edge-exact", r0 == 1.0, f"{r0:.2f} @0 flips (must be exact)")
    r8 = rate(8)
    check("edge-robust8", r8 == 1.0, f"{r8:.2f} @<=8 flips")
    r16 = rate(16)
    check("edge-robust16", r16 >= 0.80, f"{r16:.2f} @<=16 flips")
    # twin keys (consecutive pairs for twin edges: canonical + twin order)
    # retrieve the SAME value pattern through the listing
    rec_idx, ki, same = 0, 0, True
    for r in recs[1:]:
        n = r["n_keys"]
        if n > 1:
            a = recall(keys[ki])
            b = recall(keys[ki + 1])
            same = same and bool((a == b).all())
        ki += n
        rec_idx += 1
    check("edge-twin-same-value", same,
          "all 8 twin order-pairs retrieve identical value (order->content)")
    a = recall(keys[0])
    b = recall(keys[0])
    check("edge-deterministic", bool((a == b).all()),
          "same cue twice identical (machinery adds nothing)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
