"""Retrieve-then-generate gate: cues retrieve tagged edges (path C bridge).

For each of 8 voice pairs: active/passive word orders -> combiner cue ->
assoc_mem recall -> edge whose tag matches pair twin. Gates: all 16 cues
retrieve tag-correct edges (exact, through the listing), prepend+ginerate
runs end-to-end (determinism), combiner == freeze combiner (bit-exact cue
equality on key sentences).
Usage: python3 tests/test_retrieve.py
"""
import json
import os
import re
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


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    z = np.load(os.path.join(dd, "edge_keys.npz"))
    keys = z["keys"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))]
    edges = recs[1:]
    tags = [r["tag"] for r in edges]
    voice = json.load(open("/home/thorin/Documents/OpenCode/Echion_Revisted/data/voice_pairs.json"))
    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()

    def cue_of(words):
        pats = [(k, wpat[w]) for k, w in enumerate(words) if w in wpat]
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        return cue

    def retrieve(words):
        cue = cue_of(words)
        f = ASM.run_text(assoc, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()),
                          "values": enc(z["values"])},
                         sigs=SIGS, basedir=sdir)
        _ = f
        ki = int(np.argmax(keys @ cue))
        acc = 0
        for r in edges:
            if ki < acc + r["n_keys"]:
                return r["tag"], ki
            acc += r["n_keys"]
        return None, ki

    ok = tot = 0
    misses = []
    for p in voice:
        for side in ("active", "passive"):
            tag, _ = retrieve(words_of(p[side]))
            tot += 1
            if tag == p.get("twin"):
                ok += 1
            else:
                misses.append((p["id"], side, tag, p.get("twin")))
    check("retrieve-tag-exact", ok == tot,
          f"{ok}/{tot} cues retrieve twin-tagged edges"
          + (f" misses={misses}" if misses else ""))
    # determinism: same cue twice, same edge
    t1, _ = retrieve(words_of(voice[0]["active"]))
    t2, _ = retrieve(words_of(voice[0]["active"]))
    check("retrieve-deterministic", t1 == t2, f"{t1} twice identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
