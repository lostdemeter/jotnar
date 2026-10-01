"""Freeze mined edges as assoc store (offline, content scale-up).

Mined edges (mined_edges.jsonl, grokipedia minsup-2) + curated 55:
same combiner + bipolar machinery as freeze_edges (D=64, seeds 0/1).
Mined keys are SEPARATE rows (provenance kept: mined vs curated never
mix silently). Writes data/mined_keys.npz + manifest.
Usage: python3 scripts/freeze_mined.py
"""
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")
DIM = 64


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    zw = np.load(os.path.join(OUT, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    # extend lexicon with mined words (seeded, same D)
    mined = [json.loads(l) for l in open(os.path.join(OUT, "mined_edges.jsonl"))][1:]
    new_words = sorted({w for e in mined for f in ("subj", "pred", "obj")
                        for w in words_of(e[f])} - set(wpat))
    # extend lexicon with mined words: HASH-addressed patterns (seed 99 +
    # sha256(word) -> per-word deterministic, order-independent: banks
    # frozen at different scales share identical patterns per word)
    rng = np.random.default_rng(99)
    _ = rng  # seed namespace retained for provenance
    for w in sorted(new_words):
        h = int(hashlib.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
        wpat[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=DIM)
    rv = np.random.default_rng(101)
    keys, values, key_edge = [], [], []
    oov = 0
    for ei, e in enumerate(mined):
        order = words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"])
        pats = [(k, wpat[w]) for k, w in enumerate(order) if w in wpat]
        if not pats:
            oov += 1
            continue
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        keys.append(cue)
        values.append(rv.choice([-1.0, 1.0], size=DIM))
        key_edge.append(ei)
    keys, values = np.array(keys), np.array(values)
    np.savez(os.path.join(OUT, "mined_keys.npz"), keys=keys, values=values,
             key_edge=np.array(key_edge))
    json.dump({"n_edges": len(mined), "n_keys": len(keys), "oov": oov,
               "lex_ext": len(new_words),
               "keys_sha": hashlib.sha256(keys.tobytes()).hexdigest()[:16]},
              open(os.path.join(OUT, "mined_keys_manifest.json"), "w"), indent=2)
    print(f"mined keys={len(keys)}/{len(mined)} oov={oov} lex+{len(new_words)}")
    print(f"wrote {OUT}/mined_keys.npz + manifest")


if __name__ == "__main__":
    main()
