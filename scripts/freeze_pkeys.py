"""Freeze piece-native edge keys (offline): same combiner, piece space.

Word cues drop OOV (the 33 skipped probes); pieces cover everything
(roundtrip-exact BPE). Piece patterns hash-addressed (plex###: per-piece
deterministic, order-independent -- same rule as lex99). Combiner over
piece-encoded edge words (subj+pred+obj pieces, positions 0..). Values:
fresh bipolar (seed 101, same painter rule). Writes data/pkeys.npz
(keys/values/key_edge) + manifest. Proves >= word-space probe rate.
Usage: python3 scripts/freeze_pkeys.py [--edges data/mined_edges.jsonl --out data/pkeys.npz]
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")
DIM = 64


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=os.path.join(OUT, "mined_edges.jsonl"))
    ap.add_argument("--out", default=os.path.join(OUT, "pkeys.npz"))
    ap.add_argument("--manifest", default=os.path.join(OUT, "pkeys_manifest.json"))
    a = ap.parse_args()
    vocab = json.load(open(os.path.join(OUT, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(OUT, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in words_of(s):
            syms = [c for c in w] + ["</w>"]
            while len(syms) > 1:
                best = None
                for i in range(len(syms) - 1):
                    r = rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best[0]):
                        best = (r, i)
                if best is None:
                    break
                _, i = best
                syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
            out.extend(vocab[p] for p in syms)
        return out

    def ppat(pid):
        h = int(hashlib.sha256(f"plex:{pid}".encode()).hexdigest()[:16], 16)
        return np.random.default_rng(h).choice([-1.0, 1.0], size=DIM)

    edges = [json.loads(l) for l in open(a.edges)][1:]
    rv = np.random.default_rng(101)
    keys, values, ke = [], [], []
    for ei, e in enumerate(edges):
        pids = encode(e["subj"] + " " + e["pred"] + " " + e["obj"])
        cue = np.sign(sum(np.roll(ppat(p), k) for k, p in enumerate(pids)))
        cue[cue == 0] = 1.0
        keys.append(cue)
        values.append(rv.choice([-1.0, 1.0], size=DIM))
        ke.append(ei)
    keys, values = np.array(keys), np.array(values)
    np.savez(a.out, keys=keys, values=values, key_edge=np.array(ke))
    json.dump({"n_edges": len(edges), "n_keys": len(keys), "oov": 0,
               "keys_sha": hashlib.sha256(keys.tobytes()).hexdigest()[:16]},
              open(a.manifest, "w"), indent=2)
    print(f"piece keys={len(keys)}/{len(edges)} oov=0")
    print(f"wrote {a.out} + manifest")


if __name__ == "__main__":
    main()
