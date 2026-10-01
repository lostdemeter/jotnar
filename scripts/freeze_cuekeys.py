"""Freeze cue projection + hidden-space edge keys (offline, cue-from-hidden).

P (16->64, seed 7, ±0.25) bridges hidden space to cue space, frozen with
manifest. Edge keys rebuilt in the SAME space: order-sensitive combiner
over SVD-emb rows (emb rows live hidden-adjacent) @ P, signed. Twin
second-orders included (same rule as freeze_edges). Writes
data/cueproj.npz + data/hkeys.npz + manifests. Offline; listings consume.
Usage: python3 scripts/freeze_cuekeys.py
"""
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ECH = "/home/thorin/Documents/OpenCode/Echion_Revisted/data"
OUT = os.path.join(ROOT, "data")


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    rng = np.random.default_rng(7)
    P = rng.choice([-1.0, 1.0], size=(16, 64)) / 4.0
    np.savez(os.path.join(OUT, "cueproj.npz"), P=P)
    json.dump({"seed": 7, "shape": [16, 64], "scale": 0.25,
               "sha": hashlib.sha256(P.tobytes()).hexdigest()[:16]},
              open(os.path.join(OUT, "cueproj_manifest.json"), "w"), indent=2)
    svd = np.load(os.path.join(OUT, "lm_svd.npz"))
    E = svd["emb"]
    vocab = json.load(open(os.path.join(OUT, "lm_vocab.json")))
    edges = json.load(open(os.path.join(ECH, "edge_store.json")))["edges"]
    voice = json.load(open(os.path.join(ECH, "voice_pairs.json")))
    twin_order = {}
    for p in voice:
        if p.get("twin"):
            twin_order[p["twin"]] = (words_of(p["active"]), words_of(p["passive"]))

    def erow(w):
        i = vocab.get(w)
        return E[i] if i is not None else None

    def combine(order):
        terms = [np.roll(erow(w), k) for k, w in enumerate(order)
                 if erow(w) is not None]
        if not terms:
            return None
        return np.sign(sum(terms))

    keys, key_edge = [], []
    for ei, e in enumerate(edges):
        canon = words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"])
        orders = [canon]
        tw = twin_order.get(e["tag"])
        if tw:
            for t in tw:
                if t != canon:
                    orders.append(t)
        for order in orders:
            c = combine(order)
            if c is None:
                continue
            keys.append(np.sign(c @ P))
            key_edge.append(ei)
    keys = np.array(keys)
    np.savez(os.path.join(OUT, "hkeys.npz"), keys=keys,
             key_edge=np.array(key_edge))
    json.dump({"seed": 7, "n_keys": len(keys), "n_edges": len(edges),
               "keys_sha": hashlib.sha256(keys.tobytes()).hexdigest()[:16]},
              open(os.path.join(OUT, "hkeys_manifest.json"), "w"), indent=2)
    print(f"hkeys={len(keys)} edges={len(edges)}")
    print(f"wrote {OUT}/cueproj.npz + hkeys.npz + manifests")


if __name__ == "__main__":
    main()
