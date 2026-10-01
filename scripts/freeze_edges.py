"""Freeze relation edges as assoc store (offline, path C first step).

Echion edge_store (55 edges) + voice twins (8 pairs, twin->edge tags):
word patterns (seeded bipolar D=64, edge lexicon) -> order-sensitive cue
combiner sign(sum roll(pat,pos)) -> keys; random values (seed 1).
Twin edges get TWO keys (canonical + twin word order) -> same value:
two execution orders, one content -- as data. Writes data/edge_keys.npz
+ edge_wordpats + edge_records.jsonl (echion-store/1 envelope, own
reader) + edge_manifest.json. Freezing offline; listing consumes frozen.
Usage: python3 scripts/freeze_edges.py [--seed 0]
"""
import hashlib
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ECH = "/home/thorin/Documents/OpenCode/Echion_Revisted/data"
OUT = os.path.join(ROOT, "data")
DIM = 64


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    edges = json.load(open(os.path.join(ECH, "edge_store.json")))["edges"]
    voice = json.load(open(os.path.join(ECH, "voice_pairs.json")))
    twin_order = {}
    for p in voice:
        tag = p.get("twin")
        if not tag:
            continue
        twin_order[tag] = (words_of(p["active"]), words_of(p["passive"]))
    lex = sorted({w for e in edges
                  for f in ("subj", "pred", "obj") for w in words_of(e[f])})
    rng = np.random.default_rng(a.seed)
    wpat = {w: rng.choice([-1.0, 1.0], size=DIM) for w in lex}
    rv = np.random.default_rng(a.seed + 1)
    keys, values, records = [], [], []
    for ei, e in enumerate(edges):
        v = rv.choice([-1.0, 1.0], size=DIM)
        canon = words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"])
        orders = [canon]
        tw = twin_order.get(e["tag"])
        if tw:
            for t in tw:
                if t != canon:
                    orders.append(t)
        for oi, order in enumerate(orders):
            cue = np.sign(sum(np.roll(wpat[w], k)
                              for k, w in enumerate(order) if w in wpat))
            cue[cue == 0] = 1.0
            keys.append(cue)
            values.append(v)
        records.append({"type": "edge", "id": e["id"], "tag": e["tag"],
                        "cat": e["cat"], "subj": e["subj"], "pred": e["pred"],
                        "obj": e["obj"], "canon": canon,
                        "n_keys": len(orders),
                        "digest": hashlib.sha256(
                            json.dumps(canon).encode()).hexdigest()[:16]})
    keys, values = np.array(keys), np.array(values)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "edge_keys.npz"), keys=keys, values=values)
    np.savez(os.path.join(OUT, "edge_wordpats.npz"),
             **{f"w{i}": wpat[w] for i, w in enumerate(lex)},
             lex=np.array(lex))
    with open(os.path.join(OUT, "edge_records.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "manifest", "format": "echion-store/1",
                             "n_records": len(records)}) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    man = {"seed": a.seed, "dim": DIM, "n_edges": len(edges),
           "n_keys": len(keys), "n_lex": len(lex),
           "twin_edges": sum(1 for r in records if r["n_keys"] > 1),
           "keys_sha": hashlib.sha256(keys.tobytes()).hexdigest()[:16]}
    json.dump(man, open(os.path.join(OUT, "edge_manifest.json"), "w"), indent=2)
    print(f"edges={len(edges)} keys={len(keys)} lex={len(lex)} "
          f"twins={man['twin_edges']}")
    print(f"wrote {OUT}/edge_keys.npz + wordpats + records.jsonl + manifest")


if __name__ == "__main__":
    main()
