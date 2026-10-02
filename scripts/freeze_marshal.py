"""Freeze marshalled ideas as assoc store (offline, fill-up program).

Ideas come from CORPUS-mined edges (attested by support/witness --
never from bare teacher text: harvest verdict). Optional teacher
paraphrase orders pass an exact-word filter (all triple content
words present) or are dropped, loudly (turn 1: 0/4 faithful --
canon order only). Keys use the freeze_mined combiner + lex99
extension (same machinery, separate rows: provenance kept, marshal
bank never flat-merges into mined/curated). Appends
data/marshal_catalog.jsonl (one entry per idea: geometry, support,
orders, key shas) and writes data/marshal_keys.npz +
marshal_records.jsonl. Deterministic (frozen bytes on rerun).
Usage: python3 scripts/freeze_marshal.py m0054 [m0060 ...]
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--new", nargs=3, metavar=("SUBJ", "PRED", "OBJ"),
                    help="sub-minsup novel triple (w103 attestation enforced)")
    ap.add_argument("--unique", nargs=3, metavar=("SUBJ", "PRED", "OBJ"),
                    help="groki-unique triple (absent everywhere; groki-witness recorded, thin evidence class)")
    a = ap.parse_args()
    assert a.ids or a.new or a.unique, "usage: freeze_marshal.py m0054 [...] [--new s p o] [--unique s p o]"
    zw = np.load(os.path.join(OUT, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    mined = [json.loads(l) for l in open(os.path.join(OUT, "mined_edges.jsonl"))][1:]
    byid = {e["id"]: e for e in mined}
    # novelty: fill-up adds, never duplicates consumer-tier banks
    # (curated/mined202/marshal). w103m5 is SOURCE mass (10k, noisy),
    # not consumer tier: it ATTESTS sub-minsup ideas, and cross-tier
    # duplication resolves by tier policy (marshal first = quality
    # wins ties, stated). Turn 1 m0054 class: mined202-dupe, kept.
    banked = set()
    mm = [json.loads(l) for l in open(os.path.join(OUT, "mined_edges.jsonl"))][1:]
    banked.update((e["subj"], e["pred"], e["obj"]) for e in mm)
    w3sup = {}
    try:
        zw3 = [json.loads(l) for l in open(os.path.join(OUT, "banks", "w103m5_edges.jsonl"))][1:]
        for e in zw3:
            w3sup[(e["subj"], e["pred"], e["obj"])] = e["support"]
    except Exception:
        pass
    try:
        recs0 = [json.loads(l) for l in open(os.path.join(OUT, "edge_records.jsonl"))][1:]
        banked.update((r["subj"], r["pred"], r["obj"]) for r in recs0)
    except Exception:
        pass
    # existing marshal bank (append-only across turns; reruns dedupe by id)
    old_keys, old_vals, old_recs, old_ids = [], [], [], set()
    try:
        z0 = np.load(os.path.join(OUT, "marshal_keys.npz"))
        old_keys, old_vals = [np.array(z0["keys"])], [np.array(z0["values"])]
        for l in open(os.path.join(OUT, "marshal_records.jsonl")):
            pass
        rr = [json.loads(l) for l in open(os.path.join(OUT, "marshal_records.jsonl"))][1:]
        old_recs = rr
        old_ids = {r["id"] for r in rr}
        banked.update((r["subj"], r["pred"], r["obj"]) for r in rr)
    except Exception:
        pass
    edges = []
    for i in a.ids:
        assert i in byid, f"unknown edge {i} (must be corpus-mined)"
        e = byid[i]
        if (e["subj"], e["pred"], e["obj"]) in banked:
            print(f"NOTE {i} duplicates a banked triple (turn-1 class, recorded, frozen anyway)")
        edges.append(e)
    cat_triples = set()
    try:
        for l in open(os.path.join(OUT, "marshal_catalog.jsonl")):
            cat_triples.add(tuple(json.loads(l)["triple"]))
    except Exception:
        pass
    if a.new:
        s, p, o = (x.lower() for x in a.new)
        mid = "n" + hashlib.sha256(json.dumps([s, p, o]).encode()).hexdigest()[:8]
        if (s, p, o) in cat_triples:
            print(f"NOTE triple {(s, p, o)} already marshalled (rerun is not a new idea, skipped)")
        else:
            assert (s, p, o) not in banked, f"triple {(s, p, o)} already banked (novel-only)"
            assert (s, p, o) in w3sup, f"triple {(s, p, o)} NOT attested in w103m5 (independent mining required)"
            edges.append({"id": mid, "subj": s, "pred": p, "obj": o,
                          "support": 1, "witness": f"groki-subminsup+w103m5x{w3sup[(s, p, o)]}",
                          "evidence": "attested"})
    if a.unique:
        import glob as _glob
        import html as _html
        from html.parser import HTMLParser as _HP

        class _TT(_HP):
            def __init__(self):
                super().__init__()
                self.p = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                self.skip = tag in ("script", "style", "nav", "header", "footer",
                                    "aside")

            def handle_endtag(self, tag):
                self.skip = False

            def handle_data(self, d):
                if not self.skip:
                    self.p.append(d)

        gsents = []
        for _f in sorted(_glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
            _t = _TT()
            _t.feed(open(_f, encoding="utf-8", errors="replace").read())
            _txt = _html.unescape(" ".join(_t.p))
            gsents += [_s.strip() for _s in re.split(r"(?<=[.!?])\s+", _txt)
                       if len(_s.strip().split()) >= 4]
        s, p, o = (x.lower() for x in a.unique)
        mid = "u" + hashlib.sha256(json.dumps([s, p, o]).encode()).hexdigest()[:8]
        if (s, p, o) in cat_triples:
            print(f"NOTE triple {(s, p, o)} already marshalled (rerun is not a new idea, skipped)")
        else:
            assert (s, p, o) not in banked, f"triple {(s, p, o)} already banked (novel-only)"
            assert (s, p, o) not in w3sup, f"triple {(s, p, o)} in w103m5 (use --new attested path)"
            wit = [i for i, _s in enumerate(gsents)
                   if s in words_of(_s) and p in words_of(_s) and o in words_of(_s)]
            assert wit, f"triple {(s, p, o)} has no groki witness sentence (thin evidence minimum)"
            edges.append({"id": mid, "subj": s, "pred": p, "obj": o,
                          "support": 1, "witness": f"groki-sent{wit[0]}-only",
                          "evidence": "unique-thin"})
    edges = [e for e in edges if e["id"] not in old_ids]
    for w in sorted({x for e in edges for f in ("subj", "pred", "obj")
                     for x in words_of(e[f])} - set(wpat)):
        h = int(hashlib.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
        wpat[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=DIM)
    keys, values, recs = [], [], []
    for e in edges:
        order = words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"])
        pats = [(k, wpat[w]) for k, w in enumerate(order) if w in wpat]
        assert pats, f"empty cue for {e['id']}"
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        keys.append(cue)
        # per-idea value stream (id-seeded: distinct across ideas, stable across reruns)
        iseed = int(hashlib.sha256(f"marshal-val:{e['id']}".encode()).hexdigest()[:16], 16)
        values.append(np.random.default_rng(iseed).choice([-1.0, 1.0], size=DIM))
        recs.append({"type": "edge", "id": e["id"], "subj": e["subj"],
                     "pred": e["pred"], "obj": e["obj"],
                     "support": e["support"], "witness": e["witness"],
                     "evidence": e.get("evidence", "banked"),
                     "orders": ["canon"],
                     "digest": hashlib.sha256(
                         json.dumps(order).encode()).hexdigest()[:16]})
    keys, values = ((np.array(keys), np.array(values)) if keys
                      else (np.zeros((0, DIM)), np.zeros((0, DIM))))
    if len(old_keys):
        if len(keys):
            keys = np.concatenate([old_keys[0], keys], axis=0)
            values = np.concatenate([old_vals[0], values], axis=0)
        else:
            keys, values = old_keys[0], old_vals[0]
        recs = old_recs + recs
    np.savez(os.path.join(OUT, "marshal_keys.npz"), keys=keys, values=values)
    with open(os.path.join(OUT, "marshal_records.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "manifest", "format": "echion-store/1",
                             "n_records": len(recs)}) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "marshal_catalog.jsonl"), "a") as fh:
        n_old = len(old_recs)
        for r, k in zip(recs[n_old:], keys[n_old:] if len(keys) else []):
            fh.write(json.dumps(
                {"id": r["id"], "triple": [r["subj"], r["pred"], r["obj"]],
                 "support": r["support"], "orders": r["orders"],
                 "evidence": r.get("evidence", "banked"),
                 "paraphrase": "canon only unless turn log notes otherwise",
                 "key_sha": hashlib.sha256(
                     np.ascontiguousarray(k).tobytes()).hexdigest()[:16]}) + "\n")
    print(f"marshalled {len(recs) - len(old_recs)} new ideas "
          f"(bank now {len(recs)}): {[r['id'] for r in recs[len(old_recs):]]}")
    print(f"wrote {OUT}/marshal_keys.npz + records + catalog append")


if __name__ == "__main__":
    main()
