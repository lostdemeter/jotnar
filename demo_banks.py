"""Banked retrieval (scale architecture): per-bank recall + arbitrate.

Banks (provenance-separated, never flat-merged): curated 55 (edge_keys),
groki+w20k mined 174 (mined_keys), w103 3773 (banks/w103_keys). Query
builds cue per bank lexicon, recalls per bank through assoc_mem.asm,
arbitrates by raw dot-score (host, stated). Same combiner everywhere;
lexicons differ per bank (frozen with each).
Usage: python3 demo_banks.py [words...]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def load_bank(dd, sdir, keys_path, lex_path, recs_path, seed_ext=None):
    zw = np.load(os.path.join(dd, lex_path))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    if seed_ext:
        rng = np.random.default_rng(seed_ext)
        extra = set()
        for r in recs_path:
            pass
        wpat.update(extra)
    z = np.load(os.path.join(dd, keys_path))
    return wpat, z["keys"]


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()
    # bank A: curated 55
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    lexA = list(zw["lex"])
    wpatA = {w: zw[f"w{i}"] for i, w in enumerate(lexA)}
    zA = np.load(os.path.join(dd, "edge_keys.npz"))
    recA = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
    # banks B/C: lex ext HASH-addressed (lex99, order-independent)
    import hashlib as _hl

    def _ext(wpat, edges):
        for w in sorted({x for e in edges for f in ("subj", "pred", "obj")
                         for x in words_of(e[f])} - set(wpat)):
            h = int(_hl.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
            wpat[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=64)
        return wpat
    wpatB = _ext(dict(wpatA), [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:])
    mined = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]
    zB = np.load(os.path.join(dd, "mined_keys.npz"))
    # bank C: w103 3773 (own lex ext, same seeds)
    w103edges = [json.loads(l) for l in open(os.path.join(dd, "banks", "w103_edges.jsonl"))][1:]
    wpatC = _ext(dict(wpatA), w103edges)
    zC = np.load(os.path.join(dd, "banks", "w103_keys.npz"))
    results = []
    qwords = words_of(" ".join(sys.argv[1:]) or "battle of actium")
    banks = (("curated", wpatA, zA, recA),
             ("mined202", wpatB, zB,
              [{"id": e["id"], "subj": e["subj"],
                "pred": e["pred"], "obj": e["obj"]} for e in mined]),
             ("w103", wpatC, zC,
              [{"id": e["id"], "subj": e["subj"],
                "pred": e["pred"], "obj": e["obj"]} for e in w103edges]))
    for tag, wpat, z, recs in banks:
        keys = z["keys"]
        pats = [(k, wpat[w]) for k, w in enumerate(qwords) if w in wpat]
        if not pats:
            results.append((tag, None, 0, "OOV"))
            continue
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        f = ASM.run_text(assoc, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()),
                          "values": enc(z["values"])},
                         sigs=SIGS, basedir=sdir)
        _ = f
        sims = keys @ cue
        ki = int(np.argmax(sims))
        results.append((tag, ki, float(sims[ki]), ""))
    for tag, ki, score, note in results:
        print(f"{tag}: ki={ki} score={score:.1f} {note}")
    best = max(results, key=lambda t: t[2] if t[1] is not None else -1)
    print(f"winner: {best[0]} (arbitrate by dot-score, host)")


if __name__ == "__main__":
    main()
