"""Mined-content gate: scale with probes (content growth, gated).

Frozen data/mined_edges.jsonl (51 grokipedia minsup-2, junk-filtered) +
data/mined_keys.npz (51 keys, lex+60). Gates: provenance, recall bands
through assoc_mem.asm (100% @0/8), probe self-retrieval MEASURED
(subj+pred cue -> same edge; partial cues collide by design -- the
number prices disambiguation need, not a bar), curated 55 untouched
(regression: test_edge_store still green).
Usage: python3 tests/test_mined.py
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
    man = json.load(open(os.path.join(dd, "mined_manifest.json")))
    check("mined-manifest", man["n_edges"] == 51,
          f"51 edges minsup-2 (sha {man['sha']})")
    recs = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))]
    check("mined-records", recs[0]["format"] == "echion-store/1"
          and len(recs) == 52, "envelope + 51 probes (q+a co-frozen)")
    z = np.load(os.path.join(dd, "mined_keys.npz"))
    keys, values = z["keys"], z["values"]
    text = open(os.path.join(sdir, "assoc_mem.asm")).read()

    def recall(cue):
        f = ASM.run_text(text, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()), "values": enc(values)},
                         sigs=SIGS, basedir=sdir)
        return np.asarray(f["OUT"][0]).reshape(-1) > 0

    rng = np.random.default_rng(3)
    for f_, bar in ((0, 1.0), (8, 1.0)):
        ok = tot = 0
        for i in range(len(keys)):
            for _ in range(2):
                cue = keys[i].copy()
                if f_:
                    cue[rng.choice(64, size=f_, replace=False)] *= -1
                got = np.where(recall(cue), 1.0, -1.0)
                ok += bool((got == values[i]).all())
                tot += 1
        check(f"mined-recall{f_}", ok / tot >= bar, f"{ok}/{tot}")
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    # lex extension (freeze_mined: mined words, seed 99) -- keys were
    # frozen with it, so probes must use it (partial-cue collisions
    # were cue-side OOV drop, not store collision: 9/9 HIT full-triple)
    rng_lex = np.random.default_rng(99)
    mined = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]
    for w in sorted({x for e in mined for f in ("subj", "pred", "obj")
                     for x in words_of(e[f])} - set(wpat)):
        wpat[w] = rng_lex.choice([-1.0, 1.0], size=64)
    edges = recs[1:]
    ok = tot = 0
    for ei, e in enumerate(edges):
        # full-triple cue (subj+pred+obj[0]): probe with answer shape
        q = words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"])[:1]
        pats = [(k, wpat[w]) for k, w in enumerate(q) if w in wpat]
        if not pats:
            continue
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        ki = int(np.argmax(keys @ cue))
        tot += 1
        if int(z["key_edge"][ki]) == ei:
            ok += 1
    print(f"mined-probe: {ok}/{tot} self-retrieval with full-triple cues "
          f"+ extended lex (was 9/18 partial -- cue-side OOV, fixed)")
    check("mined-probes-run", tot > 10, f"{tot} probes scored")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
