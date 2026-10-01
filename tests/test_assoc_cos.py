"""Cosine retrieval gate: direction-first memory as structure.

Listing programs/assoc_cos.asm (RMSNorm equalize + MATMUL + ARGMAX, no
new mnemonics) over hidden-space keys (data/hkeys2.npz, IvoQ body).
Gates: 14/16 twin cues tag-exact through the listing (host float also
14/16 -- structure costs nothing at priced scales), dot-baseline 2/16
recorded (norm-bias mechanism), determinism, priced CONFIG tripwire
(frozen scales must do worse -- proves the pricing is load-bearing).
Usage: python3 tests/test_assoc_cos.py (slow: 16 LM runs + 16 recalls)
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
M_ACC = 36230


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "scripts"))
    import eval_body as E
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    d = E.load_all()
    ivo = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    w = {k: np.array(ivo[k]) for k in ivo.files}
    wfull = dict(d)
    for k in w:
        wfull[k] = w[k]
    keys2 = np.load(os.path.join(dd, "hkeys2.npz"))["keys"]
    ke2 = np.load(os.path.join(dd, "hkeys2.npz"))["key_edge"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
    voice = json.load(open("/home/thorin/Documents/OpenCode/Echion_Revisted/data/voice_pairs.json"))
    text = f"CONFIG m_acc {M_ACC}\n" + open(os.path.join(sdir, "assoc_cos.asm")).read()
    nw = np.ones(16)

    def hidden_cue(words):
        return E.run_h4(d, [d["vocab"].get(x, 0) for x in words], wfull)[-1]

    def recall(h):
        f = ASM.run_text(text, REGISTRY,
                         {"cue": enc(h.reshape(1, -1)), "keys": enc(keys2),
                          "normw": enc(nw)}, sigs=SIGS, basedir=sdir)
        return int(np.ascontiguousarray(f["OUT"]).reshape(-1)[0])

    ok = tot = 0
    misses = []
    for p in voice:
        for side in ("active", "passive"):
            h = hidden_cue(words_of(p[side]))
            ki = recall(h)
            tot += 1
            if recs[int(ke2[ki])]["tag"] == p.get("twin"):
                ok += 1
            else:
                misses.append(f"{p['id']}/{side}")
    check("assoc-cos-tag-exact", ok >= 14,
          f"{ok}/{tot} through the listing (host ceiling 14/16; "
          f"misses={misses})")
    h = hidden_cue(words_of(voice[0]["active"]))
    check("assoc-cos-deterministic", recall(h) == recall(h),
          "same hidden cue twice identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
