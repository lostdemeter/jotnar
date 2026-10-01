"""Loop on our creature (v1.6 gate 4): read/label/implant/verify bigram LM.

H1 (silence column): zero the 'bc' column (26 predicting contexts) --
every such context MUST stop predicting bc (exact claim: argmax flips,
deterministic pipeline, no statistics). H2 (implant): 5 unseen pairs set
above their row max -- listing MUST predict each (creation with guarantee).
Bands are exact (own model, fully comprehensible) -- stronger than dB.
Usage: python3 tests/test_lm_loop.py (fast: ~520 tiny listing runs)
"""
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


def main():
    import json
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    text = open(os.path.join(root, "programs", "bigram_lm.asm")).read()
    sdir = os.path.join(root, "programs")

    def triples_of(C):
        bs = np.zeros_like(C, np.int8)
        be = np.zeros_like(C, np.int32)
        bz = np.ones_like(C, np.uint8)
        for v in np.unique(C):
            if v == 0:
                continue
            ws, we, wz = S.encode(np.array([float(v)]))
            m = C == v
            bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
        return (np.ascontiguousarray(bs), np.ascontiguousarray(be),
                np.ascontiguousarray(bz))

    def predict(tok, bank):
        feeds = ASM.run_text(text, REGISTRY,
                             {"tok": np.array([tok], np.int64),
                              "bank": bank}, sigs=SIGS, basedir=sdir)
        return int(np.ascontiguousarray(feeds["OUT"]).reshape(-1)[0])

    bank0 = triples_of(counts)
    # H1: silence 'bc' column exhaustively (every row, via listing)
    bc = vocab["bc"]
    base = np.array([predict(i, bank0) for i in range(counts.shape[0])])
    ctx = [i for i in range(counts.shape[0]) if base[i] == bc]
    check("loop-silexists", len(ctx) > 5,
          f"{len(ctx)} contexts predict 'bc' (label exists)")
    C1 = counts.copy()
    C1[:, bc] = 0
    bank1 = triples_of(C1)
    flipped = [i for i in ctx if predict(i, bank1) != bc]
    check("loop-silence", len(flipped) == len(ctx),
          f"{len(flipped)}/{len(ctx)} flipped away from 'bc' (exact)")
    # H2: implant 5 unseen pairs above row max (content rows only)
    inv = {i: w for w, i in vocab.items()}
    pairs = []
    for i in range(counts.shape[0]):
        if counts[i].sum() == 0 or i == 0:
            continue
        for j in np.argsort(counts[i]):
            if counts[i, j] == 0:
                pairs.append((i, j))
                break
        if len(pairs) >= 5:
            break
    ok = 0
    for i, j in pairs:
        C2 = counts.copy()
        C2[i, j] = int(counts[i].max()) + 1
        if predict(i, triples_of(C2)) == j:
            ok += 1
    check("loop-implant", ok == len(pairs) == 5,
          f"{ok}/5 implanted pairs predicted (creation with guarantee; "
          f"e.g. {inv.get(pairs[0][0],'?')}->{inv.get(pairs[0][1],'?')})")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
