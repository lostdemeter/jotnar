"""Demo: our LLM generating text (v1.6 construction, working).

Seeds -> greedy argmax chain through programs/bigram_lm.asm (frozen
counts, no trained weights) -> printed sentences. UNK-aware detokenizer
(simple join; the creature speaks lowercase grokipedia). Run it:
python3 demo_lm.py [seed words...] [--n 30]
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


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    text = open(os.path.join(root, "programs", "bigram_lm.asm")).read()
    sdir = os.path.join(root, "programs")
    words, n, i, no_unk = [], 30, 1, True
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a.startswith("--n="):
            n = int(a.split("=", 1)[1])
            i += 1
        elif a == "--allow-unk":
            no_unk = False
            i += 1
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the"
    # UNK is absorbing (tail mass concentrates there): masked in the demo
    # BANK (column 0 -> zero triples) by default -- a decoding rule at the
    # boundary per design (counts file untouched; --allow-unk restores raw
    # argmax). The listing itself never changes.
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    if no_unk:
        counts = counts.copy()
        counts[:, 0] = 0
    bs = np.zeros_like(counts, np.int8)
    be = np.zeros_like(counts, np.int32)
    bz = np.ones_like(counts, np.uint8)
    for v in np.unique(counts):
        if v == 0:
            continue
        ws, we, wz = S.encode(np.array([float(v)]))
        m = counts == v
        bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
    bank = (np.ascontiguousarray(bs), np.ascontiguousarray(be),
            np.ascontiguousarray(bz))

    def step(tok):
        feeds = ASM.run_text(text, REGISTRY,
                             {"tok": np.array([tok], np.int64),
                              "bank": bank}, sigs=SIGS, basedir=sdir)
        return int(np.ascontiguousarray(feeds["OUT"]).reshape(-1)[0])

    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    out = list(ids)
    # Stops at UNK regardless (leaving the model's world ends the walk).
    for _ in range(n):
        nxt = step(out[-1])
        if nxt == 0:
            break
        out.append(nxt)
    words = [inv.get(i, "<unk>") for i in out]
    print("seed:", seed)
    print("out :", " ".join(words))
    print(f"({len(out)} tokens, V={len(vocab)}, greedy argmax, no trained "
          f"weights; stops at UNK)")


if __name__ == "__main__":
    main()
