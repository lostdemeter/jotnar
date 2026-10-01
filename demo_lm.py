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
    topk, temp, seedn, nrep, cand = 0, 1.0, 0, 0, 1
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
        elif a == "--topk" and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
            i += 2
        elif a.startswith("--topk="):
            topk = int(a.split("=", 1)[1])
            i += 1
        elif a == "--temp" and i + 1 < len(sys.argv):
            temp = float(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        elif a == "--no-repeat" and i + 1 < len(sys.argv):
            nrep = int(sys.argv[i + 1])
            i += 2
        elif a == "--cand" and i + 1 < len(sys.argv):
            cand = int(sys.argv[i + 1])
            i += 2
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

    def row_counts(tok):
        return counts[tok].astype(np.float64)

    def sample_next(tok, hist, rng):
        # host-boundary decoding (design): top-k + temperature + no-repeat
        # blocking over the listing's counts. Model/listing untouched.
        w = row_counts(tok).copy()
        if nrep > 0 and len(hist) >= nrep - 1:
            seen = {tuple(hist[k:k + nrep])
                    for k in range(len(hist) - nrep + 1)}
            prefix = tuple(hist[-(nrep - 1):]) if nrep > 1 else ()
            for j in range(len(w)):
                if prefix + (j,) in seen:
                    w[j] = 0
        if topk > 0:
            keep = np.argsort(-w)[:topk]
            mask = np.zeros_like(w)
            mask[keep] = w[keep]
            w = mask
        w = np.power(np.maximum(w, 0), 1.0 / temp)
        tot = w.sum()
        if tot <= 0:
            return step(tok)  # fall back to listing argmax
        w = w / tot
        return int(rng.choice(len(w), p=w))

    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    use_sample = topk > 0 or nrep > 0 or cand > 1
    if not use_sample:
        out = list(ids)
        # Stops at UNK regardless (leaving the model's world ends the walk).
        for _ in range(n):
            nxt = step(out[-1])
            if nxt == 0:
                break
            out.append(nxt)
    else:
        # Echion pattern: generate CAND candidates, keep max unique-ratio
        # (fitness selection over sampled continuations; seeded = replayable).
        best, best_u = None, -1.0
        for c in range(max(cand, 1)):
            rng = np.random.default_rng(seedn * 100003 + c)
            out = list(ids)
            for _ in range(n):
                if out[-1] == 0:
                    break
                if topk > 0 or nrep > 0:
                    nxt = sample_next(out[-1], out, rng)
                else:
                    nxt = step(out[-1])
                if nxt == 0:
                    break
                out.append(nxt)
            u = len(set(out)) / max(len(out), 1)
            if u > best_u:
                best, best_u = list(out), u
        out = best
        print(f"(cand={max(cand,1)} unique-ratio={best_u:.2f})")
    words = [inv.get(i, "<unk>") for i in out]
    print("seed:", seed)
    print("out :", " ".join(words))
    print(f"({len(out)} tokens, V={len(vocab)}, "
          f"{'sampled' if use_sample else 'greedy argmax'}, no trained "
          f"weights; stops at UNK)")


if __name__ == "__main__":
    main()
