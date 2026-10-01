"""Demo: bankhn2 transformer generating (current best stack).

IvoQ weights + dual storebank MLPs, causal, sliding window 8. Greedy by
default; --topk/--seed for host-boundary sampling. Stated: bigram-level
coil with transformer modulation (top1 0.35 class), not fluency -- read
for content-following and order, not prose.
Run: python3 demo_bankhn.py [seed words...] [--n 20] [--topk 12 --seed 7]
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

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
WIN = 8


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
         "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
         "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def logits_of(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          **P}, sigs=SIGS, basedir=sdir)
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    words, n, i, topk, seedn, no_unk = [], 20, 1, 0, 7, True
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--topk" and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        elif a == "--allow-unk":
            no_unk = False
            i += 1
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    rng = np.random.default_rng(seedn)
    out = list(ids)
    for _ in range(n):
        lg = logits_of(out).copy()
        if no_unk:
            lg[0] = -1e9  # UNK-absorbing boundary rule (demo_lm precedent)
        if topk > 0:
            keep = np.argsort(-lg)[:topk]
            w = np.zeros_like(lg)
            w[keep] = np.exp(lg[keep] - lg[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        else:
            out.append(int(np.argmax(lg)))
    print("seed:", seed)
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))
    print(f"({len(out)} tokens, bankhn2 stack)")


if __name__ == "__main__":
    main()
