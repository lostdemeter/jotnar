"""Demo: piece-loop end-to-end (BPE through hidden states, no UNK class).

Seed words -> BPE piece-encode -> bankhn2 stack on piece ends (SVD
rank-16, bankpiece512, S=16 window) -> piece AR -> word-decode.
Stated: piece top1 ~0.06 class (2038 choices); words come out whole
(no UNK -- every id decodes). Read for speakability, not accuracy.
Run: python3 demo_piece.py [seed words...] [--n 20] [--topk 12 --seed 7]
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

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in re.findall(r"[a-z0-9']+", s.lower()):
            syms = [c for c in w] + ["</w>"]
            while len(syms) > 1:
                best = None
                for i in range(len(syms) - 1):
                    r = rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best[0]):
                        best = (r, i)
                if best is None:
                    break
                _, i = best
                syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
            out.extend(vocab[p] for p in syms)
        return out

    def decode(ids):
        t = "".join(inv[i] for i in ids).replace("</w>", " ")
        return " ".join(t.split())

    Ep = np.load(os.path.join(dd, "lm_piece.npz"))
    b = np.load(os.path.join(dd, "bankpiece64.npz"))
    d = np.load(os.path.join(dd, "lm_piece_fit.npz"))
    B = {k: np.array(d[k]) for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown", "rms1", "rms2"]}
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(Ep["emb"]), "wq": enc(B["wq"]), "wk": enc(B["wk"]),
         "wv": enc(B["wv"]), "wo": enc(B["wo"]),
         "wup": enc(B["wup"]), "wgate": enc(B["wgate"]),
         "wdown": enc(B["wdown"]), "rms_w1": enc(B["rms1"]),
         "rms_w2": enc(B["rms2"]), "wlog": enc(Ep["wlog"]),
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

    words, n, i, topk, seedn, nrep = [], 20, 1, 12, 7, 4
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
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    out = encode(seed)
    rng = np.random.default_rng(seedn)
    for _ in range(n):
        lg = logits_of(out).copy()
        if len(out) >= nrep - 1:
            seen = {tuple(out[k:k + nrep]) for k in range(len(out) - nrep + 1)}
            prefix = tuple(out[-(nrep - 1):]) if nrep > 1 else ()
            for j in range(len(lg)):
                if prefix + (j,) in seen:
                    lg[j] = -1e9
        keep = np.argsort(-lg)[:topk]
        w = np.zeros_like(lg)
        w[keep] = np.exp(lg[keep] - lg[keep].max())
        w = w / w.sum()
        out.append(int(rng.choice(len(w), p=w)))
    print("seed:", seed)
    print("out :", decode(out))
    print(f"({len(out)} pieces -> words, no UNK class)")


if __name__ == "__main__":
    main()
