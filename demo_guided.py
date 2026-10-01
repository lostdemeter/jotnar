"""Demo: guided generation (boundary steering, listings untouched).

Guides (all host-boundary, all declared in output):
- no-repeat N (blocks seen N-grams; default 4)
- repetition penalty rho (divides sampled weight of seen unigrams)
- topic steer: word list + strength s (adds s to their logits pre-softmax)
- UNK mask (always on, demo_lm precedent)
Topics steer WITHOUT touching weights/listings: the same doctrine as
temperature/top-k (host decoding rules, design LLM_DESIGN.md:20).
Run: python3 demo_guided.py [seed words...] [--topic w1,w2] [--strength 1.5]
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

CFG = "CONFIG m_acc 36849\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
WIN = 8


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data_ingest")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "wikitext103_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    E = np.load(os.path.join(dd, "w103_d32.npz"))
    b = np.load(os.path.join(dd, "w103_d32bank.npz"))
    d = np.load(os.path.join(root, "data", "lm_d32_fit.npz"))
    B = {k: np.array(d[k]) for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown", "rms1", "rms2"]}
    text = CFG + open(os.path.join(sdir, "lm_d32_depth4.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(E["emb"]), "wq": enc(B["wq"]), "wk": enc(B["wk"]),
         "wv": enc(B["wv"]), "wo": enc(B["wo"]),
         "wup": enc(B["wup"]), "wgate": enc(B["wgate"]),
         "wdown": enc(B["wdown"]), "rms_w1": enc(B["rms1"]),
         "rms_w2": enc(B["rms2"]), "wlog": enc(E["wlog"]),
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

    args, n, i = [], 20, 1
    topk, seedn, nrep, rho = 15, 7, 4, 1.3
    topic, strength = [], 1.5
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
        elif a == "--topic" and i + 1 < len(sys.argv):
            topic = [w.strip().lower() for w in sys.argv[i + 1].split(",")]
            i += 2
        elif a == "--strength" and i + 1 < len(sys.argv):
            strength = float(sys.argv[i + 1])
            i += 2
        elif a == "--rho" and i + 1 < len(sys.argv):
            rho = float(sys.argv[i + 1])
            i += 2
        else:
            args.append(a)
            i += 1
    seed = " ".join(args) if args else "the king of"
    tids = {vocab[w] for w in topic if w in vocab}
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    rng = np.random.default_rng(seedn)
    out = list(ids)
    seen1 = set(ids)
    for _ in range(n):
        lg = logits_of(out).copy()
        lg[0] = -1e9
        if len(out) >= nrep - 1:
            seen = {tuple(out[k:k + nrep]) for k in range(len(out) - nrep + 1)}
            prefix = tuple(out[-(nrep - 1):]) if nrep > 1 else ()
            for j in range(len(lg)):
                if prefix + (j,) in seen:
                    lg[j] = -1e9
        for j in tids:
            lg[j] += strength
        keep = np.argsort(-lg)[:topk]
        w = np.zeros_like(lg)
        w[keep] = np.exp(lg[keep] - lg[keep].max())
        for j in seen1:
            w[j] /= rho
        w = w / w.sum()
        nxt = int(rng.choice(len(w), p=w))
        out.append(nxt)
        seen1.add(nxt)
    print(f"seed: {seed} | topic={topic} s={strength} rho={rho} nrep={nrep}")
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))


if __name__ == "__main__":
    main()
