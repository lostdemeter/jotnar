"""Key-rank gate: does a mapped teacher key win retrieval, on paper?

Ranks mapped-key dot (ridge-residual and ridge-embedding maps) vs the
128 native keys on real Italy HN queries. Top-1 here is NECESSARY for
any lattice install (not sufficient: softmax/values follow). Cheap
minutes; gates the expensive full screen.
Usage: python3 research/key_rank.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"


def main():
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    import torch
    _, tok7 = load7b()
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import json
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0 = b["ukt"]
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def our_hn(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                          "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                          "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                          "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                          "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                          "wlog": enc(dE["wlog"]),
                          "ukt": enc(ukt0), "evb": enc(b["evb"])},
                         sigs=SIGS, basedir=sdir)
        return dec(f["HN"])

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")
    anchor_lines = lines[:100]
    A_src, A_tgt = [], []
    for s in anchor_lines:
        ids = ids_of(s)
        if len(ids) < 6:
            continue
        t7, tids7 = fwdH(s, keep="all")
        hn = our_hn(ids[:5])
        kt = min(int(round(4 / max(len(ids) - 1, 1) * (len(tids7) - 1))), len(tids7) - 1)
        A_src.append(t7[2][kt])
        A_tgt.append(hn[min(4, len(hn) - 1)])
        if len(A_src) >= 80:
            break
    print(f"residual anchors: {len(A_src)}", flush=True)
    A_src = np.stack(A_src)
    A_tgt = np.stack(A_tgt)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(A_src))
    ntr = int(0.8 * len(A_src))
    tr, te = idx[:ntr], idx[ntr:]
    Mr = np.linalg.solve(A_src[tr].T @ A_src[tr] + 1e3 * np.eye(3584),
                         A_src[tr].T @ A_tgt[tr])
    # Italy key through the residual map
    z = np.load("/tmp/t_xfer.npz")
    kR = (z["k"] @ Mr)
    kR /= np.linalg.norm(kR)
    # embedding-anchored ridge map (transfer_v1 recipe) for comparison
    E7 = load7b()[0]("model.embed_tokens.weight")
    SA, TA = [], []
    for w, i in vocab.items():
        if w == "<unk>" or len(w) < 2:
            continue
        ids = tok7(" " + w, return_tensors="pt")["input_ids"][0].numpy()
        if len(ids) > 4:
            continue
        SA.append(E7[ids].mean(axis=0))
        TA.append(np.ascontiguousarray(dE["emb"][i]))
    SA, TA = np.stack(SA), np.stack(TA)
    Me = np.linalg.solve(SA.T @ SA + 10.0 * np.eye(3584), SA.T @ TA)
    kE = (z["k"] @ Me)
    kE /= np.linalg.norm(kE)
    # rank resid-mapped key at several scales (unit-norm is convention;
    # softmax competition cares about dot magnitude, so scale is free)
    for sc in (1.0, 2.0, 4.0):
        kRs = kR * sc
        ranks = []
        for s in lines:
            ids = ids_of(s)
            for k, v in enumerate(ids):
                if v != 261 or k < 1 or len(ranks) >= 4:
                    continue
                hn = our_hn(ids[max(0, k - 7):k + 1])[-1]
                dots = hn @ ukt0
                ranks.append(int((dots > hn @ kRs).sum()) + 1)
        print(f"residmap x{sc}: ranks={ranks}", flush=True)
    # emb-map once (dead-last baseline, unscaled)
    ranks = []
    for s in lines:
        ids = ids_of(s)
        for k, v in enumerate(ids):
            if v != 261 or k < 1 or len(ranks) >= 4:
                continue
            hn = our_hn(ids[max(0, k - 7):k + 1])[-1]
            dots = hn @ ukt0
            ranks.append(int((dots > hn @ kE).sum()) + 1)
    print(f"embmap x1: ranks={ranks}", flush=True)


if __name__ == "__main__":
    main()
