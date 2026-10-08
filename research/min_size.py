"""Minimum size, curve 1 (unstructured baseline): bank K sweep.

How small can the store get before structure breaks? Truncate bankhn
to top-K stores (by evb row-norm: keep the giants, documented), run
the top-1 probe per K, split hits into glue (frequent-truth) vs
content (rest) by bigram frequency rank. Expectation: content hits
cliff to zero below some K (the minimum!), glue degrades gracefully.
Curve 2 (next): same sweep on an ORGANIZED yarn ball (hierarchical
gates + exact tier + orthogonal bank) -- organization should shift
the cliff left (more aim per parameter). The gap between curves IS
the value of organization, measured.
Usage: python3 research/min_size.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
KS = [128, 64, 32, 16, 8]


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt, evb):
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
                          "wlog": enc(wlog),
                          "ukt": enc(ukt), "evb": enc(evb)},
                         sigs=SIGS, basedir=sdir)
        return dec(f["LOGITS"])[-1]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    order = np.argsort(-np.linalg.norm(evb0, axis=1))
    for K in KS:
        keep = np.sort(order[:K])
        ukt, evb = ukt0[:, keep], evb0[keep, :]
        gh = gt = ch = ct = uh = ut = tot = 0
        for s in lines:
            ids = ids_of(s)
            for k in range(1, len(ids)):
                truth = ids[k]
                lg = run(ids[max(0, k - 8):k], ukt, evb)
                hit = int(lg.argmax()) == truth
                tot += 1
                ut += hit
                if truth == 0:
                    uh += hit  # UNK-truth counted separately (gate parity)
                    continue
                if freq[truth] >= np.sort(freq)[-64]:
                    gh += hit
                    gt += 1
                else:
                    ch += hit
                    ct += 1
        print(f"K={K:3}: top1={ut / max(tot, 1):.3f} "
              f"glue={gh}/{gt} content={ch}/{ct} unk={uh}", flush=True)


if __name__ == "__main__":
    main()
