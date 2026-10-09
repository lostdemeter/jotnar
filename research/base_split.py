"""Base full split: the comparison the breadth split lacked.

fullsplit.log measured the dualhead2 bank at full scale (319 lines):
top1 0.237, glue 147/3000, content 1/1774, unk 1779. Uninterpretable
alone (10-line base was 0.337, later lines may just be harder). This
runs the SKEWED BASE (lm_bankhn2, zero bank) over all lines: if base
is also ~0.24, scale degrades the MODEL not the bank (prior intact
relatively); if base stays ~0.34, the bank costs at scale (breadth
finding against the ball). One number decides which.
Usage: python3 research/base_split.py (CPU lattice, ~8000 runs)
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


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    wlog = np.ascontiguousarray(dE["wlog"])
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")
    gh = gt = ch = ct = uh = ut = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            toks = np.array(ids[max(0, k - 8):k], dtype=np.int64)
            n = len(toks)
            pos = np.arange(n, dtype=np.int64)
            cm = np.tril(np.ones((n, n), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": toks, "pos": pos, "cmask": cm,
                              "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                              "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                              "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                              "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                              "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                              "wlog": enc(wlog),
                              "ukt": enc(ukt0), "evb": enc(evb0),
                              "ukt2": enc(ukt0[:, :1]),
                              "evb2": enc(np.zeros((1, 16)))},
                             sigs=SIGS, basedir=sdir)
            lg = dec(f["LOGITS2"])[-1]
            hit = int(lg.argmax()) == truth
            tot += 1
            ut += hit
            if truth == 0:
                uh += hit
                continue
            if freq[truth] >= fcut:
                gh += hit
                gt += 1
            else:
                ch += hit
                ct += 1
    print(f"base full: top1={ut / max(tot, 1):.3f} glue={gh}/{gt} "
          f"content={ch}/{ct} unk={uh} "
          f"(bank full was: top1 0.237 glue 147/3000 content 1/1774 unk 1779)",
          flush=True)


if __name__ == "__main__":
    main()
