"""Micro-dose re-ladder: were the content verdicts overshoot artifacts?

content_flip.py started at 8x and found nothing to 64x (truth
direction). The NY window (install exactly @0.05, dead by 0.25)
proves ladders starting high can MISS narrow windows entirely.
Same candidates/method (thinnest-margin content positions, post-L2
bank, truth-direction values), doses 0.05/0.1/0.25/0.5/1/2. Any FLIP
rewrites a "never flips" verdict with mechanism (overshoot); none
confirms infeasibility at BOTH ends (too small to matter, too big
to survive) -- the complete dose picture per candidate.
Usage: python3 research/micro_ladder.py (CPU lattice, fast)
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
DOSES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt2, evb2):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    cands = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            if truth == 0 or freq[truth] >= fcut:
                continue
            ctx = ids[max(0, k - 8):k]
            f = run(ctx, ukt0[:, :1], np.zeros((1, 16)))
            lg = dec(f["LOGITS2"])[-1]
            o = np.argsort(-lg)
            cands.append((float(lg[o[0]] - lg[truth]), ctx, truth))
    cands.sort()
    print("thinnest: " + str([(inv.get(t, "?"), round(m, 2))
                              for m, _, t in cands[:6]]), flush=True)
    for m, ids, truth in cands[:3]:
        f = run(ids, ukt0[:, :1], np.zeros((1, 16)))
        hnb = dec(f["HNB"])[-1]
        key = (hnb / np.linalg.norm(hnb))[:, None]
        vd = np.ascontiguousarray(wlog[:, truth])
        vd = vd / np.linalg.norm(vd)
        cells = []
        for dx in DOSES:
            f2 = run(ids, key, (dx * evn * vd)[None, :])
            lg = dec(f2["LOGITS2"])[-1]
            top = int(lg.argmax())
            rk = int((lg > lg[truth]).sum()) + 1
            cells.append(f"{dx}x:r{rk}{'!' if top == truth else ''}")
        print(f"truth={inv.get(truth, '?')} margin={m:.2f} " + " ".join(cells),
              flush=True)


if __name__ == "__main__":
    main()
