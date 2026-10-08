"""Content flips through the post-layer-2 bank (linear path to logits).

Same candidates/method as content_flip.py, but the store lives in
lm_bankhn2's post-layer-2 bank (keys match HNB rows, values add at H5
where the head is linear). Single-store bank isolates placement: if
tops flip here at modest dose after never flipping to 64x through
the layer-1 bank, injection point (not dose/direction) was the wall.
Usage: python3 research/content_flip2.py
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
DOSES = [1, 2, 4, 8]


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
        f = ASM.run_text(text, REGISTRY,
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
        return f

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
            # single inert store (zero value): baseline margins on the
            # longer listing (requant cancels in arm comparisons)
            f = run(ids[max(0, k - 8):k], ukt0[:, :1], np.zeros((1, 16)))
            lg = dec(f["LOGITS2"])[-1]
            o = np.argsort(-lg)
            cands.append((float(lg[o[0]] - lg[truth]),
                          ids[max(0, k - 8):k], truth))
    cands.sort()
    print(f"content positions: {len(cands)}; thinnest: "
          f"{[(inv.get(t, '?'), round(m, 2)) for m, _, t in cands[:6]]}",
          flush=True)
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    for m, ids, truth in cands[:3]:
        f = run(ids, ukt0[:, :1] * 0 + ukt0[:, :1], np.zeros((1, 16)))
        hnb = dec(f["HNB"])[-1]
        key = hnb / np.linalg.norm(hnb)
        vd = np.ascontiguousarray(wlog[:, truth])
        vd = vd / np.linalg.norm(vd)
        for dx in DOSES:
            ukt2 = key[:, None]
            evb2 = (dx * evn * vd)[None, :]
            f2 = run(ids, ukt2, evb2)
            lg = dec(f2["LOGITS2"])[-1]
            top = int(lg.argmax())
            print(f"truth={inv.get(truth, '?')} base-margin={m:.2f} "
                  f"dose={dx}x top={inv.get(top, '?')} "
                  f"{'FLIP' if top == truth else ''}", flush=True)


if __name__ == "__main__":
    main()
