"""First native content flip: margins + extended dose on best candidates.

Curve 1 verdict: content 0/40 at every K; UNK carries the score.
Install physics says dose > local margin + correct direction, so:
(1) record base margins on all content positions (which are thin?),
(2) native dose ladder (8x/16x/32x/64x values) on the thinnest-margin
candidates. A flip anywhere = first native content top-1 = curve 2's
first non-zero point (organization creating content, not preserving).
Usage: python3 research/content_flip.py
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
DOSES = [8, 16, 32, 64]


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
        return dec(f["LOGITS"])[-1], dec(f["HN"])[-1]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    cands = []  # (margin, ids, truth)
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            if truth == 0 or freq[truth] >= fcut:
                continue
            lg, hn = run(ids[max(0, k - 8):k], ukt0, evb0)
            o = np.argsort(-lg)
            cands.append((float(lg[o[0]] - lg[truth]), ids[max(0, k - 8):k],
                          truth, hn))
    cands.sort()
    print(f"content positions: {len(cands)}; thinnest margins: "
          f"{[(inv.get(t, '?'), round(m, 2)) for m, _, t, _ in cands[:6]]}",
          flush=True)
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    for m, ids, truth, hn in cands[:3]:
        vd = np.ascontiguousarray(wlog[:, truth])
        vd = vd / np.linalg.norm(vd)
        key = hn / np.linalg.norm(hn)
        for dx in DOSES:
            ukt = np.concatenate([ukt0, (2.0 * key)[:, None]], axis=1)
            evb = np.concatenate([evb0, (dx * evn * vd)[None, :]], axis=0)
            lg, _ = run(ids, ukt, evb)
            top = int(lg.argmax())
            print(f"truth={inv.get(truth, '?')} base-margin={m:.2f} "
                  f"dose={dx}x top={inv.get(top, '?')} "
                  f"{'FLIP' if top == truth else ''}", flush=True)


if __name__ == "__main__":
    main()
