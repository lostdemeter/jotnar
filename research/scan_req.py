"""Scan all content positions: rank, margin, full-vocab required dose (float maximin)."""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"


def maximin_t(A, seeds=4, iters=2500):
    rng = np.random.default_rng(0)
    best = -1e18
    for _ in range(seeds):
        v = rng.normal(size=A.shape[1])
        v /= np.linalg.norm(v)
        step = 0.2
        for _it in range(iters):
            s = A.dot(v)
            j = int(np.argmin(s))
            g = A[j] - (A[j].dot(v)) * v
            n = np.linalg.norm(g)
            if n > 1e-12:
                v = v + step * g / n
                v /= np.linalg.norm(v)
            step *= 0.9993
        best = max(best, float(np.min(A.dot(v))))
    return best


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

    def run(ids):
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
                          "ukt2": enc(ukt0[:, :1]),
                          "evb2": enc(np.zeros((1, 16)))},
                         sigs=SIGS, basedir=sdir)
        return dec(f["LOGITS2"])[-1]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    rows = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            if truth == 0 or freq[truth] >= fcut:
                continue
            ctx = ids[max(0, k - 8):k]
            lg = run(ctx)
            o = np.argsort(-lg)
            margin = float(lg[o[0]] - lg[truth])
            rank = int((lg > lg[truth]).sum()) + 1
            rows.append((margin, rank, truth, tuple(ctx)))
    rows.sort()
    print(f"content positions: {len(rows)}", flush=True)
    V = wlog.shape[1]
    for margin, rank, truth, ctx in rows[:12]:
        keep = [c for c in range(V) if c != truth]
        A = np.stack([wlog[:, truth] - wlog[:, c] for c in keep], axis=0)
        lg = run(list(ctx))
        deficits = np.array([float(lg[c] - lg[truth]) for c in keep])
        t = maximin_t(A)
        Av_note = ""
        if t > 1e-9:
            pos = deficits > 0
            # need per-row gain, not just worst: approximate req with worst-gain
            # lower bound; refine only for promising (req<12) candidates
            approx = float(deficits.max() / (evn * t))
            Av_note = f"approx-req={approx:.1f}x"
        print(f"{inv.get(truth, '?'):12} margin={margin:.2f} rank={rank:3} "
              f"t/unit={t:.4f} per1x={t * evn:.3f} {Av_note}", flush=True)


if __name__ == "__main__":
    main()
