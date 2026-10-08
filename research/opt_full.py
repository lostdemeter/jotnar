"""Full-vocab maximin direction + empirical dose ladder (overwrite regime)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
IDS = [0, 6, 0, 3, 137, 0, 28, 0]
TRUTH = 74
DOSES = [10, 20, 40, 64]


def maximin(A, seeds=6, iters=4000):
    rng = np.random.default_rng(1)
    best_t, best_v = -1e18, None
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
            step *= 0.9995
        t = float(np.min(A.dot(v)))
        if t > best_t:
            best_t, best_v = t, v.copy()
    return best_v, best_t


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

    f0 = run(IDS, ukt0[:, :1], np.zeros((1, 16)))
    lg0 = dec(f0["LOGITS2"])[-1]
    hnb0 = dec(f0["HNB"])[-1]
    V = wlog.shape[1]
    keep = [c for c in range(V) if c != TRUTH]
    A = np.stack([wlog[:, TRUTH] - wlog[:, c] for c in keep], axis=0)
    v, t = maximin(A)
    print(f"full-vocab t/unit={t:.4f} per-1x={t * evn:.4f}", flush=True)
    key = (hnb0 / np.linalg.norm(hnb0))[:, None]
    for dx in DOSES:
        f2 = run(IDS, key, (dx * evn * v)[None, :])
        lg = dec(f2["LOGITS2"])[-1]
        actual = lg - lg0
        pred = (dx * evn * v) @ wlog
        o = np.argsort(-lg)[:5]
        print(f"dose={dx}x top5=" + str([(inv.get(int(i), "?"),
                                         round(float(lg[int(i)]), 2)) for i in o])
              + f" truth-rank={int((lg > lg[TRUTH]).sum()) + 1}"
              + f" corr={float(np.corrcoef(actual, pred)[0, 1]):.3f}"
              + f" {'FLIP' if int(lg.argmax()) == TRUTH else ''}", flush=True)


if __name__ == "__main__":
    main()
