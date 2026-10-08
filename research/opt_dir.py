"""Maximin (N-way) steering direction + required dose per content candidate."""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
CTXS = {
    "amid": ([0, 6, 0, 3, 137, 0, 28, 0], 74),
    "son": ([0, 0, 61, 0], 147),
    "without": ([0, 312, 104, 31, 201, 0, 0, 0], 83),
}


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

    for name, (ids, truth) in CTXS.items():
        f0 = run(ids, ukt0[:, :1], np.zeros((1, 16)))
        lg0 = dec(f0["LOGITS2"])[-1]
        hnb0 = dec(f0["HNB"])[-1]
        o = np.argsort(-lg0)
        rank = int((lg0 > lg0[truth]).sum()) + 1
        print(f"truth={name} logit={float(lg0[truth]):.2f} rank={rank}", flush=True)
        print(" top10=" + str([(inv.get(int(i), "?"), round(float(lg0[int(i)]), 2))
                               for i in o[:10]]), flush=True)
        comps = [int(i) for i in o[:8] if int(i) != truth]
        A = np.stack([wlog[:, truth] - wlog[:, c] for c in comps], axis=0)
        deficits = np.array([float(lg0[c] - lg0[truth]) for c in comps])
        rng = np.random.default_rng(0)
        best_v, best_t = None, -1e18
        for _ in range(8):
            v = rng.normal(size=16)
            v /= np.linalg.norm(v)
            step = 0.2
            for _it in range(3000):
                scores = A.dot(v)
                j = int(np.argmin(scores))
                g = A[j] - (A[j].dot(v)) * v
                n = np.linalg.norm(g)
                if n > 1e-12:
                    v = v + step * g / n
                    v /= np.linalg.norm(v)
                step *= 0.9993
            t = float(np.min(A.dot(v)))
            if t > best_t:
                best_t, best_v = t, v.copy()
        v = best_v
        print(f" opt worst-gain/unit={best_t:.4f} per-1x={best_t * evn:.3f}", flush=True)
        Av = A.dot(v)
        req = []
        for k, c in enumerate(comps):
            req.append((inv.get(c, "?"), float(deficits[k] / (evn * Av[k]))
                        if Av[k] > 1e-9 else float("inf")))
        finite = [r for _, r in req if np.isfinite(r) and r > 0]
        print(" req-per-comp=" + str([(a, round(r, 2) if np.isfinite(r) else "inf")
                                      for a, r in req]), flush=True)
        print(f" max-req-dose={max(finite) if finite else 'inf'}", flush=True)
        vt = wlog[:, truth] / np.linalg.norm(wlog[:, truth])
        print(f" truth-dir worst/unit={float(np.min(A.dot(vt))):.4f} "
              f"per-1x={float(np.min(A.dot(vt))) * evn:.3f}", flush=True)
        lead = int(o[0])
        vd = wlog[:, truth] - wlog[:, lead]
        vd /= np.linalg.norm(vd)
        print(f" lead-disc worst/unit={float(np.min(A.dot(vd))):.4f} "
              f"per-1x={float(np.min(A.dot(vd))) * evn:.3f}", flush=True)
        # empirical test of the optimal direction at its predicted dose
        key = (hnb0 / np.linalg.norm(hnb0))[:, None]
        for dx in sorted(set([1, 2, 4] + ([round(max(finite), 1)] if finite else []))):
            if not np.isfinite(dx) or dx <= 0 or dx > 64:
                continue
            f2 = run(ids, key, (dx * evn * v)[None, :])
            lg = dec(f2["LOGITS2"])[-1]
            top = int(lg.argmax())
            print(f"  dose={dx}x top={inv.get(top, '?')} "
                  f"truth-rank={int((lg > lg[truth]).sum()) + 1} "
                  f"{'FLIP' if top == truth else ''}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
