"""Caesar maximin value: raw Rome row stalls (rank 5-7, egypt absorbs
dose ~as fast). Narrowed maximin: exact direction over Caesar's actual
top blockers under the FLAT head (linear post-L2 path, corr 1.000, so
float statics ARE dynamics). Base = zero-bank flat logits on the
Caesar prompt; A_c = (wU_rome - wU_c); req norm from deficits.
Install v*x req as the Caesar value (key unchanged), measure rank +
holds. Gates: FLIP at predicted norm (linearity receipt) or stall
with mechanism (which blocker binds).
Usage: python3 research/caesar_maximin.py (CPU lattice, fast)
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


def maximin(A, seeds=8, iters=4000):
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
    wlogU = np.ascontiguousarray(np.load(os.path.join(dd, "wlogU.npz"))["wlogU"])
    text = CFG + open(os.path.join(sdir, "lm_siphon.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run0(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text0, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(np.ascontiguousarray(dE["wlog"])),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt0[:, :1]),
                             "evb2": enc(np.zeros((1, 16)))},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    cprompt = ids_of("Julius Caesar was assassinated in the city of")[-8:]
    # flat-head zero-bank base: H4 through siphon front with empty route.
    # NOTE: flat logits need H5=H4 under wlogU: emulate by zero-value bank
    # through the real program (routing irrelevant at zero value).
    f0 = run0(cprompt)
    h4 = dec(f0["H4"])[-1]
    # HNB-space not needed: flat logits = H4 @ wlogU only if H5=H4 and no
    # second norm... lm_siphon H5=H4+YB(zero)=H4; LOGFLAT=H5@wlogU. Exact.
    lgf = h4 @ wlogU
    o = np.argsort(-lgf)[:10]
    print("flat zero-bank top10=" +
          str([(inv.get(int(i), "?"), round(float(lgf[int(i)]), 2)) for i in o]),
          flush=True)
    print(f"Rome flat-rank={int((lgf > lgf[98]).sum()) + 1}", flush=True)
    comps = [int(i) for i in o[:8] if int(i) != 98]
    A = np.stack([wlogU[:, 98] - wlogU[:, c] for c in comps], axis=0)
    deficits = np.array([float(lgf[c] - lgf[98]) for c in comps])
    vstar, t = maximin(A)
    print(f"maximin t/unit={t:.4f} deficits={np.round(deficits, 2)}", flush=True)
    Av = A.dot(vstar)
    if np.any(Av <= 1e-9):
        print("INFEASIBLE through flat head", flush=True)
        return
    req = float(np.max(deficits / Av))
    print(f"required ||v||={req:.2f} (Rome-row 12x norm="
          f"{12 * float(np.linalg.norm(evb0, axis=1).mean()):.2f})", flush=True)
    # install v* (mass-adjusted: Caesar routing w~0.6) as the Caesar value
    wmass = 0.6
    vuse = vstar * (req / wmass)
    epos = {"italy": 3, "alex": 0, "caesar": 1}
    eids = {"italy": 261, "alex": 12, "caesar": 40}
    order = ["italy", "alex", "caesar"]
    prompts = {"italy": ids_of("The capital of Italy is")[-8:],
               "alex": ids_of("Alexander the Great founded the city of")[-8:],
               "caesar": cprompt}
    text = CFG + open(os.path.join(sdir, "lm_siphon.asm")).read()
    wlog = np.ascontiguousarray(dE["wlog"])
    V = wlog.shape[1]

    def runS(ids, Uhn, Ue, Vc):
        toks = np.array(ids, dtype=np.int64)
        n = len(ids)
        pos = np.arange(n, dtype=np.int64)
        cm = np.tril(np.ones((n, n), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog), "wlogU": enc(wlogU),
                             "Uhn": enc(Uhn), "Ue": enc(Ue), "Vc": enc(Vc),
                            "unity": enc(np.ones(16)),
                             "onesS1": enc(np.ones((n, 1))),
                             "ones1V": enc(np.ones((1, V))),
                             "onesSV": enc(np.ones((n, V))),
                             "ukt": enc(ukt0), "evb": enc(evb0)},
                            sigs=SIGS, basedir=sdir)

    dd2 = dd
    lines = open(os.path.join(dd2, "lm_test.txt")).read().split("\n")[:10]
    emb = np.ascontiguousarray(dE["emb"])
    fkeys = {}
    for f in order:
        ids = prompts[f]
        h = dec(run0(ids)["HN"])[epos[f]]
        fkeys[f] = h / np.linalg.norm(h)
    bgHN, bgE = [], []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or 12 in ctx or 40 in ctx or len(ctx) < 3 \
                    or len(bgHN) >= 6:
                continue
            h = dec(run0(ctx)["HN"])[-1]
            bgHN.append(h / np.linalg.norm(h))
        if len(bgHN) >= 6:
            break
    for w in ["city", "son", "syria", "egypt", "her", "his"]:
        if w in vocab and len(bgE) < 6:
            e = np.ascontiguousarray(emb[vocab[w]])
            bgE.append(e / np.linalg.norm(e))
    KS = 2.0
    Uhn = np.concatenate([k[:, None] for k in bgHN]
                         + [fkeys[f][:, None] for f in order], axis=1) * KS
    Ue = np.concatenate([k[:, None] for k in bgE]
                        + [(np.ascontiguousarray(emb[eids[f]])
                            / np.linalg.norm(emb[eids[f]]))[:, None]
                           for f in order], axis=1) * KS
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    Vc = np.concatenate([np.zeros((6, 16)), (8.0 * evn * romeU)[None, :],
                         (16.0 * evn * alexU)[None, :], vuse[None, :]], axis=0)
    fi = runS(cprompt, Uhn, Ue, Vc)
    li = dec(fi["LOGITS2"])[-1]
    pr = dec(fi["PR"])[0]
    o = np.argsort(-li)[:5]
    print(f"maximin install: rank={int((li > li[98]).sum()) + 1} "
          f"top5={[(inv.get(int(i), '?'), round(float(li[int(i)]), 2)) for i in o]} "
          f"wsum={float(pr[6:9].sum()):.3f} "
          f"{'FLIP' if int(li.argmax()) == 98 else ''}", flush=True)


if __name__ == "__main__":
    main()
