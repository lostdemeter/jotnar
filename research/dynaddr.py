"""Dynamic addressing gate: entity anywhere, retrieval invariant.

Program lm_dynaddr.asm finds the entity row by content (dynaddr_match:
MATMUL match + ARGMAX row + GATHER, no position literals) and reads
bank retrieval there. Battery: Italy entity at every window slot
0..7 (filler contexts) must retrieve the Italy store (index 128);
non-Italy windows must not. Gates: 8/8 position invariance +
exclusion on controls. A baked-position address scores 1/8 here by
construction -- this gate is the anti-template.
Usage: python3 research/dynaddr.py (CPU lattice)
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
KS = 2.0
FILL = [6, 0, 3, 28, 11, 2, 7, 5]  # frequent filler ids (a/<unk>/of/but/...)


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
    ukt0 = b["ukt"]
    text = CFG + open(os.path.join(sdir, "lm_dynaddr.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    e261 = np.ascontiguousarray(dE["emb"][261])
    e261 /= np.linalg.norm(e261)
    ckey = (e261 * KS)[:, None]
    ukt = np.concatenate([ukt0, (e261 * 2.0)[:, None]], axis=1)
    KIT = ukt.shape[1] - 1

    def run(ids):
        toks = np.array(ids, dtype=np.int64)
        n = len(ids)
        pos = np.arange(n, dtype=np.int64)
        cm = np.tril(np.ones((n, n), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                          "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                          "wo": enc(dE["wo"]),
                          "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                          "ukt": enc(ukt), "ckey": enc(ckey)},
                         sigs=SIGS, basedir=sdir)
        return f

    def Ikey(f):
        c = [k for k in f if "dynaddr_match" in k and k.endswith(".I")]
        assert len(c) == 1, c
        return c[0]

    def ival2(t):
        return int(np.ascontiguousarray(t).flat[-1])

    ok_pos, ok_neg = 0, 0
    for p in range(8):
        ids = list(FILL[:8])
        ids[p] = 261
        f = run(ids)
        ret = ival2(f["RET"])
        found = ival2(f[Ikey(f)])
        good = ret == KIT
        ok_pos += good
        print(f"pos={p}: found-row={found} (want {p}) ret={ret} "
              f"{'OK' if good else 'MISS'}", flush=True)
    for ent, name in ((40, "caesar"), (12, "alexander"), (7, "as")):
        ids = list(FILL[:8])
        ids[3] = ent
        f = run(ids)
        ret = ival2(f["RET"])
        good = ret != KIT
        ok_neg += good
        print(f"neg {name}: ret={ret} {'OK' if good else 'FALSE-POSITIVE'}",
              flush=True)
    print(f"invariance {ok_pos}/8, exclusion {ok_neg}/3", flush=True)
    # contrast key: unit(HN_italy - HN_caesar), same filler, pos 3 --
    # discriminative addressing (the maximin lesson, now for keys).
    # argmax is scale-invariant per-row; only DIRECTION separates overlap.
    fI = run([6, 0, 3, 261, 11, 2, 7, 5])
    fC = run([6, 0, 3, 40, 11, 2, 7, 5])
    hI = dec(fI["HN"])[3]
    hC = dec(fC["HN"])[3]
    ck = hI - hC
    ck /= np.linalg.norm(ck)
    ukt3 = np.concatenate([ukt0, (ck * 2.0)[:, None]], axis=1)

    def run3(ids):
        toks = np.array(ids, dtype=np.int64)
        n = len(ids)
        pos = np.arange(n, dtype=np.int64)
        cm = np.tril(np.ones((n, n), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]),
                             "rms_w1": enc(dE["rms1"]),
                             "rms_w2": enc(dE["rms2"]),
                             "ukt": enc(ukt3), "ckey": enc(ckey)},
                            sigs=SIGS, basedir=sdir)

    okc, okn = 0, 0
    for p in range(8):
        ids = list(FILL[:8])
        ids[p] = 261
        okc += ival2(run3(ids)["RET"]) == KIT
    for ent, name in ((40, "caesar"), (12, "alexander"), (7, "as")):
        ids = list(FILL[:8])
        ids[3] = ent
        r = ival2(run3(ids)["RET"])
        good = r != KIT
        okn += good
        print(f"contrast neg {name}: ret={r} {'OK' if good else 'STILL-POSITIVE'}",
              flush=True)
    print(f"contrast invariance {okc}/8, exclusion {okn}/3", flush=True)
    if ok_neg == 3:
        return
    # negmine pass: false-positive HNB... HN-space rows as null stores
    # (the loop is the product: addressing that learns its mistakes)
    import phi_core.lattice as S2
    from chain import asm as ASM2
    fneg = run([6, 0, 3, 40, 11, 2, 7, 5])
    xa = dec(fneg["XA"])[-1:]
    e40 = np.ascontiguousarray(dE["emb"][40])
    e40 /= np.linalg.norm(e40)
    ukt2 = np.concatenate([ukt, (xa[0] / np.linalg.norm(xa[0]) * 2.0)[:, None]],
                          axis=1)

    def run2(ids):
        toks = np.array(ids, dtype=np.int64)
        n = len(ids)
        pos = np.arange(n, dtype=np.int64)
        cm = np.tril(np.ones((n, n), dtype=np.int64))
        return ASM2.run_text(text, REGISTRY,
                             {"tok": toks, "pos": pos, "cmask": cm,
                              "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                              "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                              "wo": enc(dE["wo"]),
                              "rms_w1": enc(dE["rms1"]),
                              "rms_w2": enc(dE["rms2"]),
                              "ukt": enc(ukt2), "ckey": enc(ckey)},
                             sigs=SIGS, basedir=sdir)

    ok2 = 0
    for ent, name in ((40, "caesar"), (12, "alexander"), (7, "as")):
        ids = list(FILL[:8])
        ids[3] = ent
        ret = ival2(run2(ids)["RET"])
        good = (ret != KIT) if ent != 261 else (ret == KIT)
        ok2 += good
        print(f"negmine neg {name}: ret={ret} "
              f"{'OK' if good else 'STILL-POSITIVE'}", flush=True)
    okp = 0
    for p in range(8):
        ids = list(FILL[:8])
        ids[p] = 261
        okp += ival2(run2(ids)["RET"]) == KIT
    print(f"negmine invariance {okp}/8, exclusion {ok2}/3", flush=True)


if __name__ == "__main__":
    main()
