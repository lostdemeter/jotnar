"""Dual-channel gate: log-space AND separates overlap.

Program lm_dynaddr2.asm matches every row in HN space (contextual)
and E space (lexical) and fuses BEFORE softmax: a row must match
BOTH to win. Bank [Italy | Caesar-null], keys both spaces. Battery:
Italy at slots 0..7 must retrieve 0; Caesar must retrieve 1
(the false-positive that beat emb keys, negmine, and scales).
Alexander/as reported (unbanked third class, no gate). P read
host-side at the known entity row (precedent: catalog receipts).
Gates: 8/8 invariance + Caesar exclusion.
Usage: python3 research/dualchan.py (CPU lattice)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
KSH = 2.0
KSE = 2.0
FILL = [6, 0, 3, 28, 11, 2, 7, 5]


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    text = CFG + open(os.path.join(sdir, "lm_dynaddr2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, Uhn, Ue):
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
                             "Uhn": enc(Uhn), "Ue": enc(Ue)},
                            sigs=SIGS, basedir=sdir)

    def Pkey(f):
        # multi-out CALL binds DEF outputs to caller names (CPU path):
        # YP is ours; namespaced dualbank_apply#1.* are internals.
        assert "YP" in f, sorted(f.keys())
        return "YP"

    # mine HN-space keys (E-space keys are raw embeddings: free)
    f0 = run([6, 0, 3, 261, 11, 2, 7, 5],
             np.zeros((16, 2)), np.zeros((16, 2)))
    hI = dec(f0["HN"])[3]
    f1 = run([6, 0, 3, 40, 11, 2, 7, 5],
             np.zeros((16, 2)), np.zeros((16, 2)))
    hC = dec(f1["HN"])[3]
    eI = np.ascontiguousarray(dE["emb"][261])
    eI /= np.linalg.norm(eI)
    eC = np.ascontiguousarray(dE["emb"][40])
    eC /= np.linalg.norm(eC)
    Uhn = np.stack([hI / np.linalg.norm(hI), hC / np.linalg.norm(hC)],
                   axis=1) * KSH
    Ue = np.stack([eI, eC], axis=1) * KSE

    okp = 0
    for p in range(8):
        ids = list(FILL[:8])
        ids[p] = 261
        f = run(ids, Uhn, Ue)
        P = dec(f[Pkey(f)])[p]
        ret = int(np.argmax(P))
        okp += ret == 0
        print(f"pos={p}: P={np.round(P, 3)} ret={ret} "
              f"{'OK' if ret == 0 else 'MISS'}", flush=True)
    okn = 0
    for ent, name in ((40, "caesar"), (12, "alexander"), (7, "as")):
        ids = list(FILL[:8])
        ids[3] = ent
        f = run(ids, Uhn, Ue)
        P = dec(f[Pkey(f)])[3]
        ret = int(np.argmax(P))
        if name == "caesar":
            okn += ret == 1
            print(f"neg {name}: P={np.round(P, 3)} ret={ret} "
                  f"{'OK' if ret == 1 else 'FALSE-POSITIVE'}", flush=True)
        else:
            print(f"info {name}: P={np.round(P, 3)} ret={ret}", flush=True)
    print(f"invariance {okp}/8, caesar-exclusion {okn}/1", flush=True)


if __name__ == "__main__":
    main()
