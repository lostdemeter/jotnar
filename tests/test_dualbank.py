"""Dual-channel addressing gate (generic ball, stdlib dualbank).

Program lm_dynaddr2.asm matches every row in HN space (contextual)
and E space (lexical) and fuses BEFORE softmax (log-space AND):
a row must match BOTH to win. Bank [Italy | Caesar-null], ks 2/2
both channels, no tuning. Battery: Italy at slots 0..7 retrieves 0;
Caesar retrieves 1 (the false-positive that beat emb keys, negmine,
and the 0.5-4.0 scale sweep).
Gates: 8/8 invariance + Caesar exclusion (Alexander/as reported).
Usage: python3 tests/test_dualbank.py (fast: ~15 runs)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
KSH = 2.0
KSE = 2.0
FILL = [6, 0, 3, 28, 11, 2, 7, 5]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    text = CFG + open(os.path.join(sdir, "lm_dynaddr2.asm")).read()

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

    f0 = run([6, 0, 3, 261, 11, 2, 7, 5], np.zeros((16, 2)), np.zeros((16, 2)))
    hI = dec(f0["HN"])[3]
    f1 = run([6, 0, 3, 40, 11, 2, 7, 5], np.zeros((16, 2)), np.zeros((16, 2)))
    hC = dec(f1["HN"])[3]
    eI = np.ascontiguousarray(dE["emb"][261])
    eI /= np.linalg.norm(eI)
    eC = np.ascontiguousarray(dE["emb"][40])
    eC /= np.linalg.norm(eC)
    Uhn = np.stack([hI / np.linalg.norm(hI), hC / np.linalg.norm(hC)],
                   axis=1) * KSH
    Ue = np.stack([eI, eC], axis=1) * KSE

    inv = 0
    for p in range(8):
        ids = list(FILL[:8])
        ids[p] = 261
        f = run(ids, Uhn, Ue)
        P = dec(f["YP"])[p]
        inv += int(np.argmax(P)) == 0
    check("dualbank-invariance", inv == 8, f"{inv}/8 slots retrieve Italy")
    exc = 0
    for ent, name in ((40, "caesar"),):
        ids = list(FILL[:8])
        ids[3] = ent
        f = run(ids, Uhn, Ue)
        P = dec(f["YP"])[3]
        good = int(np.argmax(P)) == 1
        exc += good
        check(f"dualbank-exclusion-{name}", good,
              f"P={np.round(P, 3).tolist()}")
    for ent, name in ((12, "alexander"), (7, "as")):
        ids = list(FILL[:8])
        ids[3] = ent
        f = run(ids, Uhn, Ue)
        P = dec(f["YP"])[3]
        print(f"dualbank-info-{name}: P={np.round(P, 3).tolist()} "
              f"ret={int(np.argmax(P))} (unbanked, reported)", flush=True)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
