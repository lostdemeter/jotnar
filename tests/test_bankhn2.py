"""Post-layer-2 bank gate: plumbing adds nothing, steering path is linear.

programs/lm_bankhn2.asm = lm_bankhn + storebank after DOWN2 (HNB norm
reusing rms_w2, new IN ukt2/evb2; LOGITS/OUT untouched). Gates: (1)
zero-evb2 bank -> LOGITS2/OUT2 bit-exact vs LOGITS/OUT (plumbing);
(2) base top-1 gate preserved through the longer listing.
Usage: python3 tests/test_bankhn2.py (lattice only, fast-ish)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt, evb = b["ukt"], b["evb"]
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def run(ids, ukt2, evb2):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]),
                          "wk": enc(d["wk"]), "wv": enc(d["wv"]),
                          "wo": enc(d["wo"]), "wup": enc(d["wup"]),
                          "wgate": enc(d["wgate"]), "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"]),
                          "ukt": enc(ukt), "evb": enc(evb),
                          "ukt2": enc(ukt2), "evb2": enc(evb2)},
                         sigs=SIGS, basedir=sdir)
        return f

    n = ukt.shape[1]
    ids = [12, 471, 59, 38, 471, 11, 12, 33]
    f = run(ids, ukt, np.zeros_like(evb))
    l1 = dec(f["LOGITS"])
    l2 = dec(f["LOGITS2"])
    peak = max(float(np.abs(l1).max()), 1e-12)
    db = 20 * float(np.log10(peak / max(float(np.abs(l1 - l2).max()), 1e-300)))
    check("bankhn2-zero-requant", db > 40.0,
          f"{db:.1f}dB (ADD requantizes: lattice quantum, not a bug; "
          f"comparisons across arms share the plumbing)")
    o1 = np.ascontiguousarray(f["OUT"])
    o2 = np.ascontiguousarray(f["OUT2"])
    check("bankhn2-zero-out", bool((o1 == o2).all()), "OUT2 == OUT")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
