"""Bank-MLP gate: counts-structured DOWN (paths A+B+C convergent).

Listing programs/lm_bankhn2.asm (layer-1 MLP replaced by softmax
storebank over top-128 next-word stores: HN@ukt -> TSHIFT+WIDE ->
@evb; layer-2 swiglu control). Frozen data/bankhn.npz (unit-norm ukt,
next-emb evb; m_acc 36118 priced for bank-write magnitudes -- global
raise held other paths, per-block M-dicts stay follow-up).
Gates: parity vs torch mirror, twin_dist beats IvoQ (0.832),
top1 no-collapse (>=0.34 vs 0.354).
Usage: python3 tests/test_lm_bankhn.py (slow: parity + ~200 runs)
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

BAR_DB = 40.0
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
    man = json.load(open(os.path.join(dd, "bankhn_manifest.json")))
    check("bankhn2-manifest", man["m_acc"] == 36118,
          f"priced cover (sha {man['sha']})")
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt, evb = b["ukt"], b["evb"]
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def run(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"]),
                          "ukt": enc(ukt), "evb": enc(evb)},
                         sigs=SIGS, basedir=sdir)
        return dec(f["H4"]), dec(f["LOGITS"])

    import sys as _s
    _s.path.insert(0, os.path.join(root, "tests"))
    fa, _ = run([12, 471, 59])
    fp, _ = run([59, 38, 471, 11, 12])
    td = float(np.linalg.norm(fa[-1] - fp[-1]) / max(np.linalg.norm(fa[-1]), 1e-12))
    check("bankhn2-twins", td < 0.78, f"twin_dist={td:.3f} (bankhn 0.781)")

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    t1 = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            _, lg = run(ids[max(0, k - 8):k])
            t1 += (int(np.argmax(lg[-1])) == ids[k])
            tot += 1
    check("bankhn2-top1", t1 / tot >= 0.34,
          f"top1={t1 / tot:.3f} (bankhn 0.350, no-collapse)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
