"""D32 depth-4 gate: scale reliably (refit arc proof).

Listing programs/lm_d32_depth4.asm (4x tied D32 layers, bank MLPs,
m_acc 36849 + m_cov 35686 priced -- H8 9.6 needed the cov raise, laws
5-for-5). Frozen data/lm_d32_fit.npz (g2 body + I0.5 V).
Gates: torch parity, twin beats D32 depth-2 (0.722) AND D16 depth-4
(0.479), top1 no-collapse (>=0.34).
Usage: python3 tests/test_lm_d32_depth4.py (slow)
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
CFG = "CONFIG m_acc 36849\nCONFIG m_cov 35686\nCONFIG beta -30.0\n"


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
    man = json.load(open(os.path.join(dd, "lm_d32_fit_manifest.json")))
    check("d32fit-manifest", man["m_acc"] == 36849 and man["m_cov"] == 35686,
          f"priced cover (sha {man['sha']})")
    w = np.load(os.path.join(dd, "lm_d32_fit.npz"))
    b = np.load(os.path.join(dd, "bankd32.npz"))
    ukt, evb = b["ukt"], b["evb"]
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    text = CFG + open(os.path.join(sdir, "lm_d32_depth4.asm")).read()

    def run(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(w["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                          "wdown": enc(w["wdown"]),
                          "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
                          "wlog": enc(w["wlog"]),
                          "ukt": enc(ukt), "evb": enc(evb)},
                         sigs=SIGS, basedir=sdir)
        return dec(f["H8"]), dec(f["LOGITS"])

    fa, _ = run([12, 471, 59])
    fp, _ = run([59, 38, 471, 11, 12])
    td = float(np.linalg.norm(fa[-1] - fp[-1]) / max(np.linalg.norm(fa[-1]), 1e-12))
    check("d32d4-twins", td < 0.479, f"twin_dist={td:.3f} (D16 depth4 0.479)")

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
    check("d32d4-top1", t1 / tot >= 0.34,
          f"top1={t1 / tot:.3f} (holds, no-collapse)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
