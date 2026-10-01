"""Twin fingerprints: order-sensitive encoding in assembly (measured).

Fingerprint = causal H4 last-position row (fixed geometry, order flows
through mask + RoPE structurally). Gates: reversal sensitivity (order
matters -- must differ), twin active/passive difference MEASURED not
barred (random weights carry no content: invariance is the bar for
non-random weights, stated here with today's number as the baseline).
Same pattern as test_xf_block's measured out-of-contract row.
Usage: python3 tests/test_twin_print.py
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
ACTIVE = [12, 471, 59]
PASSIVE = [59, 38, 471, 11, 12]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def fingerprint(d, ids, text, sdir):
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    feeds = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
    return dec(feeds["H4"])[-1]


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    text = open(os.path.join(sdir, "lm_depth2causal.asm")).read()
    d = np.load(os.path.join(root, "data", "lm_block1.npz"))
    fa = fingerprint(d, ACTIVE, text, sdir)
    fp = fingerprint(d, PASSIVE, text, sdir)
    fr = fingerprint(d, ACTIVE[::-1], text, sdir)
    d_rev = float(np.linalg.norm(fa - fr) / max(np.linalg.norm(fa), 1e-12))
    check("print-reversal-sensitive", d_rev > 0.1,
          f"rel-dist {d_rev:.3f} (order structurally encoded)")
    d_tw = float(np.linalg.norm(fa - fp) / max(np.linalg.norm(fa), 1e-12))
    print(f"print-twin-distance-measured: {d_tw:.3f} (random weights: "
          f"no content to preserve -- invariance bar waits on non-random "
          f"weights; this number is the baseline, not a failure)")
    fa2 = fingerprint(d, ACTIVE, text, sdir)
    check("print-deterministic", bool((fa == fa2).all()),
          "same order twice bit-exact (machinery adds nothing)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
