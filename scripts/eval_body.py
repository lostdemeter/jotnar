"""Eval harness for body search (offline): twin-dist + top1 + ret-hit.

Metric costs (SVD body, S<=5): twin_dist = 16 LM runs (cheapest,
content-direct); ret_hit = same 16 + host recall (free); top1_sample =
~200 runs (validation only). Search on twin_dist, validate winners on
the rest. All runs through programs/lm_depth2causal.asm + SVD head.
Usage: python3 scripts/eval_body.py (baseline numbers only)
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
CFG = "CONFIG m_acc 35492\nCONFIG m_cov 35048\n"
ACTIVE = [12, 471, 59]
PASSIVE = [59, 38, 471, 11, 12]


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def load_all():
    d = dict(np.load(os.path.join(DD, "lm_svd.npz")))
    d["P"] = np.load(os.path.join(DD, "cueproj.npz"))["P"]
    hk = np.load(os.path.join(DD, "hkeys.npz"))
    d["hkeys"], d["key_edge"] = hk["keys"], hk["key_edge"]
    d["vocab"] = json.load(open(os.path.join(DD, "lm_vocab.json")))
    d["text"] = CFG + open(os.path.join(ROOT, "programs",
                                        "lm_depth2causal.asm")).read()
    d["sdir"] = os.path.join(ROOT, "programs")
    return d


def run_h4(d, ids, weights=None):
    w = weights or d
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    f = ASM.run_text(d["text"], REGISTRY,
                     {"tok": toks, "pos": pos, "cmask": cm,
                      "emb": enc(w["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                      "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                      "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                      "wdown": enc(w["wdown"]),
                      "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
                      "wlog": enc(w["wlog"])}, sigs=SIGS, basedir=d["sdir"])
    return dec(f["H4"])


def twin_dist(d, weights=None):
    fa = run_h4(d, ACTIVE, weights)[-1]
    fp = run_h4(d, PASSIVE, weights)[-1]
    return float(np.linalg.norm(fa - fp) / max(np.linalg.norm(fa), 1e-12))


def main():
    d = load_all()
    t0 = time.time()
    print(f"twin_dist baseline: {twin_dist(d):.3f} ({time.time()-t0:.0f}s/2 runs)")
    print("random-body reference: 0.845 (test_twin_print)")


if __name__ == "__main__":
    main()
