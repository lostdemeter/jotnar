"""Cue-from-hidden gate: projection parity + hidden retrieval baseline.

Listing programs/cueproj.asm (H @ P) vs host (parity). Baseline: twin
sentence -> causal SVD LM -> H4-last @ P -> sign -> recall over hkeys ->
tag-exact rate MEASURED not barred (random body scrambles content; the
bar waits on body weights -- same honest pattern as twin 0.845).
Usage: python3 tests/test_cuehidden.py
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
CFG = "CONFIG m_acc 35492\nCONFIG m_cov 35048\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    P = np.load(os.path.join(dd, "cueproj.npz"))["P"]
    hk = np.load(os.path.join(dd, "hkeys.npz"))
    hkeys, key_edge = hk["keys"], hk["key_edge"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))]
    edges = recs[1:]
    voice = json.load(open("/home/thorin/Documents/OpenCode/Echion_Revisted/data/voice_pairs.json"))
    d = np.load(os.path.join(dd, "lm_svd.npz"))
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    lmtext = CFG + open(os.path.join(sdir, "lm_depth2causal.asm")).read()

    def run_h4(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(lmtext, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
        return dec(f["H4"])

    # 1. projection parity (listing MATMUL vs host, S=4 H4)
    man = json.load(open(os.path.join(dd, "lm_block1_manifest.json")))
    H4 = run_h4(man["toks"])
    g = ASM.run_text(CFG + open(os.path.join(sdir, "cueproj.asm")).read(), REGISTRY,
                     {"h": enc(H4), "proj": enc(P)}, sigs=SIGS, basedir=sdir)
    ref = H4 @ P
    got = dec(g["OUT"])
    mse = float(np.mean((got - ref) ** 2))
    db = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("cueproj-parity", db >= BAR_DB, f"{db:.2f}dB (H~3 x 0.25 regime)")
    # 2. hidden-cue retrieval baseline (measured, not barred)
    ok = tot = 0
    for p in voice:
        for side in ("active", "passive"):
            ids = [vocab.get(w, 0) for w in words_of(p[side])]
            H = run_h4(ids)
            cue = np.sign(H[-1] @ P)
            ki = int(np.argmax(hkeys @ cue))
            tot += 1
            if edges[int(key_edge[ki])]["tag"] == p.get("twin"):
                ok += 1
    print(f"cuehidden-baseline-measured: {ok}/{tot} tag-exact from hidden "
          f"cues (random body scrambles content -- bar waits on body "
          f"weights; word-cue bridge holds 16/16, this is the gap to close)")
    check("cuehidden-runs", tot == 16, f"{tot} hidden cues scored")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
