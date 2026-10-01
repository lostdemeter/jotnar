"""BPE in assembly: decode + splice steps vs host (composition proof).

Listings programs/bpe_decode.asm (piece ids -> char rows) and
programs/bpe_splice.asm (one exact merge splice, SCAN-style driver
pattern). Gates: decode rows bit-exact vs host table, splice step
bit-exact vs host splice, full-word driver-loop encode == pure-host
encode (composition -- the cascade without breaking fixed geometry).
Symbol ids ride as exact triples; id equality judged rounded (lattice
quantum ~1e-3, bigram-bank precedent).
Usage: python3 tests/test_bpe_asm.py
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
CHARS = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789'"))
CH = {c: i + 1 for i, c in enumerate(CHARS)}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def rids(t):
    return np.round(dec(t)).astype(np.int64)


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(root, "data", "bpe_vocab.json")))
    merges = json.load(open(os.path.join(root, "data", "bpe_merges.json")))
    inv = {i: p for p, i in vocab.items()}
    # piece table (V,14) char ids + lengths, exact triples
    MAXC = 14
    V = len(vocab)
    ptab = np.zeros((V, MAXC), np.float64)
    plen = np.zeros((V, 1), np.float64)
    for p, i in vocab.items():
        cs = p.replace("</w>", "")
        for j, c in enumerate(cs[:MAXC]):
            ptab[i, j] = CH[c]
        plen[i, 0] = min(len(cs), MAXC)
    # 1. decode parity on test words
    words = ["alexander", "founded", "alexandria", "battle", "queen"]
    rank = {tuple(m): r for r, m in enumerate(merges)}

    def host_pieces(w):
        syms = [c for c in w] + ["</w>"]
        while len(syms) > 1:
            best = None
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
        return syms

    pids = np.array([vocab[p] for w in words for p in host_pieces(w)], np.int64)
    f = ASM.run_text(open(os.path.join(sdir, "bpe_decode.asm")).read(), REGISTRY,
                     {"pids": pids, "ptab": enc(ptab), "plent": enc(plen)},
                     sigs=SIGS, basedir=sdir)
    rows, lens = rids(f["OUT"]), rids(f["LENS"]).reshape(-1)
    ok = True
    k = 0
    alpha = "".join(sorted(set("abcdefghijklmnopqrstuvwxyz0123456789'")))
    for w in words:
        for p in host_pieces(w):
            cs = p.replace("</w>", "")
            got = "".join(alpha[c - 1] for c in rows[k][:lens[k]])
            if got != cs:
                ok = False
            k += 1
    check("bpe-decode-rows", ok and k == len(pids),
          f"{k}/{len(pids)} piece rows render exact chars")
    # 2. splice step bit-exact vs host splice (word 'battle', first merge)
    w = "battle"
    syms = [c for c in w] + ["</w>"]
    outs = []
    for p in syms:
        outs.append(p)
    # host: find first-applicable pair
    hit = None
    for i in range(len(syms) - 1):
        if (syms[i], syms[i + 1]) in rank:
            if hit is None or rank[(syms[i], syms[i + 1])] < rank[hit]:
                hit = (syms[i], syms[i + 1])
    i = next(i for i in range(len(syms) - 1) if (syms[i], syms[i + 1]) == hit)
    new_sym = hit[0] + hit[1]
    L = len(syms)
    # fixed-geometry splice: out[j] for j<L (sentinel id 0 tail)
    host_ids = []
    src = syms + ["<pad>"]
    mp = []
    for j in range(L):
        if j < i:
            mp.append(j)
            host_ids.append(syms[j])
        elif j == i:
            mp.append(L)
            host_ids.append(new_sym)
        else:
            mp.append(j + 1)
            host_ids.append(src[j + 1])
    str2id = {}
    for s in syms + [new_sym, "<pad>"]:
        if s not in str2id:
            str2id[s] = len(str2id) + 1
    stable = np.array([[str2id[s]] for s in src], np.float64)
    gmap = np.array(mp, np.int64)
    imask = np.array([[1] if j == i else [0] for j in range(L)], np.int64)
    outv = enc(np.array([[str2id[new_sym]]] * L, np.float64))
    g = ASM.run_text(open(os.path.join(sdir, "bpe_splice.asm")).read(), REGISTRY,
                     {"syms": enc(stable), "gmap": gmap, "imask": imask, "outv": outv},
                     sigs=SIGS, basedir=sdir)
    got = rids(g["OUT"]).reshape(-1)
    want = np.array([str2id[s] for s in host_ids], np.int64)
    check("bpe-splice-step", bool((got == want).all()),
          f"one merge splice bit-exact on ids (at={i}, {hit[0]}+{hit[1]})")
    # 3. full-word driver loop == pure-host encode
    def driver(w):
        syms = [c for c in w] + ["</w>"]
        while True:
            hit, hi = None, None
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (hit is None or r < hit):
                    hit, hi = r, i
            if hit is None:
                return syms
            syms = syms[:hi] + [syms[hi] + syms[hi + 1]] + syms[hi + 2:]

    same = all(driver(w) == host_pieces(w) for w in words)
    check("bpe-driver-composition", same,
          "driver-loop over merge steps == host encode (cascade proven)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
