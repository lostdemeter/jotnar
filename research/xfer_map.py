"""Shared residual-anchored correspondence map (teacher L2 -> our HN).

Single source of truth for transfer addressing (was duplicated across
key_rank/resid_map with divergent bugs). Anchors: same texts through
both models (teacher L2 row <-> our HN row), ridge fit, held-out
cosine reported. Promoted on third use.
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"


def our_hn(ids, dd=None, sdir=None):
    """Our HN rows (S,D16 float) for a token-id prefix (lattice)."""
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = dd or os.path.join(ROOT, "data")
    sdir = sdir or os.path.join(ROOT, "programs")
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    f = ASM.run_text(text, REGISTRY,
                     {"tok": toks, "pos": pos, "cmask": cm,
                      "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                      "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                      "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                      "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                      "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                      "wlog": enc(dE["wlog"]),
                      "ukt": enc(b["ukt"]), "evb": enc(b["evb"])},
                     sigs=SIGS, basedir=sdir)
    return dec(f["HN"])


def build_resid_map(n_lines=100, n_cap=80, ridge=1e3, verbose=True,
                    cache="/tmp/xfer_Mr.npz"):
    """Ridge map teacher-L2 (3584) -> our-HN (16) + held-out cosine.
    Deterministic inputs (seeded split, frozen snapshot) so the fit is
    cached on disk; delete the cache to refit."""
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    import json
    if os.path.isfile(cache):
        z = np.load(cache)
        if verbose:
            print(f"resid map: cached ({int(z['n'])} anchors, "
                  f"held-out cos={float(z['held']):.3f})", flush=True)
        return z["Mr"], {"n": int(z["n"]), "held": float(z["held"])}
    _, tok7 = load7b()
    dd = os.path.join(ROOT, "data")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")
    A_src, A_tgt = [], []
    for s in lines[:n_lines]:
        ids = ids_of(s)
        if len(ids) < 6:
            continue
        t7, tids7 = fwdH(s, keep="all")
        hn = our_hn(ids[:5])
        kt = min(int(round(4 / max(len(ids) - 1, 1) * (len(tids7) - 1))), len(tids7) - 1)
        A_src.append(t7[2][kt])
        A_tgt.append(hn[min(4, len(hn) - 1)])
        if len(A_src) >= n_cap:
            break
    A_src = np.stack(A_src)
    A_tgt = np.stack(A_tgt)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(A_src))
    ntr = int(0.8 * len(A_src))
    tr, te = idx[:ntr], idx[ntr:]

    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    Mr = np.linalg.solve(A_src[tr].T @ A_src[tr] + ridge * np.eye(3584),
                         A_src[tr].T @ A_tgt[tr])
    held = float(np.mean([cos(A_src[te][i] @ Mr, A_tgt[te][i]) for i in range(len(te))]))
    np.savez(cache, Mr=Mr, n=len(A_src), held=held)
    if verbose:
        print(f"resid map: {len(A_src)} anchors, held-out cos={held:.3f}", flush=True)
    return Mr, {"n": len(A_src), "held": held}
