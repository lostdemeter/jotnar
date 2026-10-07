"""Residual-level correspondence: same prompts through both models.

transfer_v1's fitted map failed (0/4 retrieval) plausibly because its
anchors pair EMBEDDINGS while the map is applied to L2 RESIDUAL rows
(out-of-domain). This builds anchors at the right level: teacher L2
row <-> our HN row on shared prompts, fits ridge, held-out cosine.
If residual-anchored maps clear ~0.8+, they go into the transfer
screen; if not, the correspondence itself is the wall (stated).
Usage: python3 research/resid_map.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"


def main():
    from qwen_torch import fwdH
    from chain.qwen7b import load7b
    _, tok7 = load7b()
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import json
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def our_hn(ids):
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

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:30]
    A_src, A_tgt, kept = [], [], []
    for s in lines:
        ids = ids_of(s)
        if len(ids) < 6:
            continue
        k = 4  # mid-sentence row, both models see prefix ids[:5]
        pre = ids[:5]
        # teacher side sees the same TEXT (its own tokenization)
        t7, tids7 = fwdH(s, keep="all")
        hn = our_hn(pre)
        # align by TEXT position k: teacher row with closest relative slot
        # (lengths differ by tokenizer; use proportional position)
        kt = min(int(round(k / max(len(ids) - 1, 1) * (len(tids7) - 1))), len(tids7) - 1)
        A_src.append(t7[2][kt])
        A_tgt.append(hn[min(k, len(hn) - 1)])
        kept.append(s[:40])
        if len(A_src) >= 40:
            break
    A_src = np.stack(A_src)
    A_tgt = np.stack(A_tgt)
    print(f"residual anchors: {len(A_src)}", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(A_src))
    ntr = int(0.8 * len(A_src))
    tr, te = idx[:ntr], idx[ntr:]

    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    Xt, Xv = A_src[tr], A_src[te]
    Yt, Yv = A_tgt[tr], A_tgt[te]
    for a in (1e0, 1e1, 1e2, 1e3):
        Mr = np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ Yt)
        print(f"resid ridge a={a:.0e} held-out cos: "
              f"{np.mean([cos(Xv[i] @ Mr, Yv[i]) for i in range(len(te))]):.3f}",
              flush=True)
    print("chance: ~0.25", flush=True)


if __name__ == "__main__":
    main()
