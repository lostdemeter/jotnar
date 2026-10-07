"""Map diagnostic: is 3584->16 correspondence learnable from 499 anchors?

Held-out test of three map families (raw least-squares, ridge sweep,
two-stage PCA->ridge). If none clears the bar, no downstream transfer
result means anything and the map is the problem, not the idea.
Also reports PCA energy (how concentrated teacher anchor geometry is)
and the dose-ladder target both tracks need.
Usage: python3 research/map_diag.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))


def main():
    from qwen_torch import _get as _tget
    _, _, tok7 = _tget()
    import torch
    from chain.qwen7b import load7b
    g7, _ = load7b()
    E7 = g7("model.embed_tokens.weight")
    dd = os.path.join(ROOT, "data")
    import json
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    A_src, A_tgt, words = [], [], []
    for w, i in vocab.items():
        if w == "<unk>" or len(w) < 2:
            continue
        ids = tok7(" " + w, return_tensors="pt")["input_ids"][0].numpy()
        if len(ids) > 4:
            continue
        A_src.append(E7[ids].mean(axis=0))
        A_tgt.append(np.ascontiguousarray(dE["emb"][i]))
        words.append(w)
    A_src = np.stack(A_src)
    A_tgt = np.stack(A_tgt)
    print(f"anchors: {len(words)}", flush=True)
    # PCA energy of teacher anchors
    U, Sv, _ = np.linalg.svd(A_src - A_src.mean(0), full_matrices=False)
    e = np.cumsum(Sv ** 2)
    e /= e[-1]
    for k in (16, 32, 64, 128):
        print(f"teacher-anchor PCA@{k}: {e[k-1]:.3f} variance", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(words))
    ntr = int(0.8 * len(words))
    tr, te = idx[:ntr], idx[ntr:]
    Xt, Xv = A_src[tr], A_src[te]
    Yt, Yv = A_tgt[tr], A_tgt[te]

    def cos(a, b):
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    Mls, _, _, _ = np.linalg.lstsq(Xt, Yt, rcond=None)
    print(f"raw-LS held-out cos: {np.mean([cos(Xv[i] @ Mls, Yv[i]) for i in range(len(te))]):.3f}",
          flush=True)
    for a in (1e1, 1e2, 1e3, 1e4):
        Mr = np.linalg.solve(Xt.T @ Xt + a * np.eye(Xt.shape[1]), Xt.T @ Yt)
        print(f"ridge a={a:.0e} held-out cos: "
              f"{np.mean([cos(Xv[i] @ Mr, Yv[i]) for i in range(len(te))]):.3f}",
              flush=True)
    for k in (32, 64, 128):
        Uk, Sk, _ = np.linalg.svd(Xt - Xt.mean(0), full_matrices=False)
        P = (Uk[:, :k] * Sk[:k]).T  # not needed; project directly below
        mu = Xt.mean(0)
        # PCA basis from train, applied to test (no leakage)
        _, _, Vt = np.linalg.svd(Xt - mu, full_matrices=False)
        Bk = Vt[:k].T
        Zt, Zv = (Xt - mu) @ Bk, (Xv - mu) @ Bk
        Mr = np.linalg.solve(Zt.T @ Zt + 1e2 * np.eye(k), Zt.T @ Yt)
        pred = Zv @ Mr
        print(f"two-stage PCA{k}+ridge held-out cos: "
              f"{np.mean([cos(pred[i], Yv[i]) for i in range(len(te))]):.3f}",
              flush=True)
    # baseline: what does chance look like?
    print(f"chance cos (|N(0,1/sqrt16)|): ~{1/4:.3f}", flush=True)


if __name__ == "__main__":
    main()
