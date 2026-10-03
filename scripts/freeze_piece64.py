"""Freeze piece-64 ends + bank (offline, D64-refit Step 0).

Groki BPE-2000 piece bigram counts (data/piece_bigrams.npz, built by the
piece-fit arc from the frozen 938/235 train split, seed 0) -> log1p SVD
rank-64 ends (E = U*sqrt(s), W = sqrt(s)*Vt, same recipe as
scripts/freeze_piece.py at rank 16 and scripts/freeze_w20k.py at 16/32)
+ row-sum top-64 bank (same reverse-engineered recipe as bankp32_64:
evb = E[words], ukt = unit(E[words]).T, stable sort) + manifest
(spec64, nnz, top200/top64 mass, provenance). Deterministic throughout
(stable sorts only; SVD is the only float op and is byte-pinned by the
manifest spec64, not by bytes).
Usage: python3 scripts/freeze_piece64.py
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")


def main():
    counts = np.load(os.path.join(OUT, "piece_bigrams.npz"))["counts"]
    assert counts.shape == (2038, 2038), counts.shape
    V = counts.shape[0]
    nnz = int((counts > 0).sum())
    fr = counts.sum(axis=0).astype(np.float64)
    fr /= fr.sum()
    fr = np.sort(fr)[::-1]
    L = np.log1p(counts.astype(np.float64))
    U, sv, Vt = np.linalg.svd(L, full_matrices=False)
    k = 64
    emb = U[:, :k] * np.sqrt(sv[:k])
    wlog = np.sqrt(sv[:k])[:, None] * Vt[:k, :]
    np.savez(os.path.join(OUT, "lm_piece64.npz"), emb=emb, wlog=wlog)
    occ = counts.sum(axis=1)
    top = np.argsort(-occ, kind="stable")[:64]
    evb = emb[top]
    ukt = (evb / np.linalg.norm(evb, axis=1, keepdims=True)).T.copy()
    np.savez(os.path.join(OUT, "bankpiece64_d64.npz"),
             ukt=ukt, evb=evb, words=top)
    man = {"V": V, "nnz": nnz,
           "spec64": round(float(sv[0] / sv[63]), 2),
           "spec32": round(float(sv[0] / sv[31]), 2),
           "top200": round(float(fr[:200].sum()), 3),
           "top64mass": round(float(fr[:64].sum()), 3),
           "counts": "data/piece_bigrams.npz (groki train split, seed 0)",
           "recipe": "log1p SVD rank-64 (E=U*sqrt(s), W=sqrt(s)*Vt) + "
                     "row-sum top-64 bank (stable), same as bankp32_64",
           "words_sha": hashlib.sha256(
               np.ascontiguousarray(top).tobytes()).hexdigest()[:16]}
    json.dump(man, open(os.path.join(OUT, "lm_piece64_manifest.json"),
                        "w"), indent=2)
    print(f"V={V} nnz={nnz} spec64={sv[0] / sv[63]:.2f} "
          f"spec32={sv[0] / sv[31]:.2f} top200={fr[:200].sum():.3f}")
    print(f"wrote {OUT}/lm_piece64.npz + bankpiece64_d64.npz + manifest")


if __name__ == "__main__":
    main()
