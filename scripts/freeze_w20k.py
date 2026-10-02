"""Freeze w20k piece supply (offline, P2 data audit follow-up).

Wikitext-2 train (HF parquet cache) -> sentences (same >=4-word
split as mine_edges.load_wiki2) -> groki-BPE encode (OOV-free by
base-alphabet construction; skips counted, never silent) -> bigram
counts (2038x2038, LOCAL-only per .gitignore precedent) -> log1p
SVD rank-16/32 ends (tracked evidence) + row-sum top-64 banks
(same reverse-engineered recipe as bankp32: evb=E[words],
ukt=unit(E[words]).T) + manifest (spec/mass for the peakiness
readout). Deterministic throughout (stable sorts, fixed recipes).
Spectral flattening (ends-side knob, velocity arc): E_tau =
U[:,:k]*s[:k]**(tau/2), WLOG_tau = s[:k]**(tau/2)*Vt[:k,:];
tau=1.0 reproduces (groki-fit geometry assumed); tau=0.6 matches
groki spec32 (4.6) from w20k 12.9 -- gradient confirmed, not gated.
Usage: python3 scripts/freeze_w20k.py (fast: ~85k sents, minutes)
"""
import html
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")
PARQ = ("/home/thorin/.cache/huggingface/hub/datasets--wikitext/snapshots"
        "/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"
        "/train-00000-of-00001.parquet")


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    import pandas as pd
    df = pd.read_parquet(PARQ, columns=["text"])
    sents = []
    for t in [str(t) for t in df["text"].tolist()]:
        for s in re.split(r"(?<=[.!?])\s+", html.unescape(t)):
            if len(s.strip().split()) >= 4:
                sents.append(s.strip())
    vocab = json.load(open(os.path.join(OUT, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(OUT, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in words_of(s):
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
            try:
                out.extend(vocab[p] for p in syms)
            except KeyError:
                return None
        return out

    V = len(vocab)
    counts = np.zeros((V, V), dtype=np.int64)
    nskip = ntok = 0
    for s in sents:
        ids = encode(s)
        if ids is None:
            nskip += 1
            continue
        ntok += len(ids)
        for x, y in zip(ids[:-1], ids[1:]):
            counts[x, y] += 1
    np.savez(os.path.join(OUT, "w20k_piece_bigrams.npz"), counts=counts)
    fr = counts.sum(axis=0).astype(np.float64)
    fr /= fr.sum()
    fr = np.sort(fr)[::-1]
    L = np.log1p(counts.astype(np.float64))
    U, sv, Vt = np.linalg.svd(L, full_matrices=False)
    for k, tag in ((16, "w20k_piece16"), (32, "w20k_piece32")):
        np.savez(os.path.join(OUT, f"{tag}.npz"),
                 emb=U[:, :k] * np.sqrt(sv[:k]),
                 wlog=np.sqrt(sv[:k])[:, None] * Vt[:k, :])
    occ = counts.sum(axis=1)
    for E_, tag in (("w20k_piece16", "w20k_bank16"),
                    ("w20k_piece32", "w20k_bank32")):
        E = np.load(os.path.join(OUT, f"{E_}.npz"))["emb"]
        top = np.argsort(-occ, kind="stable")[:64]
        evb = E[top]
        ukt = (E[top] / np.linalg.norm(E[top], axis=1, keepdims=True)).T.copy()
        np.savez(os.path.join(OUT, f"{tag}.npz"), ukt=ukt, evb=evb, words=top)
    json.dump({"V": V, "nnz": int((counts > 0).sum()),
               "spec16": round(float(sv[0] / sv[15]), 1),
               "spec32": round(float(sv[0] / sv[31]), 1),
               "n_sents": len(sents), "skip": nskip,
               "top200": round(float(fr[:200].sum()), 3),
               "top64": round(float(fr[:64].sum()), 3)},
              open(os.path.join(OUT, "w20k_piece_manifest.json"), "w"), indent=2)
    print(f"sents={len(sents)} skip={nskip} toks={ntok} "
          f"pieces/sent={ntok / max(len(sents) - nskip, 1):.2f}")
    print(f"nnz={int((counts > 0).sum())} top200={fr[:200].sum():.3f} "
          f"spec16={sv[0] / sv[15]:.1f} spec32={sv[0] / sv[31]:.1f}")
    print(f"wrote {OUT}/w20k_piece_bigrams.npz (local) + ends + banks + manifest")


if __name__ == "__main__":
    main()
