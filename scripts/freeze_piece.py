"""Freeze piece bigram counts + spectral ends (offline, BPE rebuild).

Train sentences (same 938/235 split, seed 0) -> BPE piece-encode ->
bigram counts (2038x2038) -> log1p SVD rank-16 -> lm_piece.npz
(emb=U√s, wlog=√sVᵀ) + manifest (spec ratio for the decaying check).
Counts are large (32MB, git-ignored like stores blobs); ends (512K)
are tracked frozen evidence. Regeneration is deterministic.
Usage: python3 scripts/freeze_piece.py
"""
import glob
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")
EW = "</w>"


class _T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.p = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        self.skip = tag in ("script", "style", "nav", "header", "footer", "aside")

    def handle_endtag(self, tag):
        self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.p.append(d)


def main():
    vocab = json.load(open(os.path.join(OUT, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(OUT, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in re.findall(r"[a-z0-9']+", s.lower()):
            syms = [c for c in w] + [EW]
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
            out.extend(vocab[p] for p in syms)
        return out

    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    rng = np.random.default_rng(0)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    cut = int(0.8 * len(sents))
    train = [sents[i] for i in idx[:cut]]
    V = len(vocab)
    counts = np.zeros((V, V), dtype=np.int64)
    for s in train:
        ids = encode(s)
        for x, y in zip(ids[:-1], ids[1:]):
            counts[x, y] += 1
    np.savez(os.path.join(OUT, "piece_bigrams.npz"), counts=counts)
    L = np.log1p(counts.astype(np.float64))
    U, s, Vt = np.linalg.svd(L, full_matrices=False)
    k = 16
    np.savez(os.path.join(OUT, "lm_piece.npz"), emb=U[:, :k] * np.sqrt(s[:k]),
             wlog=np.sqrt(s[:k])[:, None] * Vt[:k, :])
    json.dump({"V": V, "nnz": int((counts > 0).sum()),
               "spec16": round(float(s[0] / s[15]), 1)},
              open(os.path.join(OUT, "lm_piece_manifest.json"), "w"), indent=2)
    print(f"V={V} nnz={(counts > 0).sum()} spec16={s[0] / s[15]:.1f}")
    print(f"wrote {OUT}/piece_bigrams.npz (local) + lm_piece.npz (tracked)")


if __name__ == "__main__":
    main()
