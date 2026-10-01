"""Freeze bigram LM data (offline, v1.6 construction): grokipedia HTML ->
sentences -> seeded split -> word vocab (top-V + UNK0) -> bigram counts.

Deterministic throughout (sorted traversal, fixed seed, frequency-ranked
ids: UNK=0, then rank order). Writes data/lm_bigrams.npz (counts V+1 x
V+1 int64) + vocab.json + manifest sidecar (counts, seed, coverage, input
shas). Freezing is offline by doctrine; the listing consumes frozen data.
Usage: python3 scripts/freeze_lm.py [--v 512] [--seed 0]
"""
import glob
import hashlib
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = "/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia"
OUT = os.path.join(ROOT, "data")


class _T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.p = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        self.skip = tag in ("script", "style", "nav", "header", "footer",
                            "aside")

    def handle_endtag(self, tag):
        self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.p.append(d)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sents = []
    shas = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.html"))):
        raw = open(f, encoding="utf-8", errors="replace").read()
        shas[os.path.basename(f)] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        t = _T()
        t.feed(raw)
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    cut = int(0.8 * len(sents))
    train = [sents[i] for i in idx[:cut]]
    test = [sents[i] for i in idx[cut:]]
    toks = [w for s in train for w in re.findall(r"[a-z0-9']+", s.lower())]
    from collections import Counter
    c = Counter(toks)
    top = [w for w, _ in c.most_common(a.v)]
    vocab = {"<unk>": 0}
    vocab.update({w: i + 1 for i, w in enumerate(top)})
    cov = sum(n for _, n in c.most_common(a.v)) / max(len(toks), 1)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    V = a.v + 1
    counts = np.zeros((V, V), dtype=np.int64)
    for s in train:
        ids = ids_of(s)
        for x, y in zip(ids[:-1], ids[1:]):
            counts[x, y] += 1
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "lm_bigrams.npz"), counts=counts)
    with open(os.path.join(OUT, "lm_vocab.json"), "w") as fh:
        json.dump(vocab, fh)
    manifest = {"seed": a.seed, "v": a.v, "n_train": len(train),
                "n_test": len(test), "coverage": cov,
                "src_shas": shas,
                "counts_sha": hashlib.sha256(counts.tobytes()).hexdigest()[:16]}
    with open(os.path.join(OUT, "lm_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(OUT, "lm_test.txt"), "w") as fh:
        fh.write("\n".join(test))
    print(f"train={len(train)} test={len(test)} V={V} coverage={cov:.3f} "
          f"nnz={int((counts > 0).sum())}/{V*V}")
    print(f"wrote {OUT}/lm_bigrams.npz + vocab + manifest")


if __name__ == "__main__":
    main()
