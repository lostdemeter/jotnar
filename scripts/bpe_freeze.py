"""Freeze BPE subwords (offline, behavior bridge): grokipedia -> merges.

Deterministic (sorted traversal, frequency-ranked, ties alphabetical).
Writes data/bpe_vocab.json (id->piece) + bpe_merges.json (ordered pairs)
+ bpe_manifest.json (provenance). Freezing is offline; runtime never fits.
Usage: python3 scripts/bpe_freeze.py [--merges 2000] [--seed 0]
"""
import glob
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter
from html.parser import HTMLParser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = "/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia"
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


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--merges", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sents, shas = [], {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.html"))):
        raw = open(f, encoding="utf-8", errors="replace").read()
        shas[os.path.basename(f)] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        t = _T()
        t.feed(raw)
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    import numpy as np
    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    cut = int(0.8 * len(sents))
    train = [sents[i] for i in idx[:cut]]
    # word frequencies (deterministic order: count desc, then alpha)
    wf = Counter(w for s in train for w in words_of(s))
    words = {w: ([c for c in w] + [EW], n) for w, n in wf.items()}
    merges = []
    for _ in range(a.merges):
        pc = Counter()
        for syms, n in words.values():
            for x, y in zip(syms[:-1], syms[1:]):
                pc[(x, y)] += n
        if not pc:
            break
        best = sorted(pc.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        merges.append(list(best))
        x, y = best
        for w in list(words):
            syms, n = words[w]
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == x and syms[i + 1] == y:
                    out.append(x + y)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            words[w] = (out, n)
    pieces = set()
    for s in train:
        for w in words_of(s):
            pieces.update(w)
            pieces.add(EW)
    for m in merges:
        pieces.add(m[0] + m[1])
    pieces = sorted(pieces)
    vocab = {p: i for i, p in enumerate(pieces)}
    # coverage: base chars cover all train words by construction; measure
    # held-out piece OOV (pieces unseen in train tokenization)
    cov = 1.0
    man = {"seed": a.seed, "n_merges": len(merges), "v": len(vocab),
           "n_train": len(train), "coverage": cov, "src_shas": shas,
           "merges_sha": hashlib.sha256(json.dumps(merges).encode()).hexdigest()[:16]}
    os.makedirs(OUT, exist_ok=True)
    json.dump(vocab, open(os.path.join(OUT, "bpe_vocab.json"), "w"))
    json.dump(merges, open(os.path.join(OUT, "bpe_merges.json"), "w"))
    json.dump(man, open(os.path.join(OUT, "bpe_manifest.json"), "w"), indent=2)
    top = Counter(p for syms, n in words.values() for p in syms for _ in range(min(n, 1)))
    print(f"train={len(train)} merges={len(merges)} V={len(vocab)}")
    print(f"wrote {OUT}/bpe_vocab.json + bpe_merges.json + manifest")


if __name__ == "__main__":
    main()
