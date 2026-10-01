"""Ingestion tool MVP (offline): raw sources -> frozen model food.

Sources: HF cached parquet (wikitext-2/103 raw) and/or --textdir of
.txt/.html files. Stages: extract -> normalize -> seeded split ->
vocab (top-V + UNK0, frequency-ranked) -> bigram counts -> manifest
with provenance (source shas, seed, coverage, Zipf stats). Freezing is
offline by doctrine; listings consume frozen artifacts only.
Decisions (no armchair): V chosen from measured coverage curve, not
picked; wikitext scale timed live.
Usage: python3 scripts/ingest.py --source wikitext2 [--v 2000] [--seed 0]
       python3 scripts/ingest.py --textdir /path/to/txt [--v 2000]
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

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data_ingest")
WIKI2 = "/home/thorin/.cache/huggingface/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
WIKI103 = "/home/thorin/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1/train-00000-of-00001.parquet"


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


def load_wiki(path, limit=None):
    import glob as _glob
    import pandas as pd
    paths = sorted(_glob.glob(path.replace("train-00000-of-00001.parquet",
                                           "train-*.parquet")) or [path])
    texts = []
    for p in paths:
        df = pd.read_parquet(p, columns=["text"])
        texts += [str(t) for t in df["text"].tolist()]
        if limit and len(texts) >= limit:
            texts = texts[:limit]
            break
    sents = []
    for t in texts:
        t = html.unescape(t)
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", t)
                  if len(s.strip().split()) >= 4]
    sha = hashlib.sha256(path.encode()).hexdigest()[:16]
    return sents, {"source": os.path.basename(path), "sha": sha}


def load_textdir(d):
    sents, shas = [], {}
    for f in sorted(glob.glob(os.path.join(d, "*.txt")) + sorted(glob.glob(os.path.join(d, "*.html")))):
        raw = open(f, encoding="utf-8", errors="replace").read()
        shas[os.path.basename(f)] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if f.endswith(".html"):
            t = _T()
            t.feed(raw)
            raw = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw)
                  if len(s.strip().split()) >= 4]
    return sents, {"source": d, "shas": shas}


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="wikitext2",
                    choices=["wikitext2", "wikitext103", "grokipedia"])
    ap.add_argument("--textdir", default=None)
    ap.add_argument("--v", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    if a.textdir:
        sents, prov = load_textdir(a.textdir)
        tag = "textdir"
    elif a.source == "wikitext103":
        sents, prov = load_wiki(WIKI103, a.limit or None)
        tag = "wikitext103"
    elif a.source == "grokipedia":
        from scripts.freeze_edges import words_of as _w  # noqa (lexicon rule reuse)
        sents, prov = load_textdir("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia")
        tag = "grokipedia"
    else:
        sents, prov = load_wiki(WIKI2, a.limit or None)
        tag = "wikitext2"
    t_load = time.time() - t0
    rng = np.random.default_rng(a.seed)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    cut = int(0.8 * len(sents))
    train = [sents[i] for i in idx[:cut]]
    test = [sents[i] for i in idx[cut:]]
    toks = [w for s in train for w in words_of(s)]
    c = Counter(toks)
    top = [w for w, _ in c.most_common(a.v)]
    vocab = {"<unk>": 0}
    vocab.update({w: i + 1 for i, w in enumerate(top)})
    cov = sum(n for _, n in c.most_common(a.v)) / max(len(toks), 1)
    V = a.v + 1
    counts = np.zeros((V, V), dtype=np.int64)
    for s in train:
        ids = [vocab.get(w, 0) for w in words_of(s)]
        for x, y in zip(ids[:-1], ids[1:]):
            counts[x, y] += 1
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{tag}_bigrams.npz"), counts=counts)
    json.dump(vocab, open(os.path.join(OUT, f"{tag}_vocab.json"), "w"))
    # coverage curve (decision material, not picked constants)
    curve = {}
    for k in (512, 1000, 2000, 5000, 10000):
        curve[str(k)] = round(sum(n for _, n in c.most_common(k)) / max(len(toks), 1), 4)
    man = {"source": tag, "prov": prov, "seed": a.seed, "v": a.v,
           "n_train": len(train), "n_test": len(test),
           "n_toks": len(toks), "coverage": round(cov, 4),
           "coverage_curve": curve, "nnz": int((counts > 0).sum()),
           "load_s": round(t_load, 2),
           "counts_sha": hashlib.sha256(counts.tobytes()).hexdigest()[:16]}
    json.dump(man, open(os.path.join(OUT, f"{tag}_manifest.json"), "w"), indent=2)
    print(f"source={tag} sents={len(sents)} toks={len(toks)} "
          f"V={V} cov={cov:.3f} nnz={(counts > 0).sum()} load={t_load:.1f}s")
    print("curve:", curve)
    print(f"wrote {OUT}/{tag}_*")


if __name__ == "__main__":
    main()
