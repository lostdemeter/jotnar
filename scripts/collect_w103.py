"""Collect w103 HN statistics (offline counting): per-word mean HN rows.

Wikitext103 train sentences (seeded sample) through the w103 D32 stack
(lm_d32.asm + w103_d32 ends + D32 fit body + w103_d32bank); HN (layer-1)
and HN2 (layer-2) rows labeled by NEXT word. Freezes
data_ingest/w103_hn.npz (Ub1/Vb1 + Ub2/Vb2 + words). Counting only.
Slow (~200 listing runs D32), run once.
Usage: python3 scripts/collect_w103.py [--nsent 200] [--topk 256]
"""
import glob
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser  # noqa (sentences come from parquet)

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data_ingest")
CFG = "CONFIG m_acc 36849\nCONFIG m_cov 35686\nCONFIG beta -30.0\n"


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsent", type=int, default=200)
    ap.add_argument("--topk", type=int, default=256)
    a = ap.parse_args()
    import pandas as pd
    base = "/home/thorin/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
    paths = sorted(glob.glob(base + "/train-*.parquet"))
    texts = []
    for p in paths:
        df = pd.read_parquet(p, columns=["text"])
        texts += [str(t) for t in df["text"].tolist()]
    sents = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", html.unescape(t)):
            if len(s.strip().split()) >= 5:
                sents.append(s.strip())
    rng = np.random.default_rng(0)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    train = [sents[i] for i in idx[:a.nsent]]
    vocab = json.load(open(os.path.join(DD, "wikitext103_vocab.json")))
    E = np.load(os.path.join(DD, "w103_d32.npz"))
    d = np.load(os.path.join(ROOT, "data", "lm_d32_fit.npz"))
    b = np.load(os.path.join(DD, "w103_d32bank.npz"))
    B = {k: np.array(d[k]) for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown", "rms1", "rms2"]}
    text = CFG + open(os.path.join(ROOT, "programs", "lm_d32.asm")).read()
    sdir = os.path.join(ROOT, "programs")
    acc1, acc2 = defaultdict(list), defaultdict(list)
    nrows = 0
    for si, s in enumerate(train):
        ids = [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())][:8]
        if len(ids) < 2:
            continue
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(E["emb"]), "wq": enc(B["wq"]), "wk": enc(B["wk"]),
                          "wv": enc(B["wv"]), "wo": enc(B["wo"]),
                          "wup": enc(B["wup"]), "wgate": enc(B["wgate"]),
                          "wdown": enc(B["wdown"]),
                          "rms_w1": enc(B["rms1"]), "rms_w2": enc(B["rms2"]),
                          "wlog": enc(E["wlog"]),
                          "ukt": enc(b["ukt"]), "evb": enc(b["evb"])},
                         sigs=SIGS, basedir=sdir)
        h1, h2 = dec(f["HN"]), dec(f["HN2"])
        for t in range(len(ids) - 1):
            acc1[ids[t + 1]].append(h1[t])
            acc2[ids[t + 1]].append(h2[t])
            nrows += 1
        if (si + 1) % 50 == 0:
            print(f"  {si + 1}/{len(train)} sents, {nrows} rows", flush=True)
    cnt = Counter({w: len(v) for w, v in acc1.items()})
    top = [w for w, _ in cnt.most_common(a.topk)]
    Eemb = E["emb"]
    U1 = np.stack([np.mean(acc1[w], axis=0) for w in top], axis=1)
    U2 = np.stack([np.mean(acc2[w], axis=0) for w in top], axis=1)
    V = np.stack([Eemb[w] for w in top], axis=0)
    np.savez(os.path.join(DD, "w103_hn.npz"), U1=U1, U2=U2, Vb=V,
             words=np.array(top))
    print(f"words kept {len(top)}/{len(acc1)} mass "
          f"{sum(cnt[w] for w in top) / max(nrows, 1):.3f} of {nrows} rows")
    print(f"U1 {U1.shape} maxabs {np.abs(U1).max():.3f} | "
          f"U2 {U2.shape} maxabs {np.abs(U2).max():.3f}")
    print(f"wrote {DD}/w103_hn.npz")


if __name__ == "__main__":
    main()
