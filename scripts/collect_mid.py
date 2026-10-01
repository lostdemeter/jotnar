"""Collect MID statistics (offline counting): per-word mean MID rows.

First-8 tokens of 200 train sentences through the IvoQ body; MID1 rows
(layer-1 swiglu mid) labeled by NEXT word. Freezes data/mid_keys.npz
(Ub 32xK gains-raw + Vb Kx16 next-emb rows + key words + counts).
Counting only -- no gradients. Slow (~200 listing runs), run once.
Usage: python3 scripts/collect_mid.py [--nsent 200] [--topk 128]
"""
import glob
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
CFG = "CONFIG m_acc 35492\nCONFIG m_cov 35048\n"


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


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsent", type=int, default=200)
    ap.add_argument("--topk", type=int, default=128)
    a = ap.parse_args()
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
    train = [sents[i] for i in idx[:int(0.8 * len(sents))]][:a.nsent]
    vocab = json.load(open(os.path.join(DD, "lm_vocab.json")))
    d = np.load(os.path.join(DD, "lm_svd_IvoQ.npz"))
    E = np.load(os.path.join(DD, "lm_svd.npz"))["emb"]
    text = CFG + open(os.path.join(ROOT, "programs", "lm_depth2causal.asm")).read()
    sdir = os.path.join(ROOT, "programs")
    acc = defaultdict(list)
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
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
        mid = None
        for k in f:
            if k.endswith(".MID"):
                mid = dec(f[k])
                break
        if mid is None:
            raise RuntimeError(f"no MID stream: {sorted(f)}")
        for t in range(len(ids) - 1):
            acc[ids[t + 1]].append(mid[t])
            nrows += 1
        if (si + 1) % 50 == 0:
            print(f"  {si + 1}/{len(train)} sents, {nrows} rows", flush=True)
    cnt = Counter({w: len(v) for w, v in acc.items()})
    top = [w for w, _ in cnt.most_common(a.topk)]
    Ub = np.stack([np.mean(acc[w], axis=0) for w in top], axis=1)
    Vb = np.stack([E[w] for w in top], axis=0)
    np.savez(os.path.join(DD, "mid_keys.npz"), Ub=Ub, Vb=Vb,
             words=np.array(top))
    print(f"words kept {len(top)}/{len(acc)} mass "
          f"{sum(cnt[w] for w in top) / max(nrows, 1):.3f} of {nrows} rows")
    print(f"Ub {Ub.shape} maxabs {np.abs(Ub).max():.3f} | "
          f"Vb {Vb.shape} maxabs {np.abs(Vb).max():.3f}")
    print(f"wrote {DD}/mid_keys.npz")


if __name__ == "__main__":
    main()
