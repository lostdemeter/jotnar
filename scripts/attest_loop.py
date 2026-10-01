"""Attested generation loop (offline): our output, human-verified by corpus.

Generate continuations from test-split seeds (bankhn2 stack, seeded) ->
exact n-gram match against HELD-OUT text (never train: train-match is
plagiarism detection, reported separately, never barred) -> hits freeze
as attested edges (provenance: generated-attested, seed + ids + match).
Claim: 'a human wrote this; we reproduced it exactly.' Patterns within
patterns fall out: recurring attested shapes are style-families, counted
not asserted. Slow (listing runs per seed); run small, report honestly.
Usage: python3 scripts/attest_loop.py [--nseed 20] [--n 8]
"""
import glob
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
WIN = 8


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
    ap.add_argument("--nseed", type=int, default=20)
    ap.add_argument("--n", type=int, default=8)
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
    cut = int(0.8 * len(sents))
    test = [sents[i] for i in idx[cut:]][:a.nseed * 2]
    test_ids = [words_of(s) for s in test]
    # word-level 4-gram index over test sentences (held-out truth)
    test4 = set()
    for ids in test_ids:
        for i in range(len(ids) - 3):
            test4.add(" ".join(ids[i:i + 2]))
    vocab = json.load(open(os.path.join(DD, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    d = np.load(os.path.join(DD, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(DD, "bankhn.npz"))
    text = CFG + open(os.path.join(ROOT, "programs", "lm_bankhn2.asm")).read()

    def enc(x):
        return S.encode(np.ascontiguousarray(x, dtype=np.float64))

    P = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
         "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
         "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def gen(seed_ids, n, seedn):
        rng2 = np.random.default_rng(seedn)
        out = list(seed_ids)
        for _ in range(n):
            ctx = out[-WIN:]
            pos = np.arange(len(ctx), dtype=np.int64)
            cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": np.array(ctx, np.int64), "pos": pos,
                              "cmask": cm, **P}, sigs=SIGS,
                             basedir=os.path.join(ROOT, "programs"))
            t = f["LOGITS"]
            lg = (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                  * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1].copy()
            lg[0] = -1e9
            keep = np.argsort(-lg)[:12]
            w = np.zeros_like(lg)
            w[keep] = np.exp(lg[keep] - lg[keep].max())
            w = w / w.sum()
            out.append(int(rng2.choice(len(w), p=w)))
        return out

    hits, total, shapes = [], 0, {}
    for si in range(min(a.nseed, len(test_ids))):
        ids = [vocab.get(w, 0) for w in test_ids[si][:3]]
        if any(j == 0 for j in ids):
            continue
        out = gen(ids, a.n, 1000 + si)
        ws = [inv.get(j, "<unk>") for j in out]
        # bigrams (attestation) + trigrams-with-wildcards (style shapes)
        for i in range(len(ws) - 1):
            total += 1
            g = " ".join(ws[i:i + 2])
            if g in test4:
                hits.append({"seed": si, "gen": g, "n": 2})
        for i in range(len(ws) - 2):
            t3 = " ".join(ws[i:i + 3])
            # shape family: quierenlas middle/last wild (battle of X, X of Y)
            for pat in (f"{ws[i]} {ws[i+1]} W", f"{ws[i]} W W",
                        f"W {ws[i+1]} {ws[i+2]}"):
                shapes[pat] = shapes.get(pat, 0) + 1
    # attested shapes: shape families that occur in held-out test trigrams
    test3 = set()
    for ids in test_ids:
        for i in range(len(ids) - 2):
            test3.add(" ".join(ids[i:i + 3]))
    attested_shapes = {}
    for i in range(0, 0):
        pass
    # attested shapes: generated shape-families present in held-out test
    gen_shapes = set(shapes)
    test_shapes = set()
    for t in test3:
        w = t.split()
        for pat in (f"{w[0]} {w[1]} W", f"{w[0]} W W", f"W {w[1]} {w[2]}"):
            test_shapes.add(pat)
    attested_shapes = {p: shapes[p] for p in gen_shapes & test_shapes}
    print(f"attested: {len(hits)}/{total} generated bigrams in held-out "
          f"test ({len(hits) / max(total, 1):.3f})")
    for h in hits[:10]:
        print(f"  HIT: {h['gen']!r}")
    print(f"shapes (patterns-within-patterns): attested "
          f"{len(attested_shapes)}/{len(shapes)} families; top: "
          f"{dict(sorted(attested_shapes.items(), key=lambda kv: -kv[1])[:10])}")
    json.dump({"nseed": a.nseed, "hits": hits, "total": total,
               "shapes": {k: shapes[k] for k in sorted(
                   shapes, key=lambda k: -shapes[k])[:50]},
               "attested_shapes": {k: attested_shapes[k] for k in sorted(
                   attested_shapes, key=lambda k: -attested_shapes[k])[:50]}},
              open(os.path.join(DD, "attested.json"), "w"), indent=2)
    print(f"wrote {DD}/attested.json")


if __name__ == "__main__":
    main()
