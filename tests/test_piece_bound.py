"""Boundary-modeling gate: word-aware piece selection (host decoding rule).

Trie over train-word piece paths; generation masks logits to trie
children (word-starts at boundaries). Gates: zero invented words
(all decoded output in train word-set -- baseline 0.2 fragment rate
without the mask), seeded replay identical, listings untouched
(parity suites cover the stack; this gate covers the rule).
Usage: python3 tests/test_piece_bound.py (fast: ~30 listing runs)
"""
import glob
import html
import json
import os
import re
import sys
from collections import Counter
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16


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


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    wordset = {w for s in sents for w in re.findall(r"[a-z0-9']+", s.lower())}
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encw(w):
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
        return [vocab[p] for p in syms]

    trie = {}
    for w in wordset:
        node = trie
        for pid in encw(w):
            node = node.setdefault(pid, {})
    check("bound-trie-covers", len(trie) > 20,
          f"{len(trie)} word-start pieces (trie built)")
    w = np.load(os.path.join(dd, "lm_piece_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece.npz"))
    b = np.load(os.path.join(dd, "bankpiece64.npz"))
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()
    P = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
         "wv": enc(w["wv"]), "wo": enc(w["wo"]),
         "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
         "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
         "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def generate(seed_words, n, seedn):
        out = [p for w in seed_words for p in encw(w)]
        frag = []
        for pid in out:
            frag.append(pid)
            if inv[pid].endswith("</w>"):
                frag = []
        rng = np.random.default_rng(seedn)
        for _ in range(n):
            ctx = out[-WIN:]
            pos = np.arange(len(ctx), dtype=np.int64)
            cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": np.array(ctx, np.int64), "pos": pos,
                              "cmask": cm, **P}, sigs=SIGS, basedir=sdir)
            t = f["LOGITS"]
            lg = (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                  * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1].copy()
            node = trie
            for pid in frag:
                node = node.get(pid)
                if node is None:
                    break
            allowed = set(node) if node else set(trie)
            if allowed:
                mask = np.ones_like(lg, dtype=bool)
                mask[list(allowed)] = False
                lg[mask] = -1e9
            keep = np.argsort(-lg)[:12]
            wt = np.zeros_like(lg)
            wt[keep] = np.exp(lg[keep] - lg[keep].max())
            wt = wt / wt.sum()
            nxt = int(rng.choice(len(wt), p=wt))
            out.append(nxt)
            frag.append(nxt)
            if inv[nxt].endswith("</w>"):
                frag = []
        t = "".join(inv[i] for i in out).replace("</w>", " ")
        return " ".join(t.split()).split()

    outs = [generate(["alexander", "the", "great"], 12, 7),
            generate(["cleopatra", "and", "antony"], 12, 3)]
    bad = [x for ws in outs for x in ws[3:] if x not in wordset]
    check("bound-no-fragments", not bad,
          f"{len(bad)} invented words (baseline 0.2 rate unmasked)")
    again = generate(["alexander", "the", "great"], 12, 7)
    check("bound-deterministic", again == outs[0], "seeded replay identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
