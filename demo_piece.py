"""Demo: piece-loop end-to-end (BPE through hidden states, no UNK class).

Seed words -> BPE piece-encode -> bankhn2 stack on piece ends (SVD
rank-16, bankpiece512, S=16 window) -> piece AR -> word-decode.
Stated: piece top1 ~0.06 class (2038 choices); words come out whole
(no UNK -- every id decodes). Read for speakability, not accuracy.
Run: python3 demo_piece.py [seed words...] [--n 20] [--topk 12 --seed 7]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in re.findall(r"[a-z0-9']+", s.lower()):
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
            out.extend(vocab[p] for p in syms)
        return out

    def decode(ids):
        t = "".join(inv[i] for i in ids).replace("</w>", " ")
        return " ".join(t.split())

    Ep = np.load(os.path.join(dd, "lm_piece.npz"))
    b = np.load(os.path.join(dd, "bankpiece64.npz"))
    d = np.load(os.path.join(dd, "lm_piece_fit.npz"))
    B = {k: np.array(d[k]) for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown", "rms1", "rms2"]}
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(Ep["emb"]), "wq": enc(B["wq"]), "wk": enc(B["wk"]),
         "wv": enc(B["wv"]), "wo": enc(B["wo"]),
         "wup": enc(B["wup"]), "wgate": enc(B["wgate"]),
         "wdown": enc(B["wdown"]), "rms_w1": enc(B["rms1"]),
         "rms_w2": enc(B["rms2"]), "wlog": enc(Ep["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def logits_of(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          **P}, sigs=SIGS, basedir=sdir)
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    words, n, i, topk, seedn, nrep, rho = [], 20, 1, 12, 7, 4, 1.3
    topic, strength = [], 1.5
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--topk" and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        elif a == "--topic" and i + 1 < len(sys.argv):
            topic = [w.strip().lower() for w in sys.argv[i + 1].split(",")]
            i += 2
        elif a == "--strength" and i + 1 < len(sys.argv):
            strength = float(sys.argv[i + 1])
            i += 2
        elif a == "--rho" and i + 1 < len(sys.argv):
            rho = float(sys.argv[i + 1])
            i += 2
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    # word-trie (boundary modeling): pieces must walk valid word paths;
    # at word boundaries only word-start pieces allowed (host decoding
    # rule -- listings untouched, same doctrine as UNK-mask/no-repeat)
    import glob as _glob, html as _html
    from html.parser import HTMLParser as _HP
    from collections import Counter as _C

    class _TT(_HP):
        def __init__(self):
            super().__init__()
            self.p = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            self.skip = tag in ("script", "style", "nav", "header",
                                "footer", "aside")

        def handle_endtag(self, tag):
            self.skip = False

        def handle_data(self, d):
            if not self.skip:
                self.p.append(d)

    _sents = []
    for _f in sorted(_glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        _t = _TT()
        _t.feed(open(_f, encoding="utf-8", errors="replace").read())
        _txt = _html.unescape(" ".join(_t.p))
        _sents += [_s.strip() for _s in re.split(r"(?<=[.!?])\s+", _txt)
                   if len(_s.strip().split()) >= 4]

    def _encw(_w):
        _syms = [c for c in _w] + ["</w>"]
        while len(_syms) > 1:
            _best = None
            for _i in range(len(_syms) - 1):
                _r = rank.get((_syms[_i], _syms[_i + 1]))
                if _r is not None and (_best is None or _r < _best[0]):
                    _best = (_r, _i)
            if _best is None:
                break
            _, _i = _best
            _syms = _syms[:_i] + [_syms[_i] + _syms[_i + 1]] + _syms[_i + 2:]
        return [vocab[_p] for _p in _syms]

    _trie = {}
    try:
        _w103 = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_ingest", "wikitext103_vocab.json")))
        _extra = {w for w in _w103 if w != "<unk>"}
    except Exception:
        _extra = set()
    for _w in {w for _s in _sents for w in re.findall(r"[a-z0-9']+", _s.lower())} | _extra:
        _node = _trie
        for _pid in _encw(_w):
            _node = _node.setdefault(_pid, {})
    out = encode(seed)
    _tids = {pid for _w in topic for pid in _encw(_w)}
    rng = np.random.default_rng(seedn)
    _seen1 = set(out)
    frag = []
    for _pid in out:
        frag.append(_pid)
        if inv[_pid].endswith("</w>"):
            frag = []
    for _ in range(n):
        lg = logits_of(out).copy()
        # boundary mask: walk trie with current fragment
        _node = _trie
        for _pid in frag:
            _node = _node.get(_pid)
            if _node is None:
                break
        _allowed = set(_node) if _node else set(_trie)
        if _allowed:
            _mask = np.ones_like(lg, dtype=bool)
            _mask[list(_allowed)] = False
            lg[_mask] = -1e9
        if len(out) >= nrep - 1:
            seen = {tuple(out[k:k + nrep]) for k in range(len(out) - nrep + 1)}
            prefix = tuple(out[-(nrep - 1):]) if nrep > 1 else ()
            for j in range(len(lg)):
                if prefix + (j,) in seen:
                    lg[j] = -1e9
        for _j in _tids:
            lg[_j] += strength
        keep = np.argsort(-lg)[:topk]
        w = np.zeros_like(lg)
        w[keep] = np.exp(lg[keep] - lg[keep].max())
        for _j in _seen1:
            w[_j] /= rho
        w = w / w.sum()
        _nxt = int(rng.choice(len(w), p=w))
        out.append(_nxt)
        _seen1.add(_nxt)
        frag.append(_nxt)
        if inv[_nxt].endswith("</w>"):
            frag = []
    for _ in range(4):
        if inv[out[-1]].endswith("</w>"):
            break
        lg = logits_of(out).copy()
        _node = _trie
        for _pid in frag:
            _node = _node.get(_pid)
            if _node is None:
                break
        _allowed = set(_node) if _node else set(_trie)
        if _allowed:
            _mask = np.ones_like(lg, dtype=bool)
            _mask[list(_allowed)] = False
            lg[_mask] = -1e9
        out.append(int(np.argmax(lg)))
    print("seed:", seed)
    print("out :", decode(out))
    print(f"({len(out)} pieces -> words, no UNK class)")


if __name__ == "__main__":
    main()
