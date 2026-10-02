"""Piece32-refit gate: structural fitting at width (joint rule).

Frozen data/lm_piece32_fit.npz (piece32 ends + QK-I0.2 + V-I1.0 +
head2 T=4) + bankp32_64. Transfer baseline was word-D32-fit body in
piece space (twin 1.048 / top1 0.049): width-without-refit degrades,
same law as word-D64. Refit wins both axes at depth-2 (0.665/0.058);
V2.0 twin win rejected on top1 collapse (joint 13-for-13); depth-4
ACCEPTs as structure (0.448/0.044, cost within 0.018 bar).
Usage: python3 tests/test_lm_piece32fit.py (slow: ~700 runs S16)
"""
import glob
import html
import json
import os
import re
import sys
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG2 = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
CFG4 = "CONFIG m_acc 36849\nCONFIG m_cov 35686\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16


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
    man = json.load(open(os.path.join(dd, "lm_piece32_fit_manifest.json")))
    check("p32fit-manifest", man["objective"]["twin_depth2"]["+headt"] == 0.665,
          f"fit frozen (sha {man['sha']})")
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
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

    w = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    t2 = CFG2 + open(os.path.join(sdir, "lm_d32_headt.asm")).read()
    t4 = CFG4 + open(os.path.join(sdir, "lm_d32_d4headt.asm")).read()

    def run(ids, text, key):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(w["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                          "wdown": enc(w["wdown"]),
                          "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
                          "wlog": enc(w["wlog"]),
                          "ukt": enc(b["ukt"]), "evb": enc(b["evb"])},
                         sigs=SIGS, basedir=sdir)
        return dec(f[key]), dec(f["LOGITS"])

    fa, _ = run(encode("alexander founded alexandria"), t2, "H4")
    fp, _ = run(encode("alexandria was founded by alexander"), t2, "H4")
    td = float(np.linalg.norm(fa[-1] - fp[-1]) / max(np.linalg.norm(fa[-1]), 1e-12))
    check("p32fit-twins", td < 0.75, f"twin={td:.3f} (transfer 1.048)")

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
    test = [sents[i] for i in idx[int(0.8 * len(sents)):]][:10]
    t1 = tot = 0
    for s in test:
        ids = encode(s)
        for k in range(1, len(ids)):
            _, lg = run(ids[max(0, k - WIN):k], t2, "H4")
            t1 += (int(np.argmax(lg[-1])) == ids[k])
            tot += 1
    check("p32fit-top1", t1 / tot >= 0.05,
          f"top1={t1 / tot:.3f} (transfer 0.049, holds)")

    fa4, _ = run(encode("alexander founded alexandria"), t4, "H8")
    fp4, _ = run(encode("alexandria was founded by alexander"), t4, "H8")
    td4 = float(np.linalg.norm(fa4[-1] - fp4[-1]) / max(np.linalg.norm(fa4[-1]), 1e-12))
    check("p32fit-d4twins", td4 < 0.60, f"d4 twin={td4:.3f} (d2 0.665)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
