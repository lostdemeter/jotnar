"""Piece-fit gate: structural fitting in piece space (joint rule).

Frozen data/lm_piece_fit.npz (IvoQ body + wv I*1.0) + bankpiece64 +
head2 T=4 (lm_headt.asm, beta_b 0.25). Piece ends lm_piece.npz (SVD).
Gates: twin beats K128 transferred (0.566), piece top1 no-collapse
(>=0.05 vs 0.062; fragments remain -- boundary modeling is next).
Usage: python3 tests/test_lm_piecefit.py (slow: ~600 runs S16)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
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
    man = json.load(open(os.path.join(dd, "lm_piece_fit_manifest.json")))
    check("piecefit-manifest", man["objective"]["twin"]["+headt"] == 0.443,
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

    w = np.load(os.path.join(dd, "lm_piece_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece.npz"))
    b = np.load(os.path.join(dd, "bankpiece64.npz"))
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()

    def run(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                          "wdown": enc(w["wdown"]),
                          "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
                          "wlog": enc(Ep["wlog"]),
                          "ukt": enc(b["ukt"]), "evb": enc(b["evb"])},
                         sigs=SIGS, basedir=sdir)
        return dec(f["H4"]), dec(f["LOGITS"])

    fa, _ = run(encode("alexander founded alexandria"))
    fp, _ = run(encode("alexandria was founded by alexander"))
    td = float(np.linalg.norm(fa[-1] - fp[-1]) / max(np.linalg.norm(fa[-1]), 1e-12))
    check("piecefit-twins", td < 0.566, f"twin={td:.3f} (K128 0.566)")

    import glob
    import html
    from html.parser import HTMLParser

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
            _, lg = run(ids[max(0, k - WIN):k])
            t1 += (int(np.argmax(lg[-1])) == ids[k])
            tot += 1
    check("piecefit-top1", t1 / tot >= 0.05,
          f"top1={t1 / tot:.3f} (holds, no-collapse)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
