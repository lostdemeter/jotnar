"""Routed-specialization gate: flat starts + sharp continuations (aim breakthrough).

Composition (deterministic, no thresholds): per position, word
boundary (observable pre-decision from context) routes the stack:
word-START -> flat stack (lm_piece32_fit body, QK-I0.2, beta_b
0.25: averaging, twins, function words); MID-WORD -> sharp stack
(same body, block QK head2-I0.5 (lm_piece32_induct.npz) + beta_b
2.0: nearest-neighbor fragment completion). Boundary rule from
discord analysis (mid-word: sharp-gains 14 vs flat-gains 1;
starts: 4 vs 5 -- specialization by position class).
Gates: routed twin < 0.75 (measured 0.665 == flat: twin contexts
are all word-starts); routed top1 == 46/566 exact (fit 33;
discord fit-only 1, routed-only 14, McNemar 9.6 p~0.002 -- first
SIGNIFICANT aim movement in the program); probe size pinned
566 (process: double-count artifact caught once, never again).
Twin holds + top1 wins = first joint win on aim. R1 horizon
lives here now (routed specialization, not operating points).
Usage: python3 tests/test_routed.py (slow: ~600 listing runs)
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
CFG_F = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
CFG_S = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 2.0\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


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
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    man = json.load(open(os.path.join(dd, "lm_piece32_induct_manifest.json")))
    check("routed-manifest", man["twin"] == 0.665, f"frozen (sha {man['sha']})")
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

    w = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece32.npz"))
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    iq = np.load(os.path.join(dd, "lm_piece32_induct.npz"))

    def mk(WQ, cfg):
        text = cfg + open(os.path.join(sdir, "lm_d32_headt.asm")).read()
        P = {"emb": enc(Ep["emb"]), "wq": enc(WQ), "wk": enc(WQ),
             "wv": enc(w["wv"]), "wo": enc(w["wo"]),
             "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
             "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
             "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
             "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

        def run(ids):
            toks = np.array(ids, dtype=np.int64)
            pos = np.arange(len(ids), dtype=np.int64)
            cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
            f = ASM.run_text(text, REGISTRY,
                             {"tok": toks, "pos": pos, "cmask": cm, **P},
                             sigs=SIGS, basedir=sdir)
            return dec(f["H4"]), dec(f["LOGITS"])

        return run

    run_flat = mk(np.eye(32) * 0.2, CFG_F)
    run_sharp = mk(iq["wq2"], CFG_S)
    # routed twin (contexts all word-starts -> flat path)
    fa, _ = run_flat(encode("alexander founded alexandria"))
    fp, _ = run_flat(encode("alexandria was founded by alexander"))
    td = float(np.linalg.norm(fa[-1] - fp[-1]) / max(np.linalg.norm(fa[-1]), 1e-12))
    check("routed-twin", td < 0.75, f"routed twin={td:.3f} (flat path)")

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
    test = [sents[i] for i in idx[int(0.8 * len(sents)):][:10]]
    t1 = tot = 0
    for s in test:
        ids = encode(s)
        for k in range(1, len(ids)):
            ctx = ids[max(0, k - 16):k]
            midword = not inv[ctx[-1]].endswith("</w>")
            _, lg = run_sharp(ctx) if midword else run_flat(ctx)
            t1 += (int(np.argmax(lg[-1])) == ids[k])
            tot += 1
    check("routed-probe-size", tot == 566, f"{tot} positions (pin)")
    check("routed-top1", t1 == 46, f"routed top1={t1}/{tot} (fit 33)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
