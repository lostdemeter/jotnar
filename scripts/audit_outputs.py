"""Output audit: failure modes counted over generations (offline).

Generates from the flagship (depth-4 word stack, blocked sampling) over
fixed seeds, classifies every output into failure modes (all counted,
frozen sets -- no judges, human or model):
- fragment (words outside train set; word models only)
- repeat (duplicate 3-grams within output)
- unk (UNK ids emitted)
- glue (function-word fraction vs corpus baseline band)
- thin (content-word overlap with seed/fact below floor)
Writes data/audit.json {rates, worst examples per mode}. The audit
DRIVES fixes: top mode first, re-run shows the delta. Slow (~100
listing runs); run on change, not per commit.
Usage: python3 scripts/audit_outputs.py [--nseed 8] [--n 12]
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

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
CFG = "CONFIG m_acc 36849\nCONFIG m_cov 35048\n"
WIN = 8
GLUE = {"the", "and", "of", "in", "a", "to", "with", "as", "for", "on",
        "by", "at", "from", "is", "was", "were", "are", "be", "it",
        "that", "this", "an", "or", "his", "her", "its", "their"}


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseed", type=int, default=8)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    train_words = [w for s in sents for w in re.findall(r"[a-z0-9']+", s.lower())]
    wordset = set(train_words)
    base_glue = sum(1 for w in train_words if w in GLUE) / max(len(train_words), 1)
    vocab = json.load(open(os.path.join(DD, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    d = np.load(os.path.join(DD, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(DD, "bankhn.npz"))
    text = CFG + open(os.path.join(ROOT, "programs", "lm_depth4.asm")).read()

    def enc(x):
        return S.encode(np.ascontiguousarray(x, dtype=np.float64))

    P = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
         "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
         "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    seeds = ["alexander the great", "cleopatra and antony",
             "caesar conquered gaul", "the battle of actium",
             "mark antony and cleopatra", "the roman empire",
             "alexander founded alexandria", "the senate in rome"][:a.nseed]
    modes = Counter()
    worst = {}
    outs = []
    rng = np.random.default_rng(7)
    for seed in seeds:
        out = [vocab.get(w, 0) for w in seed.split()]
        for _ in range(a.n):
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
            if len(out) >= 2:
                seen = {tuple(out[k:k + 3]) for k in range(len(out) - 2)}
                prefix = tuple(out[-2:])
                for j in range(len(lg)):
                    if prefix + (j,) in seen:
                        lg[j] = -1e9
            keep = np.argsort(-lg)[:12]
            w = np.zeros_like(lg)
            w[keep] = np.exp(lg[keep] - lg[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        ws = [inv.get(j, "<unk>") for j in out]
        outs.append((seed, ws))
        tri = [tuple(ws[k:k + 3]) for k in range(len(ws) - 2)]
        if len(tri) != len(set(tri)):
            modes["repeat"] += 1
            worst.setdefault("repeat", []).append((seed, " ".join(ws)))
        if any(j == 0 for j in out):
            modes["unk"] += 1
            worst.setdefault("unk", []).append((seed, " ".join(ws)))
        if any(x not in wordset and x != "<unk>" for x in ws):
            modes["fragment"] += 1
            worst.setdefault("fragment", []).append((seed, " ".join(ws)))
        gr = sum(1 for x in ws if x in GLUE) / max(len(ws), 1)
        if gr > base_glue + 0.15:
            modes["glue"] += 1
            worst.setdefault("glue", []).append((seed, f"{gr:.2f} " + " ".join(ws)))
        content = [x for x in ws if x not in GLUE and x != "<unk>"]
        if len(set(content)) < 3:
            modes["thin"] += 1
            worst.setdefault("thin", []).append((seed, " ".join(ws)))
    n = len(outs)
    print(f"audit ({n} outputs, depth-4 word stack): corpus glue baseline {base_glue:.2f}")
    for m, c in modes.most_common():
        print(f"  {m}: {c}/{n} = {c / n:.2f}")
        for seed, ex in worst[m][:2]:
            print(f"    e.g. [{seed}] {ex[:100]}")
    json.dump({"n": n, "rates": {m: c / n for m, c in modes.items()},
               "worst": {m: v[:3] for m, v in worst.items()},
               "base_glue": base_glue},
              open(os.path.join(DD, "audit.json"), "w"), indent=2)
    print(f"wrote {DD}/audit.json")


if __name__ == "__main__":
    main()
