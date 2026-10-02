"""Audit B (research one-shot): failure census over hybrid+closure outputs.

Same 8 audit seeds, same 5 modes + closure as scripts/audit_outputs
(top-1-anchored comparison: flagship word stack vs hybrid piece
stack with fact prepend + bigram router + hedge-stop/trim). Hybrid
gen: 20 steps max (closure usually stops earlier), seeded topk-12 +
trie + nrep-4 + minsup-2 bigram propose + glue-trim. Compares
against data/audit.json baseline (flagship era). The audit DRIVES
fixes: top mode first. Writes research/audit_hybrid.json.
Usage: python3 research/audit_hybrid.py (slow: ~160 listing runs)
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

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
SDIR = os.path.join(ROOT, "programs")
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16
NREP = 4
MINLEN = 8
CAP = 24
GLUE = {"the", "and", "of", "in", "a", "to", "with", "as", "for", "on",
        "by", "at", "from", "is", "was", "were", "are", "be", "it",
        "that", "this", "an", "or", "his", "her", "its", "their"}


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


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    train_words = [w for s in sents for w in words_of(s)]
    wordset = set(train_words)
    base_glue = sum(1 for w in train_words if w in GLUE) / max(len(train_words), 1)
    vocab = json.load(open(os.path.join(DD, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(DD, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in words_of(s):
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

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    counts = np.load(os.path.join(DD, "lm_bigrams.npz"))["counts"]
    wv = json.load(open(os.path.join(DD, "lm_vocab.json")))
    Ep = np.load(os.path.join(DD, "lm_piece32_fit.npz"))
    w = Ep
    b = np.load(os.path.join(DD, "bankp32_64.npz"))
    fglue = set(b["words"].tolist())
    text = CFG + open(os.path.join(SDIR, "lm_d32_headt.asm")).read()
    P = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
         "wv": enc(w["wv"]), "wo": enc(w["wo"]),
         "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
         "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
         "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    zw = np.load(os.path.join(DD, "edge_wordpats.npz"))
    wpat0 = {w: zw[f"w{i}"] for i, w in enumerate(list(zw["lex"]))}
    zC = np.load(os.path.join(DD, "edge_keys.npz"))
    recs = [json.loads(l) for l in open(os.path.join(DD, "edge_records.jsonl"))][1:]
    trie = {}
    for w in wordset:
        node = trie
        for pid in encw(w):
            node = node.setdefault(pid, {})

    def logits_of(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos,
                          "cmask": cm, **P}, sigs=SIGS, basedir=SDIR)
        t = f["LOGITS"]
        return dec(t)[-1].copy()

    def retrieve(qwords):
        pats = [(k, wpat0[w]) for k, w in enumerate(qwords) if w in wpat0]
        if not pats:
            return []
        cue = np.sign(sum(np.roll(p, k) for k, p in pats)).astype(float)
        cue[cue == 0] = 1.0
        ki = int(np.argmax(zC["keys"] @ cue))
        acc = 0
        for r in recs:
            if ki < acc + r["n_keys"]:
                return words_of(r["subj"]) + words_of(r["pred"]) + words_of(r["obj"])
            acc += r["n_keys"]
        return []

    def close(seed, seedn):
        out = []
        fact = retrieve(words_of(seed))
        _sw = words_of(seed)
        while fact and _sw and fact[0] == _sw[0]:
            fact.pop(0)
            _sw.pop(0)
        for w in fact:
            try:
                out.extend(encw(w))
            except KeyError:
                pass
        out.extend(encode(seed))
        rng = np.random.default_rng(seedn)
        frag = []
        for pid in out:
            frag.append(pid)
            if inv[pid].endswith("</w>"):
                frag = []
        nseed = len("".join(inv[i] for i in out).replace("</w>", " ").split())
        for _ in range(CAP):
            lg = logits_of(out)
            top1 = int(np.argmax(lg))
            cur = "".join(inv[i] for i in out).replace("</w>", " ").split()
            lastw = cur[-1] if cur else "<unk>"
            wid = wv.get(lastw, 0)
            prop = None
            if not frag and wid:
                prow = counts[wid].astype(np.float64)
                order = np.argsort(-prow[1:])[:6] + 1
                if prow[order[0]] >= 2:
                    wt = prow[order].astype(np.float64)
                    wt = wt / wt.sum()
                    bi = int(rng.choice(order, p=wt))
                    pw = [w for w, j in wv.items() if j == bi][0]
                    step = encw(pw)
                    trial = out + step
                    fresh = {tuple(trial[k:k + NREP])
                             for k in range(len(out) - (NREP - 1), len(trial) - NREP + 1)}
                    seen = {tuple(out[k:k + NREP]) for k in range(len(out) - NREP + 1)}
                    if not (fresh & seen):
                        prop = pw
            if not frag and len(out) >= MINLEN and top1 in fglue:
                break
            if prop is not None:
                out.extend(encw(prop))
                for pid in encw(prop):
                    frag.append(pid)
                    if inv[pid].endswith("</w>"):
                        frag = []
                continue
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
            if len(out) >= NREP - 1:
                seen = {tuple(out[k:k + NREP]) for k in range(len(out) - NREP + 1)}
                prefix = tuple(out[-(NREP - 1):]) if NREP > 1 else ()
                for j in range(len(lg)):
                    if prefix + (j,) in seen:
                        lg[j] = -1e9
            keep = np.argsort(-lg)[:12]
            wt = np.zeros_like(lg)
            wt[keep] = np.exp(lg[keep] - lg[keep].max())
            wt = wt / wt.sum()
            nxt = int(rng.choice(len(wt), p=wt))
            out.append(nxt)
            frag.append(nxt)
            if inv[nxt].endswith("</w>"):
                frag = []
        ws = "".join(inv[i] for i in out).replace("</w>", " ").split()
        while len(ws) > nseed and ws[-1] in GLUE:
            ws.pop()
        return ws

    seeds = ["alexander the great", "cleopatra and antony",
             "caesar conquered gaul", "the battle of actium",
             "mark antony and cleopatra", "the roman empire",
             "alexander founded alexandria", "the senate in rome"]
    modes = Counter()
    worst = {}
    for seed in seeds:
        ws = close(seed, 7)
        print(f"  [{seed}] {' '.join(ws)[:100]}", flush=True)
        tri = [tuple(ws[k:k + 3]) for k in range(len(ws) - 2)]
        if len(tri) != len(set(tri)):
            modes["repeat"] += 1
            worst.setdefault("repeat", []).append((seed, " ".join(ws)))
        if any(x not in wordset for x in ws):
            modes["fragment"] += 1
            worst.setdefault("fragment", []).append((seed, " ".join(ws)))
        gr = sum(1 for x in ws if x in GLUE) / max(len(ws), 1)
        if gr > base_glue + 0.15:
            modes["glue"] += 1
            worst.setdefault("glue", []).append((seed, f"{gr:.2f} " + " ".join(ws)))
        content = [x for x in ws if x not in GLUE]
        if len(set(content)) < 3:
            modes["thin"] += 1
            worst.setdefault("thin", []).append((seed, " ".join(ws)))
    n = len(seeds)
    print(f"audit-hybrid ({n} outputs): corpus glue baseline {base_glue:.2f}")
    for m, c in modes.most_common():
        print(f"  {m}: {c}/{n} = {c / n:.2f}")
    old = json.load(open(os.path.join(DD, "audit.json")))
    print(f"flagship-era baseline {old['rates']}")
    json.dump({"n": n, "rates": {m: modes.get(m, 0) / n for m in
                                 ("repeat", "fragment", "glue", "thin", "closure")},
               "worst": {m: v[:3] for m, v in worst.items()},
               "base_glue": base_glue},
              open(os.path.join(ROOT, "research", "audit_hybrid.json"), "w"), indent=2)
    print("wrote research/audit_hybrid.json")


if __name__ == "__main__":
    main()
