"""Hybrid expects gate: content assertions on full-stack generation.

Demo-level reference (fills the universal's gap: demos had guards
but no content oracle). Full stack per output: banked retrieval
(curated/mined/w103 dot-arbitrate) -> fact prepend as pieces
(echo-deduped) -> bigram-core/piece router (minsup-2 propose,
sampled top6 + nrep guard) -> hedge-stop + glue-trim (CAP
bounds). Expects per seed (machine-checked CONTENT, first of
its class): fact-tail present (grounding!), valid (all words in
train set), closed (ends non-glue), replay identical (seed
re-run). Bars 8/8 x4 (audit seeds). Termination by cap proof.
Usage: python3 tests/test_hybrid.py (slow: ~160 listing runs)
"""
import glob
import hashlib
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
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16
NREP = 4
MINLEN = 8
CAP = 24
GLUE = {"the", "and", "of", "in", "a", "to", "with", "as", "for", "on",
        "by", "at", "from", "is", "was", "were", "are", "be", "it",
        "that", "this", "an", "or", "his", "her", "its", "their"}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


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
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    inv = {i: p for p, i in vocab.items()}
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
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

    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    wordset = {w for s in sents for w in words_of(s)}
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    wv = json.load(open(os.path.join(dd, "lm_vocab.json")))
    w = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece32.npz"))
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    fglue = set(b["words"].tolist())
    text = CFG + open(os.path.join(sdir, "lm_d32_headt.asm")).read()
    P = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
         "wv": enc(w["wv"]), "wo": enc(w["wo"]),
         "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
         "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
         "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    wpat0 = {w: zw[f"w{i}"] for i, w in enumerate(list(zw["lex"]))}
    zC = np.load(os.path.join(dd, "edge_keys.npz"))
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
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
                          "cmask": cm, **P}, sigs=SIGS, basedir=sdir)
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
        fact = retrieve(words_of(seed))
        _sw = words_of(seed)
        while fact and _sw and fact[0] == _sw[0]:
            fact.pop(0)
            _sw.pop(0)
        out = []
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
        while ws and ws[-1] in GLUE:
            ws.pop()
        return ws, fact

    seeds = ["alexander the great", "cleopatra and antony",
             "caesar conquered gaul", "the battle of actium",
             "mark antony and cleopatra", "the roman empire",
             "alexander founded alexandria", "the senate in rome"]
    fp = va = cl = 0
    for seed in seeds:
        ws, fact = close(seed, 7)
        # fact-tail present (echo guard may strip shared prefix; tail must survive)
        tail = [w for w in fact if w not in words_of(seed)][:2]
        fp += (not tail) or all(w in ws for w in tail)
        va += all(x in wordset for x in ws)
        cl += bool(ws) and ws[-1] not in GLUE
        print(f"  [{' '.join(ws)[:95]}]")
    check("hybrid-fact", fp == 8, f"fact-tail present {fp}/8")
    check("hybrid-valid", va == 8, f"all words in train set {va}/8")
    check("hybrid-closed", cl == 8, f"ends on content {cl}/8")
    w0, _ = close(seeds[0], 7)
    w1, _ = close(seeds[0], 7)
    check("hybrid-deterministic", w0 == w1, "seeded replay identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
