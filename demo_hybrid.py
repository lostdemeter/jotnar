"""Demo: core+fallback hybrid (piece-carried context, bigram core, fact prepend).

Word retrieval (banked dot-arbitrate, demo_fusion pattern) finds the
fact; the FACT IS PREPENDED AS PIECES (no UNK class -- the word
loop's failure mode); generation carries piece context throughout
(D32 refit stack) with a deterministic router per word boundary:
bigram-core proposes when its best continuation is attested
(minsup-2 count floor, mining doctrine), else the piece model
steps (seeded topk + trie mask + no-repeat-4, all declared). All
listings untouched; validity via trie (zero inventions by rule).
Read for quality blend (fluent core + speakable facts), not top1:
fusion-proposer verdict stands (product/selection both lose to solo;
piece OOV accuracy 0/144 -- facts arrive via RETRIEVAL COPY, not
piece prediction). Run: python3 demo_hybrid.py [seed words...]
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

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16
NREP = 4


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


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

    def decode(ids):
        t = "".join(inv[i] for i in ids).replace("</w>", " ")
        return " ".join(t.split())

    # word core (bigram counts + 512 vocab, frozen)
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    wv = json.load(open(os.path.join(dd, "lm_vocab.json")))
    # piece stack (D32 refit)
    Ep = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    w = Ep
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    text = CFG + open(os.path.join(sdir, "lm_d32_headt.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
         "wv": enc(w["wv"]), "wo": enc(w["wo"]),
         "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
         "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
         "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
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

    # retrieval side (banked word-cue, demo_fusion pattern, lean)
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    wpat0 = {w: zw[f"w{i}"] for i, w in enumerate(list(zw["lex"]))}

    def ext(wpat, edges):
        for w in sorted({x for e in edges for ff in ("subj", "pred", "obj")
                         for x in words_of(e[ff])} - set(wpat)):
            h = int(hashlib.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
            wpat[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=64)
        return wpat

    mined = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]
    zM = np.load(os.path.join(dd, "mined_keys.npz"))
    w103e = [json.loads(l) for l in open(os.path.join(dd, "banks", "w103_edges.jsonl"))][1:]
    zW = np.load(os.path.join(dd, "banks", "w103_keys.npz"))
    wpatM = ext(dict(wpat0), mined)
    wpatW = ext(dict(wpat0), w103e)

    def cue_of(words, wpat):
        pats = [(k, wpat[w]) for k, w in enumerate(words) if w in wpat]
        if not pats:
            return None
        cue = np.sign(sum(np.roll(p, k) for k, p in pats)).astype(float)
        cue[cue == 0] = 1.0
        return cue

    def retrieve(qwords):
        cands = []
        c = cue_of(qwords, wpat0)
        if c is not None:
            z = np.load(os.path.join(dd, "edge_keys.npz"))
            recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
            ki = int(np.argmax(z["keys"] @ c))
            acc = 0
            for r in recs:
                if ki < acc + r["n_keys"]:
                    cands.append((float((z["keys"] @ c)[ki]),
                                  words_of(r["subj"]) + words_of(r["pred"]) + words_of(r["obj"]),
                                  f"curated/{r['id']}"))
                    break
                acc += r["n_keys"]
        c = cue_of(qwords, wpatM)
        if c is not None:
            s = zM["keys"] @ c
            ki = int(np.argmax(s))
            e = mined[ki] if ki < len(mined) else None
            if e is not None:
                cands.append((float(s[ki]),
                              words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"]),
                              f"mined/{e['id']}"))
        c = cue_of(qwords, wpatW)
        if c is not None:
            s = zW["keys"] @ c
            ki = int(np.argmax(s))
            e = w103e[ki] if ki < len(w103e) else None
            if e is not None:
                cands.append((float(s[ki]),
                              words_of(e["subj"]) + words_of(e["pred"]) + words_of(e["obj"]),
                              f"w103/{e['id']}"))
        if not cands:
            return [], "none"
        cands.sort(key=lambda t: -t[0])
        return cands[0][1], cands[0][2]

    # trie (groki + w103 words speakable; validity rule)
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    try:
        _w103 = json.load(open(os.path.join(root, "data_ingest", "wikitext103_vocab.json")))
        _extra = {w for w in _w103 if w != "<unk>"}
    except Exception:
        _extra = set()
    trie = {}
    for w in {w for s in sents for w in words_of(s)} | _extra:
        try:
            node = trie
            for pid in encw(w):
                node = node.setdefault(pid, {})
        except KeyError:
            continue

    words, n, i, topk, seedn = [], 20, 1, 12, 7
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
        else:
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great"
    fact, fsrc = retrieve(words_of(seed))
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
    n_core = n_piece = 0
    for _ in range(n):
        # router: word boundary + core-knows-a-word -> bigram propose; else piece step
        at_bound = not frag
        cur = decode(out).split()
        lastw = cur[-1] if cur else "<unk>"
        wid = wv.get(lastw, 0)
        prop = None
        if at_bound and wid:
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
        if prop is not None:
            try:
                step = encw(prop)
            except KeyError:
                prop = None
        if prop is None:
            lg = logits_of(out).copy()
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
            keep = np.argsort(-lg)[:topk]
            wt = np.zeros_like(lg)
            wt[keep] = np.exp(lg[keep] - lg[keep].max())
            wt = wt / wt.sum()
            step = [int(rng.choice(len(wt), p=wt))]
            n_piece += 1
        else:
            n_core += 1
        out.extend(step)
        for pid in step:
            frag.append(pid)
            if inv[pid].endswith("</w>"):
                frag = []
    for _ in range(4):
        if inv[out[-1]].endswith("</w>"):
            break
        lg = logits_of(out).copy()
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
        out.append(int(np.argmax(lg)))
    print(f"seed: {seed}")
    print(f"fact: {' '.join(fact)} [{fsrc}]")
    print(f"out : {decode(out)}")
    print(f"(core {n_core} + piece {n_piece} steps, fact prepended as pieces)")


if __name__ == "__main__":
    main()
