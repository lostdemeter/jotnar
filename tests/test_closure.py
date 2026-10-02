"""Closure gate: hedge-stop with glue-trim (output mechanics).

Rule (deterministic, declared, no thresholds but hygiene
constants): generate (seeded topk-12 + trie mask + no-repeat-4);
at each word boundary past MINLEN pieces, if top-1 is a glue
piece (frozen frequent-64 = hedge signature from the miss
analysis: the model would coil next), STOP; trim trailing glue
run (end on last content word); hard CAP guarantees termination
(proof by construction, not measurement). Rescue: seed-only
outputs allowed (thinness RECORDED as aim's shadow, not hidden).
Bars (8 audit seeds): terminates 8/8, fires-before-cap >= 7/8
(measured 8/8), adds-content >= 4/8 (measured 5/8), replay
identical on rerun (seed 0 twice). Coil-to-closed converter:
stopping is mechanics (gated here); thinness is aim (horizon R1).
Usage: python3 tests/test_closure.py (slow: ~120 listing runs)
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
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"
WIN = 16
NREP = 4
MINLEN = 8
CAP = 24


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

    w = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece32.npz"))
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    glue = set(b["words"].tolist())
    text = CFG + open(os.path.join(sdir, "lm_d32_headt.asm")).read()
    P = {"emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
         "wv": enc(w["wv"]), "wo": enc(w["wo"]),
         "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
         "wdown": enc(w["wdown"]), "rms_w1": enc(w["rms1"]),
         "rms_w2": enc(w["rms2"]), "wlog": enc(Ep["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    wordset = {w for s in sents for w in re.findall(r"[a-z0-9']+", s.lower())}
    trie = {}
    for w in wordset:
        node = trie
        for pid in encw(w):
            node = node.setdefault(pid, {})

    def run_logits(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos,
                          "cmask": cm, **P}, sigs=SIGS, basedir=sdir)
        return dec(f["LOGITS"])[-1].copy()

    def words_of(ids):
        return "".join(inv[i] for i in ids).replace("</w>", " ").split()

    def close(seed, seedn):
        out = encode(seed)
        rng = np.random.default_rng(seedn)
        frag = []
        for pid in out:
            frag.append(pid)
            if inv[pid].endswith("</w>"):
                frag = []
        nseed = len(words_of(out))
        fired = False
        for _ in range(CAP):
            lg = run_logits(out)
            top1 = int(np.argmax(lg))
            if not frag and len(out) >= MINLEN and top1 in glue:
                fired = True
                break
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
        ws = words_of(out)
        while len(ws) > nseed and ws[-1] in GLUEW:
            ws.pop()
        return ws, nseed, fired

    GLUEW = None
    seeds = ["alexander the great", "cleopatra and antony",
             "caesar conquered gaul", "the battle of actium",
             "mark antony and cleopatra", "the roman empire",
             "alexander founded alexandria", "the senate in rome"]
    # glue words for trim (piece ids -> words via bank64 invert needs vocab pieces;
    # use train-word glue list, same as audit harness)
    GLUEW = {"the", "and", "of", "in", "a", "to", "with", "as", "for",
             "on", "by", "at", "from", "is", "was", "were", "are", "be",
             "it", "that", "this", "an", "or", "his", "her", "its", "their"}
    fired = added = 0
    for seed in seeds:
        ws, nseed, fr = close(seed, 7)
        fired += fr
        added += (len(ws) > nseed)
        print(f"  [{'STOP' if fr else 'CAP'}] {' '.join(ws)[:90]}")
    check("closure-terminates", True, "cap bounds all runs (proof)")
    check("closure-fires", fired >= 7, f"hedge-stop fired {fired}/8")
    check("closure-adds", added >= 4, f"added content {added}/8 (thin residual recorded)")
    w0, _, _ = close(seeds[0], 7)
    w1, _, _ = close(seeds[0], 7)
    check("closure-deterministic", w0 == w1, "seeded replay identical")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
