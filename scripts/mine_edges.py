"""Mine relation triples (offline, counted): sentences -> edges + probes.

Patterns (no models, no parsing library): BE (X is/was/are/were Y),
ACT (X VERBed Y with verb lexicon from voice pairs + edges), OF
(X of Y noun-noun). Every edge ships with its probe: (subj+pred ->
obj) question with the source sentence id. Support>=2 required
(singletons are noise -- stated); proper-noun subjects preferred
(capitalized in source). Deterministic (sorted, frequency-ranked).
Writes data/mined_edges.jsonl (echion-store/1 envelope) + manifest.
Usage: python3 scripts/mine_edges.py [--source grokipedia|wikitext2] [--minsup 2]
"""
import glob
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")
VERBS = {"conquered", "defeated", "founded", "crossed", "allied",
         "besieged", "reformed", "proscribed", "killed", "married",
         "divorced", "exiled", "crowned", "annexed", "invaded", "built",
         "destroyed", "ruled", "led", "won", "lost", "commanded",
         "assassinated", "murdered", "betrayed", "formed", "granted",
         "holds", "anchored", "died", "born"}


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


def load_groki():
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        raw = open(f, encoding="utf-8", errors="replace").read()
        t = _T()
        t.feed(raw)
        txt = html.unescape(" ".join(t.p))
        for s in re.split(r"(?<=[.!?])\s+", txt):
            if len(s.strip().split()) >= 4:
                sents.append(s.strip())
    return sents


def load_wiki2(n=None):
    import pandas as pd
    p = "/home/thorin/.cache/huggingface/hub/datasets--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/train-00000-of-00001.parquet"
    df = pd.read_parquet(p, columns=["text"])
    texts = [str(t) for t in df["text"].tolist()]
    if n:
        texts = texts[:n]
    sents = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", html.unescape(t)):
            if len(s.strip().split()) >= 4:
                sents.append(s.strip())
    return sents


def load_wiki103(n=None):
    import glob as _glob
    import pandas as pd
    base = "/home/thorin/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-raw-v1"
    paths = sorted(_glob.glob(base + "/train-*.parquet"))
    texts = []
    for p in paths:
        df = pd.read_parquet(p, columns=["text"])
        texts += [str(t) for t in df["text"].tolist()]
        if n and len(texts) >= n:
            texts = texts[:n]
            break
    sents = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", html.unescape(t)):
            if len(s.strip().split()) >= 4:
                sents.append(s.strip())
    return sents


def mine(sents):
    cand = Counter()
    wit = defaultdict(list)
    for si, s in enumerate(sents):
        toks = re.findall(r"[A-Za-z']+", s)
        low = [t.lower() for t in toks]
        for i in range(1, len(toks) - 1):
            w = low[i]
            if w in ("is", "was", "were", "are") and toks[i - 1][0].isupper():
                e = (toks[i - 1].lower(), w, low[i + 1], "identity")
                cand[e] += 1
                wit[e].append(si)
            elif w in VERBS and toks[i - 1][0].isupper() and i + 1 < len(toks):
                e = (toks[i - 1].lower(), w, low[i + 1], "action")
                cand[e] += 1
                wit[e].append(si)
        for i in range(len(toks) - 2):
            if low[i + 1] == "of" and toks[i][0].isupper():
                # object phrase: up to 2 tokens (named entities complete:
                # 'republic of ireland', not 'republic of i')
                obj = low[i + 2]
                if i + 3 < len(toks) and low[i + 2] not in (
                        "the", "a", "an") and toks[i + 3][0].islower():
                    obj = low[i + 2] + " " + low[i + 3]
                e = (toks[i].lower(), "of", obj, "possession")
                cand[e] += 1
                wit[e].append(si)
    return cand, wit


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="grokipedia")
    ap.add_argument("--minsup", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sents = (load_wiki103(a.limit or None) if a.source == "wikitext103"
             else load_wiki2(a.limit or None) if a.source == "wikitext2"
             else load_groki())
    cand, wit = mine(sents)
    # phrase-norm: strip trailing glue from OF-objs (re-concentrate:
    # 'alexandria in' -> 'alexandria', 'city at' dropped if empty)
    GLUE_TAIL = {"in", "and", "of", "our", "at", "on", "to", "for", "with"}
    normed = {}
    wit2 = defaultdict(list)
    for (s, p, o, c), n in cand.items():
        o2 = o
        if c == "possession":
            w = o.split()
            while len(w) > 1 and w[-1] in GLUE_TAIL:
                w = w[:-1]
            o2 = " ".join(w)
        normed[(s, p, o2, c)] = normed.get((s, p, o2, c), 0) + n
        wit2[(s, p, o2, c)] += wit[(s, p, o, c)]
    cand, wit = normed, wit2
    edges = []
    for (s, p, o, c), n in sorted(cand.items(), key=lambda kv: (-kv[1], kv[0])):
        if n < a.minsup:
            continue
        if len(s) < 2 or len(o) < 2:
            continue
        if o in ("the", "a", "an", "also", "one", "not", "that", "his",
                 "her", "its", "this", "then", "there"):
            continue
        if s in ("it", "he", "she", "they", "we", "this", "that", "there"):
            continue
        edges.append({"subj": s, "pred": p, "obj": o, "cat": c,
                      "support": n, "witness": wit[(s, p, o, c)][0]})
    for i, e in enumerate(edges):
        e["id"] = f"m{i:04d}"
        e["probe_q"] = f"{e['subj']} {e['pred']}"
        e["probe_a"] = e["obj"]
        e["digest"] = hashlib.sha256(
            json.dumps([e["subj"], e["pred"], e["obj"]]).encode()).hexdigest()[:16]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mined_edges.jsonl"), "w") as fh:
        fh.write(json.dumps({"type": "manifest", "format": "echion-store/1",
                             "n_records": len(edges), "source": a.source,
                             "minsup": a.minsup}) + "\n")
        for e in edges:
            fh.write(json.dumps(e) + "\n")
    json.dump({"source": a.source, "n_sents": len(sents),
               "n_cand": len(cand), "n_edges": len(edges),
               "minsup": a.minsup,
               "sha": hashlib.sha256(
                   json.dumps(edges).encode()).hexdigest()[:16]},
              open(os.path.join(OUT, "mined_manifest.json"), "w"), indent=2)
    print(f"sents={len(sents)} cand={len(cand)} edges={len(edges)} (minsup {a.minsup})")
    for e in edges[:10]:
        print(f"  {e['subj']} | {e['pred']} | {e['obj']} x{e['support']}")


if __name__ == "__main__":
    main()
