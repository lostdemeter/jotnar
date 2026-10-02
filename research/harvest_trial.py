"""Harvest trial (teacher-scaffold branch): teachers propose, ours disposes.

Prompts both teachers (Qwen2-0.5B, SmolLM2-135M, greedy =
deterministic bytes) on ~16 Ptolemaic-domain subjects, mines OUR
triple patterns (capitalized-subj + was/verbs/of, mine_edges
recipe) from their continuations, then attests each candidate two
ways: (a) witness in OUR corpora (groki 3 articles + w20k wiki2
train pool -- shape must occur, support counted); (b) multi-teacher
agreement (both teachers state the same triple). Novelty checked
against mined202 + w103m5 banks. Decision rule (stated, not a
gate): full harvest justified iff attested-novel >= 20 with junk
minority. Read-only (no weights change). Writes
research/harvest_trial.json. Usage: python3 research/harvest_trial.py
"""
import glob
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict
from html.parser import HTMLParser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


VERBS = {"defeated", "conquered", "founded", "built", "ruled", "led",
         "commanded", "crossed", "reformed", "assassinated", "married",
         "divorced", "exiled", "crowned", "succeeded", "invaded",
         "besieged", "captured", "killed", "died", "born", "reigned",
         "visited", "met", "allied", "betrayed", "murdered", "wrote",
         "built", "destroyed", "restored", "appointed", "elected"}


def mine_triples(text):
    toks = re.findall(r"[A-Za-z']+", text)
    low = [t.lower() for t in toks]
    out = []
    for i in range(1, len(toks) - 1):
        w = low[i]
        if w in ("is", "was", "were", "are") and toks[i - 1][0].isupper():
            out.append((toks[i - 1].lower(), w, low[i + 1], "identity"))
        elif w in VERBS and toks[i - 1][0].isupper() and i + 1 < len(toks):
            out.append((toks[i - 1].lower(), w, low[i + 1], "action"))
    for i in range(len(toks) - 2):
        if low[i + 1] == "of" and toks[i][0].isupper():
            obj = low[i + 2]
            if (i + 3 < len(toks) and low[i + 2] not in ("the", "a", "an")
                    and toks[i + 3][0].islower()
                    and low[i + 3] not in ("the", "a", "an", "of", "in",
                                           "and", "to", "for", "with")):
                obj = low[i + 2] + " " + low[i + 3]
            out.append((toks[i].lower(), "of", obj, "possession"))
    return out


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    PROMPTS = ["Alexander the Great was", "Cleopatra met",
               "Mark Antony fought", "Julius Caesar crossed",
               "Ptolemy ruled", "The Roman empire expanded",
               "Macedon rose under", "Persia fell when",
               "The battle of Actium", "The senate in Rome",
               "Alexandria was founded by", "The Ptolemaic dynasty",
               "Hannibal crossed", "Augustus became",
               "The library of Alexandria", "Sparta fought"]
    gens = {}
    for mid in ("Qwen/Qwen2-0.5B", "HuggingFaceTB/SmolLM2-135M"):
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        net = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True, torch_dtype=torch.float32).eval()
        if torch.cuda.is_available():
            net = net.cuda()
        outs = []
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            ids = ids.cuda() if torch.cuda.is_available() else ids
            with torch.no_grad():
                gen = net.generate(ids, max_new_tokens=40, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(gen[0][ids.shape[1]:]))
        gens[mid] = outs
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # mine candidates per teacher
    cand = {}
    for mid, outs in gens.items():
        c = Counter()
        for t in outs:
            for e in mine_triples(t):
                c[e] += 1
        cand[mid] = c
    # attestation pool: groki + w20k-train sentences (shape witness)
    pool = []

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
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        pool.append(html.unescape(" ".join(t.p)).lower())
    import pandas as pd
    p = ("/home/thorin/.cache/huggingface/hub/datasets--wikitext/snapshots"
         "/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1"
         "/train-00000-of-00001.parquet")
    df = pd.read_parquet(p, columns=["text"])
    for t in [str(t) for t in df["text"].tolist()][:2000]:
        pool.append(html.unescape(t).lower())
    pool = "\n".join(pool)

    def witnessed(e):
        s, pr, o, _ = e
        return (s in pool and pr in pool and o.split()[0] in pool
                and f"{s} {pr} {o.split()[0]}" in pool)

    # existing bank contents
    have = set()
    for path in (os.path.join(DD, "mined_edges.jsonl"),
                 os.path.join(DD, "banks", "w103m5_edges.jsonl")):
        for i, l in enumerate(open(path)):
            if i == 0:
                continue
            try:
                r = json.loads(l)
                have.add((r["subj"], r["pred"], r["obj"]))
            except Exception:
                pass
    keys = set(cand["Qwen/Qwen2-0.5B"]) | set(cand["HuggingFaceTB/SmolLM2-135M"])
    rows = []
    for e in sorted(keys):
        wit = witnessed(e)
        agree = e in cand["Qwen/Qwen2-0.5B"] and e in cand["HuggingFaceTB/SmolLM2-135M"]
        novel = (e[0], e[1], e[2]) not in have
        rows.append({"triple": list(e), "witnessed": wit, "agree": agree, "novel": novel,
                     "qwen_n": cand["Qwen/Qwen2-0.5B"].get(e, 0),
                     "smol_n": cand["HuggingFaceTB/SmolLM2-135M"].get(e, 0)})
    att_novel = [r for r in rows if r["witnessed"] and r["novel"]]
    agree_novel = [r for r in rows if r["agree"] and r["novel"]]
    junk = [r for r in rows if not r["witnessed"] and not r["agree"]]
    print(f"candidates: {len(rows)} (qwen {len(cand['Qwen/Qwen2-0.5B'])}, "
          f"smol {len(cand['HuggingFaceTB/SmolLM2-135M'])})")
    print(f"attested+novel: {len(att_novel)}; agree+novel: {len(agree_novel)}; "
          f"junk(neither): {len(junk)}")
    for r in att_novel[:12]:
        print(f"  KEEP {r['triple']} wit={r['witnessed']} agree={r['agree']}")
    for r in junk[:6]:
        print(f"  JUNK {r['triple']}")
    json.dump({"n_cand": len(rows), "attested_novel": att_novel,
               "agree_novel": agree_novel, "junk_n": len(junk),
               "rows": rows,
               "gens": gens},
              open(os.path.join(ROOT, "research", "harvest_trial.json"), "w"), indent=2)
    ok = len(att_novel) >= 20 and len(junk) < len(rows) / 2
    print(f"harvest justified: {ok} (rule: attested-novel>=20 and junk minority)")


if __name__ == "__main__":
    main()
