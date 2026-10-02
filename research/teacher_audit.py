"""Assay A (teacher-scaffold branch): same-audit on teacher outputs.

Runs Qwen2-0.5B + SmolLM2-135M (local HF cache, offline) on the audit
harness's 8 seeds x 12 new tokens (topk-12 multinomial, torch-seeded:
deterministic bytes), scores the SAME failure modes (repeat / unk /
fragment / glue>baseline+0.15 / thin) plus two assay-only modes:
closure (terminal [.!?] in span -- our word ids cannot emit it,
structural zero) and open-validity (novel words checked against
groki+w103 wordset: real-but-new vs garbage). No guards on teachers
(native behavior is the question); no weights change anywhere
(read-only assay). Writes research/teacher_audit.json.
Usage: python3 research/teacher_audit.py (needs torch+transformers+cache)
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")

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
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    try:
        w103 = json.load(open(os.path.join(ROOT, "data_ingest", "wikitext103_vocab.json")))
        open_vocab = set(wordset) | {w for w in w103 if w != "<unk>"}
    except Exception:
        open_vocab = set(wordset)
    seeds = ["alexander the great", "cleopatra and antony",
             "caesar conquered gaul", "the battle of actium",
             "mark antony and cleopatra", "the roman empire",
             "alexander founded alexandria", "the senate in rome"]
    out = {"base_glue": base_glue, "models": {}}
    for mid in ("Qwen/Qwen2-0.5B", "HuggingFaceTB/SmolLM2-135M"):
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        net = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True,
            torch_dtype=torch.float32).eval()
        if torch.cuda.is_available():
            net = net.cuda()
        modes = Counter()
        ex = {}
        for seed in seeds:
            torch.manual_seed(7)
            ids = tok(seed, return_tensors="pt").input_ids
            ids = ids.cuda() if torch.cuda.is_available() else ids
            gen = net.generate(ids, max_new_tokens=12, do_sample=True,
                               top_k=12, pad_token_id=tok.eos_token_id)
            txt = tok.decode(gen[0][ids.shape[1]:])
            ws = words_of(seed) + words_of(txt)
            tri = [tuple(ws[k:k + 3]) for k in range(len(ws) - 2)]
            if len(tri) != len(set(tri)):
                modes["repeat"] += 1
                ex.setdefault("repeat", []).append((seed, " ".join(ws)))
            if any(x not in wordset for x in ws):
                modes["fragment"] += 1
                nov = [x for x in ws if x not in wordset]
                real = [x for x in nov if x in open_vocab]
                ex.setdefault("fragment", []).append(
                    (seed, f"novel={len(nov)} open-valid={len(real)} " + " ".join(ws)))
            gr = sum(1 for x in ws if x in GLUE) / max(len(ws), 1)
            if gr > base_glue + 0.15:
                modes["glue"] += 1
                ex.setdefault("glue", []).append((seed, f"{gr:.2f} " + " ".join(ws)))
            content = [x for x in ws if x not in GLUE]
            if len(set(content)) < 3:
                modes["thin"] += 1
                ex.setdefault("thin", []).append((seed, " ".join(ws)))
            if re.search(r"[.!?]", txt):
                modes["closure"] += 1
                ex.setdefault("closure", []).append((seed, txt.strip()[:100]))
        n = len(seeds)
        print(f"== {mid} ({n} outputs, plain topk-12 sampled, seeded) ==")
        for m in ("repeat", "fragment", "glue", "thin", "closure"):
            c = modes.get(m, 0)
            print(f"  {m}: {c}/{n} = {c / n:.2f}")
            for seed, e in ex.get(m, [])[:2]:
                print(f"    e.g. [{seed}] {e[:110]}")
        out["models"][mid] = {"n": n,
                              "rates": {m: modes.get(m, 0) / n for m in
                                        ("repeat", "fragment", "glue", "thin", "closure")},
                              "worst": {m: v[:3] for m, v in ex.items()}}
    json.dump(out, open(os.path.join(ROOT, "research", "teacher_audit.json"), "w"), indent=2)
    print(f"wrote research/teacher_audit.json (ours for comparison: "
          f"data/audit.json + demos: flagship glue 0.60, hybrid loop-free)")


if __name__ == "__main__":
    main()
