"""Attention participation audit: sparse routing vs kernel fusion.

Reads HF Qwen2-7B-Instruct attentions (output_attentions=True) over
prefixes S = 16/64/128/256 (wikitext) + two hand prompts, and measures
per query row the participation ratio PR = 1/sum(p^2) (effective count
attended) and top1/top3 mass. Aggregate per layer = median over heads
of mean over late query positions (i >= S/2; early rows are trivially
sparse by causality).

Pre-registered bar: median PR over deep-half layers (15..27) <= 3.0
AND deep top3 mass >= 0.95  =>  sparse geometric routing is justified
(the distribution is ~1-3 dimensional; dense compute proves emptiness).
Else (PR grows with S, mass diffuse)  =>  the future is kernel fusion
(SDPA/fused GEMMs), and routing stays soft/dense.

Also reports S-scaling: PR(S) flat while S grows 16x => sparsity wins
grow with context (the KV-cache story, attention-side).

Usage: python3 research/attn_sparsity.py   (needs snapshot + GPU)
Writes /tmp/attn_sparse.json; prints the verdict line.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

PROMPTS = {
    "capitals": "The capital of France is Paris. It is the most populous city in",
    "tourist": ("The capital of France is Paris. It is the most populous city in "
                "France and a global center for art, fashion and culture. Its "
                "19th-century cityscape is crisscrossed by wide boulevards and "
                "the River Seine. Beyond such landmarks as the Eiffel Tower and "
                "the 12th-century Gothic Notre-Dame cathedral, the city is known "
                "for its cafe culture and designer boutiques along the Rue du "
                "Faubourg Saint-Honore. The most visited tourist attractions."),
}
WIKI_FALLBACK = ("The history of artificial intelligence spans decades of research "
                 "into reasoning, perception and learning. Early programs manipulated "
                 "symbols through hand-written rules, while modern systems learn "
                 "representations from data. Neural networks, inspired by biological "
                 "neurons, stack layers of simple units into deep architectures. "
                 "Training adjusts millions of parameters by gradient descent on "
                 "large datasets. Transformers replaced recurrence with attention, "
                 "allowing models to weigh distant context directly. Scaling laws "
                 "describe how performance improves with model size, data and "
                 "compute. Deployment raises questions of safety, alignment and "
                 "access. Researchers probe internal mechanisms to understand what "
                 "these models represent and how they generalize to new tasks. " * 4)
S_TARGETS = (16, 64, 128, 256)
DEEP_FROM = 15


def load_wiki_text():
    for cand in ("data/wikitext103_vocab.json",):
        _ = cand
    try:
        sys.path.insert(0, os.environ.get("PHI_CORE_DIR", "/tmp"))
        import glob
        raws = sorted(glob.glob(os.path.join(ROOT, "data", "*.txt")))
        raws += sorted(glob.glob(os.path.join(ROOT, "samples", "*.txt")))
        for r in raws:
            with open(r) as f:
                t = f.read().strip()
            if len(t.split()) > 300:
                return t
    except Exception:
        pass
    return WIKI_FALLBACK


def stats_for(attn_tuple, s):
    """attn_tuple: 28 x (1,H,S,S) numpy. Returns per-layer dicts."""
    layers = []
    for L, a in enumerate(attn_tuple):
        a = a[0]  # (H,S,S)
        pr_h, t1_h, t3_h = [], [], []
        for h in range(a.shape[0]):
            pr_p, t1_p, t3_p = [], [], []
            for i in range(s // 2, s):
                p = a[h, i, :i + 1].astype(np.float64)
                p = p / p.sum()
                pr_p.append(float(1.0 / (p * p).sum()))
                ps = np.sort(p)[::-1]
                t1_p.append(float(ps[0]))
                t3_p.append(float(ps[:3].sum()))
            pr_h.append(float(np.mean(pr_p)))
            t1_h.append(float(np.mean(t1_p)))
            t3_h.append(float(np.mean(t3_p)))
        layers.append({"pr_med": float(np.median(pr_h)),
                       "pr_max": float(np.max(pr_h)),
                       "top1_med": float(np.median(t1_h)),
                       "top3_med": float(np.median(t3_h))})
    return layers


def main():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()
    _, tok = load7b()
    wiki = load_wiki_text()
    runs = []
    for name, text in [("wiki", wiki)] + sorted(PROMPTS.items()):
        ids = tok(text, return_tensors="pt")["input_ids"][0]
        for s in (S_TARGETS if name == "wiki" else (len(ids),)):
            s = min(s, len(ids))
            if s < 8:
                continue
            runs.append((f"{name}@{s}", ids[:s].tolist()))
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False,
        attn_implementation="eager").to("cuda").eval()
    out = {}
    with torch.no_grad():
        for tag, ids in runs:
            x = torch.tensor([ids]).to("cuda")
            a = model(x, output_attentions=True).attentions
            tup = tuple(t[0].float().cpu().numpy()[None] for t in a)
            del a
            torch.cuda.empty_cache()
            out[tag] = stats_for(tup, len(ids))
            print(f"{tag}: layers={len(out[tag])}", flush=True)
    del model
    torch.cuda.empty_cache()
    with open("/tmp/attn_sparse.json", "w") as f:
        json.dump(out, f)
    print("=== participation ratio (median over heads, late positions) ===")
    deep_pr, deep_t3, deep_t1 = [], [], []
    for tag, layers in out.items():
        prs = [l["pr_med"] for l in layers]
        line = " ".join(f"{v:5.2f}" for v in prs[::4])
        print(f"{tag:12s} L0,4,..: {line}  deep-med={np.median(prs[DEEP_FROM:]):.2f}")
        if tag.startswith("wiki"):
            deep_pr.append(float(np.median(prs[DEEP_FROM:])))
            deep_t3.append(float(np.median([l["top3_med"] for l in layers[DEEP_FROM:]])))
            deep_t1.append(float(np.median([l["top1_med"] for l in layers[DEEP_FROM:]])))
    verdict = (float(np.median(deep_pr)) <= 3.0
               and float(np.median(deep_t3)) >= 0.95)
    print(f"deep PR={np.median(deep_pr):.2f} (bar <=3.0) "
          f"top3={np.median(deep_t3):.3f} (bar >=0.95) "
          f"top1={np.median(deep_t1):.3f}")
    print("VERDICT:", "SPARSE-ROUTING" if verdict else "KERNEL-FUSION")


if __name__ == "__main__":
    main()
