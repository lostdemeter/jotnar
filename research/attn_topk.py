"""Phase 0 sizing: top-k mass curves per layer + middle-band head spread.

Same 6 HF eager forwards as attn_sparsity.py (wiki prefixes 16/64/128/256
+ 2 hand prompts). Per layer (median over heads, late positions i>=S/2):
top-k mass for k in (1,2,4,8,16,32). Per head (middle band L12-20):
PR distribution (p10/med/p90/max) to decide per-head vs per-layer routing.

Pre-registered sizing rule: k(band) = smallest k with median top-k mass
>= 0.99 across the wiki runs. Granularity rule: per-head routing iff
middle-band head PR spread (p90-p10) > 3.0, else per-layer.

Usage: python3 research/attn_topk.py   (needs snapshot + GPU)
Writes /tmp/attn_topk.json; prints k-table + granularity verdict.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from research.attn_sparsity import PROMPTS, S_TARGETS, load_wiki_text

KS = (1, 2, 4, 8, 16, 32)
MID = range(12, 21)


def layer_topk(a):
    """a: (H,S,S) probs. Returns {k: median-over-heads(mean late-pos mass)}."""
    H, S, _ = a.shape
    out = {k: [] for k in KS}
    for h in range(H):
        acc = {k: [] for k in KS}
        for i in range(S // 2, S):
            p = a[h, i, :i + 1].astype(np.float64)
            p /= p.sum()
            ps = np.sort(p)[::-1]
            c = np.cumsum(ps)
            for k in KS:
                acc[k].append(float(c[min(k, len(c)) - 1]))
        for k in KS:
            out[k].append(float(np.mean(acc[k])))
    return {k: float(np.median(v)) for k, v in out.items()}


def head_pr(a):
    """a: (H,S,S) probs. Returns per-head mean PR over late positions."""
    H, S, _ = a.shape
    prs = []
    for h in range(H):
        v = []
        for i in range(S // 2, S):
            p = a[h, i, :i + 1].astype(np.float64)
            p /= p.sum()
            v.append(float(1.0 / (p * p).sum()))
        prs.append(float(np.mean(v)))
    return prs


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
    curves, mids = {}, {}
    with torch.no_grad():
        for tag, ids in runs:
            x = torch.tensor([ids]).to("cuda")
            a = model(x, output_attentions=True).attentions
            tup = [t[0].float().cpu().numpy() for t in a]
            del a
            torch.cuda.empty_cache()
            curves[tag] = [layer_topk(t) for t in tup]
            mids[tag] = {L: head_pr(tup[L]) for L in MID}
            print(f"{tag}: done", flush=True)
    del model
    torch.cuda.empty_cache()
    with open("/tmp/attn_topk.json", "w") as f:
        json.dump({"curves": curves, "mids": mids}, f)
    print("=== top-k mass (median heads, late pos; wiki runs) ===")
    for tag in sorted(curves):
        if not tag.startswith("wiki"):
            continue
        print(f"-- {tag}")
        for L in (0, 4, 8, 12, 16, 20, 24, 27):
            row = " ".join(f"k{k}={curves[tag][L][k]:.3f}" for k in KS)
            print(f"  L{L:02d}: {row}")
    print("=== k for >=0.99 mass (median over wiki runs, per layer) ===")
    ktab = {}
    for L in range(28):
        for k in KS:
            if all(curves[t][L][k] >= 0.99 for t in curves
                   if t.startswith("wiki")):
                ktab[L] = k
                break
        else:
            ktab[L] = ">32"
    print(" ".join(f"{L}:{ktab[L]}" for L in range(28)))
    print("=== middle-band head PR spread (wiki@256) ===")
    spr = []
    for L in MID:
        v = np.array(mids["wiki@256"][L])
        p10, med, p90 = np.percentile(v, [10, 50, 90])
        spr.append(p90 - p10)
        print(f"  L{L}: p10={p10:.2f} med={med:.2f} p90={p90:.2f} "
              f"max={v.max():.2f}")
    print(f"median spread={np.median(spr):.2f} (rule: >3.0 => per-head)")
    print("GRANULARITY:", "PER-HEAD" if np.median(spr) > 3.0 else "PER-LAYER")


if __name__ == "__main__":
    main()
