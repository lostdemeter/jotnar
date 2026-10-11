"""All-bands-masked probe: does k=8 compound across 28 layers?

Same manual torch loop as attn_topk_decide (per-group top-k union,
k=8) but masking ALL 28 layers at once -- what the listing actually
does (the phase-0.5 probe masked one band at a time and never tested
compounding). Positions: final + every 32nd token of the 6 audit runs.
Metrics: maxabs/meanabs vs unmasked reference + flips.

Reading: maxabs ~= 6-7 with few flips => compounding explains the
listing gap (fix: bigger k / band map). maxabs ~= 0.3 clean =>
listing bug (bisect layers next).
Usage: python3 research/attn_allbands.py   (needs snapshot + GPU)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from research.attn_sparsity import PROMPTS, S_TARGETS, load_wiki_text
from research.attn_topk_decide import run_prefix


def main():
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
    rot = model.model.rotary_emb
    flips, n, worst = 0, 0, 0.0
    with torch.no_grad():
        for tag, ids in runs:
            pos = list(range(31, len(ids), 32)) + [len(ids) - 1]
            for i in pos:
                pre = ids[:i + 1]
                ref = run_prefix(model, rot, pre)
                m = run_prefix(model, rot, pre,
                               mask_band=set(range(28)), k=8)
                d = np.abs(m - ref)
                worst = max(worst, float(d.max()))
                n += 1
                if m.argmax() != ref.argmax():
                    flips += 1
                order = np.argsort(-ref)
                rk = int((order == m.argmax()).argmax()) + 1
                print(f"{tag}:{i}: maxabs={d.max():.3f} "
                      f"meanabs={d.mean():.5f} flip={m.argmax() != ref.argmax()} "
                      f"rank={rk}", flush=True)
    print(f"ALL-BANDS k=8: flips={flips}/{n} worst-maxabs={worst:.3f}")


if __name__ == "__main__":
    main()
