"""Margin + temperature-stability: is the top pick signal or noise?

Traditional reading of razor picks: (1) temperature rescaling should
not flip real decisions (Guo-style: scaling recalibrates confidence,
it doesn't change well-separated argmaxes); (2) the margin
distribution itself tells whether coin-flips are the norm (then the
honest product is a SET, like our retrieval ambiguity sets, not a
point). Host-side HF measurements, no builds.
Usage: python3 research/margin_stable.py [--prompts N]
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "Alexander the Great founded",
    "The Roman Empire fell in",
    "Water boils at",
    "The quick brown fox",
    "Once upon a time",
    "def fibonacci(n):",
    "The president of the United States is",
    "In the beginning",
    "Machine learning is",
    "The Eiffel Tower is located in",
]
TEMPS = [0.5, 0.7, 1.0, 1.5, 2.0]


def main():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()
    _, tok = load7b()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    margins, flips, tops = [], [], []
    with torch.no_grad():
        for pr in PROMPTS:
            lg = model(**tok(pr, return_tensors="pt").to("cuda")).logits
            lg = lg[0, -1].float().cpu().numpy()
            o = np.argsort(-lg)
            margins.append(float(lg[o[0]] - lg[o[1]]))
            tops.append(tok.decode([int(o[0])]))
            # temperature stability: same argmax at all temps?
            seq = [int((lg / t).argmax()) for t in TEMPS]
            flips.append(len(set(seq)) - 1)
    del model
    torch.cuda.empty_cache()
    print(f"{'prompt':40.40} {'margin':>7} {'top':>12} {'tempflips'}",
          flush=True)
    for pr, m, t, f in zip(PROMPTS, margins, tops, flips):
        print(f"{pr!r:40.40} {m:7.2f} {t!r:>12} {f}", flush=True)
    print(f"median margin: {np.median(margins):.2f}; "
          f"prompts with temp-unstable top: {sum(1 for f in flips if f)}/12",
          flush=True)
    print(f"margin<0.5 (coin-flip zone): "
          f"{sum(1 for m in margins if m < 0.5)}/12", flush=True)


if __name__ == "__main__":
    main()
