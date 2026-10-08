"""Fact scouting for distance: unsteered baselines across relations.

Installs need headroom (target NOT already top-1) plus reachability
(target rank 2-50, unambiguous single token). Battery beyond capitals:
more countries, authors/books, symbols/elements, dates. Reports
unsteered top + target rank per candidate; viable facts graduate to
install batteries (mirror then listing). Cheap: one forward each.
Usage: python3 research/scout_facts.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CANDS = [
    ("spain-cap", "The capital of Spain is", " Madrid"),
    ("china-cap", "The capital of China is", " Beijing"),
    ("italy-cap", "The capital of Italy is", " Rome"),
    ("japan-cap", "The capital of Japan is", " Tokyo"),
    ("france-cap", "The capital of France is", " Paris"),
    ("hamlet", "Shakespeare wrote the play", " Hamlet"),
    ("gold", "The chemical symbol for gold is", " Au"),
    ("water", "The chemical formula for water is", " H2O"),
    ("orwell", "George Orwell wrote the novel", " 1984"),
    ("relativity", "Einstein published relativity in", " 1905"),
    ("everest", "The tallest mountain on Earth is", " Everest"),
    ("nile", "The longest river in Africa is", " Nile"),
]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd
    _, tok = load7b()
    for name, prompt, target in CANDS:
        try:
            tids = tok(target, return_tensors="pt")["input_ids"][0].tolist()
        except Exception as e:  # noqa
            print(f"{name:12} target-encode-FAIL {e}", flush=True)
            continue
        lg, _ = fwd(prompt)
        top = int(lg.argmax())
        o = np.argsort(-lg)[:3]
        tr = [(int(t), int((lg > lg[t]).sum()) + 1) for t in tids]
        print(f"{name:12} top={tok.decode([top])!r} "
              f"target={target!r}{tids} ranks={tr} "
              f"top3={[(tok.decode([int(i)]), round(float(lg[int(i)]), 1)) for i in o]} "
              f"{'VIABLE' if any(r > 1 and r <= 50 for _, r in tr) else ''}",
              flush=True)


if __name__ == "__main__":
    main()
