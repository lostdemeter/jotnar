"""Distance scout v2: viable facts beyond capitals (authors/dates/defs).

Installs need headroom (target rank 2-50, unambiguous single token
preferred; multi-token flagged for the generation track, not here).
Relations: authors/books, dates/events, definitions/symbols, places.
One forward each; viable graduate to the distance install battery.
Usage: python3 research/scout_v2.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CANDS = [
    ("tolkien", "Tolkien wrote The Lord of the", " Rings"),
    ("herbert", "Frank Herbert wrote the novel", " Dune"),
    ("austen", "Jane Austen wrote the novel", " Emma"),
    ("moon", "Apollo 11 landed on the Moon in", " 1969"),
    ("wall", "The Berlin Wall fell in", " 1989"),
    ("water", "Water is also known as", " H2O"),
    ("gold", "The chemical symbol for gold is", " Au"),
    ("pacific", "The largest ocean on Earth is the", " Pacific"),
    ("sahara", "The largest desert in the world is the", " Sahara"),
    ("darwin", "Darwin published On the Origin of Species in", " 1859"),
    ("curie", "Marie Curie discovered the element", " Radium"),
    ("tesla", "The SI unit of magnetic flux density is the", " Tesla"),
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
            print(f"{name:10} target-encode-FAIL {e}", flush=True)
            continue
        lg, _ = fwd(prompt)
        top = int(lg.argmax())
        o = np.argsort(-lg)[:3]
        tr = [(int(t), int((lg > lg[t]).sum()) + 1) for t in tids]
        multi = "MULTI" if len(tids) > 1 else ""
        viab = any(r > 1 and r <= 50 for _, r in tr) and len(tids) == 1
        print(f"{name:10} top={tok.decode([top])!r} "
              f"target={target!r}{tids} ranks={tr} "
              f"top3={[(tok.decode([int(i)]), round(float(lg[int(i)]), 1)) for i in o]} "
              f"{'VIABLE' if viab else multi}", flush=True)


if __name__ == "__main__":
    main()
