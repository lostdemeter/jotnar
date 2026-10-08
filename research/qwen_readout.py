"""Qwen readout norm structure (statics, CPU): is the teacher flat?

Rung-2 question: Qwen-late accepts top-1 installs, our skewed head
blocks them at any dose. Measure lm_head row norms: content rows
(Paris/Berlin/Rome/Tokyo/Madrid/Beijing), norm distribution
(max/p99/median), top-20 rows decoded. Comparandum: ours
(w_rome 0.84 vs unk 6.98 = 8.3x skew). Flat teacher + fluent teacher
=> skew is OURS (counts-SVD construction), not intrinsic; and flat
alone doesn't explain glue (probe 2: our glue died flat) -- Qwen's
glue lives in alignment (d=3584 trained), ours in norms (d=16
counts). Redesign fork: grow alignments vs dual-head routing.
Usage: python3 research/qwen_readout.py (needs snapshot; CPU only)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def main():
    from chain.qwen7b import load7b, snapshot_ok
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    g, tok = load7b()
    W = np.asarray(g("lm_head.weight"), dtype=np.float64)
    print(f"wlog shape {W.shape}", flush=True)
    n = np.linalg.norm(W, axis=1)
    for t in [" Paris", " Berlin", " Rome", " Tokyo", " Madrid",
              " Beijing", "China", "Japan"]:
        ids = tok(t, return_tensors="pt")["input_ids"][0].tolist()
        print(f"{t}: ids={ids} norms="
              f"{[round(float(n[j]), 2) for j in ids]}", flush=True)
    o = np.argsort(-n)[:20]
    print("top20=" + str([(tok.decode([int(i)]).replace("\n", "\\n"),
                           round(float(n[int(i)]), 2)) for i in o]), flush=True)
    print(f"max={n.max():.2f} p99={np.quantile(n, 0.99):.2f} "
          f"median={np.median(n):.2f} min={n.min():.2f}", flush=True)


if __name__ == "__main__":
    main()
