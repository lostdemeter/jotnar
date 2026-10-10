"""Toll vs displacement: is the toll off-manifold distance? (mirror-side)

Hamlet prompt, gains {0.25, 0.5, 1.0, 2.0} first-token readout value:
per gain, steered-vs-unsteered prompt-end state displacement +
top token + degenerate count (single forward each, no generation:
toll priced at the step, not the trajectory). Prediction under
test: toll/degradation monotonic in displacement (knock size), with
install living where displacement is moderate; transparent below,
collapse above. One table closes the mechanism to a number.
Usage: python3 research/toll_curve.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

PROMPT = "Shakespeare wrote the play"
GAINS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwd_full
    _, tok = load7b()
    g, _ = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    ham = tok(" Hamlet", return_tensors="pt")["input_ids"][0].tolist()
    v = Wlog[ham[0]] / np.linalg.norm(Wlog[ham[0]])
    lg0, h0, _ = fwd_full(PROMPT)
    n0 = float(np.linalg.norm(h0))
    for gn in GAINS:
        lg, h, _ = fwd_full(PROMPT, steer=(27, v, gn) if gn else None)
        disp = float(np.linalg.norm(h - h0) / (n0 + 1e-12))
        o = np.argsort(-lg)[:3]
        deg = 1 if "\ufffd" in tok.decode([int(o[0])]) else 0
        print(f"gain={gn}: displ={disp:.3f} top3="
              f"{[(tok.decode([int(i)]), round(float(lg[int(i)]), 1)) for i in o]} "
              f"Ham-r={int((lg > lg[ham[0]]).sum()) + 1}", flush=True)


if __name__ == "__main__":
    main()
