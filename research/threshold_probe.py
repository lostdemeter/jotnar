"""Threshold + micro probes (distance fallout, mirror-side).

Two findings to close: (1) chatter routes to installs at maxP
0.52-0.66 (open dialogue hijacked: every installed fact captures
hello). Mechanism test: gate the steer on maxP>0.9 (GT-gated value;
SELECT exists in listings, mirror first) -> chatter must hold while
installs (P~1.0) proceed. (2) Austen junk-tops at gains 1-2 with
rank improving (r26->r2): overshoot signature -> micro ladder
0.05/0.1/0.25/0.5 for the install (inverted-U predicts a window
below).
Usage: python3 research/threshold_probe.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

CHATTER = ["hello", "Cry havoc and let slip", "how are you doing today"]
KEY_SCALE = 8.0
THR = 0.9


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from distance import span_pos
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    # -- (1) chatter threshold: austen ball + gated steer ---------------
    pa, _, ta = ("Jane Austen wrote the novel", "Austen", " Emma")
    tid = tok(ta, return_tensors="pt")["input_ids"][0].tolist()[0]
    posa = span_pos(pa, "Austen", tok)
    ka, _ = early_key(pa, pos=posa)
    v = Wlog[tid] / np.linalg.norm(Wlog[tid])
    Ua, Vc, _ = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": ka, "value": v, "dose": 1.0, "tier": "assoc",
          "support": "austen"},
         {"key": -ka, "value": np.zeros(D), "tier": "null",
          "support": "bg"}],
        key_scale=KEY_SCALE)
    for c in CHATTER:
        t7, gids = fwdH(c, keep="all")
        xa = t7[2][-1]
        xa = xa / np.linalg.norm(xa)
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        mag = float(np.linalg.norm(t7[27][-1]))
        y = (P @ Vc) * (1.0 * mag)
        gated = float(P.max()) > THR
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(c, steer=(27, yn, float(np.linalg.norm(y) / mag))
                    if gated else None)
        lg0, _ = fwd(c)
        top, top0 = int(lg.argmax()), int(lg0.argmax())
        print(f"chatter-gated {c[:14]!r}: maxP={float(P.max()):.2f} "
              f"{'STEER' if gated else 'skip'} top={tok.decode([top])[:12]!r} "
              f"(base {tok.decode([top0])[:12]!r}) "
              f"{'HOLD' if top == top0 else 'MOVED'}", flush=True)
    # -- (2) austen micro ladder ----------------------------------------
    t7, _ = fwdH(pa, keep="all")
    xa = t7[2][posa]
    xa = xa / np.linalg.norm(xa)
    C = xa @ Ua
    P = np.exp(C - C.max())
    P /= P.sum()
    mag = float(np.linalg.norm(t7[27][-1]))
    for gn in (0.05, 0.1, 0.25, 0.5):
        y = (P @ Vc) * (gn * mag)
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(pa, steer=(27, yn, float(np.linalg.norm(y) / mag)))
        top = int(lg.argmax())
        print(f"austen-micro gain={gn}: top={tok.decode([top])[:12]!r} "
              f"Emma-r={int((lg > lg[tid]).sum()) + 1}", flush=True)


if __name__ == "__main__":
    main()
