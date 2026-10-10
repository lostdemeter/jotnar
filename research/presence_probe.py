"""Presence gate: absolute cosine (entity present?), not softmax share.

Threshold probe killed relative gating (chatter maxP 0.93+: softmax
shares can't separate open text from facts). Presence: does ANY row
match an install key above absolute cosine 0.95 (entity present)?
Facts: Italy row vs Italy key ~1.0 -> fire. Chatter: best row 0.93
-> skip. Same ball (austen + anti-null), gate on presence at gains
{0.05, 1.0}: facts must install, chatter must hold. If green: the
addressing rule is presence-then-routing (two stages), and the GT
op has its second jobspec (in-listing presence masks).
Usage: python3 research/presence_probe.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

PROMPTS = ["Jane Austen wrote the novel", "hello", "Cry havoc and let slip"]
PRESENCE = 0.95
KEY_SCALE = 8.0


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from siphon_battery import country_pos
    _, tok = load7b()
    g, _ = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    pa = PROMPTS[0]
    _, posa = country_pos(pa, "Austen", tok)
    ka, _ = early_key(pa, pos=posa)
    emma = tok(" Emma", return_tensors="pt")["input_ids"][0].tolist()[0]
    v = Wlog[emma] / np.linalg.norm(Wlog[emma])
    Ua, Vc, _ = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": ka, "value": v, "dose": 1.0, "tier": "assoc",
          "support": "austen"},
         {"key": -ka, "value": np.zeros(D), "tier": "null",
          "support": "bg"}],
        key_scale=KEY_SCALE)
    for p in PROMPTS:
        t7, gids = fwdH(p, keep="all")
        # presence: best row cosine vs install key (column 0)
        uk = Ua[:, 0] / np.linalg.norm(Ua[:, 0])
        cos = [float((t7[2][i] / np.linalg.norm(t7[2][i])) @ uk)
               for i in range(len(gids))]
        pres = max(cos)
        lg0, _ = fwd(p)
        base = int(lg0.argmax())
        for gn in (0.05, 1.0):
            fire = pres > PRESENCE
            xa = t7[2][-1]
            xa = xa / np.linalg.norm(xa)
            C = xa @ Ua
            P = np.exp(C - C.max())
            P /= P.sum()
            mag = float(np.linalg.norm(t7[27][-1]))
            y = (P @ Vc) * (gn * mag)
            yn = y / (np.linalg.norm(y) + 1e-12)
            lg, _ = fwd(p, steer=(27, yn, float(np.linalg.norm(y) / mag))
                        if fire else None)
            top = int(lg.argmax())
            print(f"{p[:22]!r} pres={pres:.3f} {'FIRE' if fire else 'skip'} "
                  f"gain={gn}: top={tok.decode([top])[:12]!r} "
                  f"(base {tok.decode([base])[:12]!r}) "
                  f"{'HOLD' if top == base else 'MOVED'}", flush=True)


if __name__ == "__main__":
    main()
