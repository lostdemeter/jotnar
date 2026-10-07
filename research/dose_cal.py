"""Dose calibration per fact: minimal dose for rank-1 + margin, with controls.

Same regime as spec_matrix (vs-mean late directions, lex-gated keys,
torch mirror): gains x facts, every cell with target rank/margin/top
+ all five control tops. Turns the rank-3s into rank-1s (or prices
why not) and calibrates one dose per fact for composition.
Usage: python3 research/dose_cal.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH
from chain.qwen7b import load7b

COUNTRIES = ["France", "Germany", "Italy", "Spain", "Japan", "China"]
GAINS = [0.5, 0.8, 1.1]


def main():
    _, tok = load7b()
    prom = {c: f"The capital of {c} is" for c in COUNTRIES}
    traj = {c: fwdH(prom[c]) for c, p in prom.items()}
    exp = {}
    for c in COUNTRIES:
        lg, _ = fwd(prom[c])
        exp[c] = int(lg.argmax())
    print("targets:", {c: tok.decode([exp[c]]) for c in COUNTRIES}, flush=True)
    dirs, keys = {}, {}
    for c in COUNTRIES:
        others = [traj[o][0][27] for o in COUNTRIES if o != c]
        d = traj[c][0][27] - np.mean(others, axis=0)
        d /= np.linalg.norm(d)
        dirs[c] = d
    for c in COUNTRIES:
        t, ids = fwdH(prom[c], keep="all")
        toks = tok.convert_ids_to_tokens(ids)
        frag = c[1:].lower()
        pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                   len(ids) - 1)
        v = t[2][pos]
        keys[c] = v / np.linalg.norm(v)
    # trajectories cached once (gate rows reused across gains)
    ttraj = {}
    for c in COUNTRIES:
        t, ids = fwdH(prom[c], keep="all")
        toks = tok.convert_ids_to_tokens(ids)
        frag = c[1:].lower()
        pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                   len(ids) - 1)
        ttraj[c] = (t, pos)
    for f in COUNTRIES:
        for a in GAINS:
            cells = []
            for p in COUNTRIES:
                t, pos = ttraj[p]
                toks = tok.convert_ids_to_tokens(
                    tok(prom[p], return_tensors="pt")["input_ids"][0].numpy())
                m = float(t[2][pos] @ keys[f] / (np.linalg.norm(t[2][pos]) + 1e-12))
                s = m ** 6 if m > 0 else 0.0
                lg, _ = fwd(prom[p], steer=(27, dirs[f], a * 6 * s))
                top = int(lg.argmax())
                rk = int((lg > lg[exp[f]]).sum()) + 1
                mg = float(np.sort(lg)[-1] - np.sort(lg)[-2])
                mark = "R1" if (f == p and rk == 1) else (
                    "hold" if (f != p and top == exp[p]) else "X")
                cells.append(f"{p[:4]}:{tok.decode([top])[:8]}r{rk}"
                             f"{'+' if mark == 'R1' else ''}")
            print(f"{f:8} a={a}: " + " ".join(cells), flush=True)


if __name__ == "__main__":
    main()
