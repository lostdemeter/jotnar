"""Specificity matrix: N gated installs x N prompts, who moves whom.

For each country fact: mine a late contrast direction (country vs mean
of others, L27) + early lexical key (L2 country token). Install f on
prompt p with lex-gating; record target rank, top, margin, hold/move.
Answers: diagonal installs (all rank 1?), off-diagonal holds
(specificity?), collisions (f installs onto g?). First segment of the
minimum-information curve (aim vs store count).
Usage: python3 research/spec_matrix.py
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
ALPHA = 0.8


def main():
    from qwen_torch import _get as _gg
    torch, g, tok = _gg()
    prom = {c: f"The capital of {c} is" for c in COUNTRIES}
    traj = {c: fwdH(p) for c, p in prom.items()}
    # expected first tokens (verify singles, fall back to first piece)
    exp = {}
    for c in COUNTRIES:
        lg, _ = fwd(prom[c])
        exp[c] = int(lg.argmax())
    print("unsteered tops:", {c: tok.decode([exp[c]]) for c in COUNTRIES},
          flush=True)
    dirs, keys = {}, {}
    for c in COUNTRIES:
        others = [traj[o][0][27] for o in COUNTRIES if o != c]
        d = traj[c][0][27] - np.mean(others, axis=0)
        d /= np.linalg.norm(d)
        dirs[c] = d
    # keys: L2 country-token states
    for c in COUNTRIES:
        t, ids = fwdH(prom[c], keep="all")
        toks = tok.convert_ids_to_tokens(ids)
        frag = c[1:].lower()
        pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                   len(ids) - 1)
        v = t[2][pos]
        keys[c] = v / np.linalg.norm(v)
    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(prom[c])
        base[c] = int(lg.argmax())
    print(f"{'install/prompt':16} " + " ".join(f"{c[:4]:>12}"
          for c in COUNTRIES), flush=True)
    diag_ok, hold_ok, n = 0, 0, 0
    for f in COUNTRIES:
        row = []
        for p in COUNTRIES:
            # gate from L2 country row of the TARGET prompt
            tt, ids = fwdH(prom[p], keep="all")
            toks = tok.convert_ids_to_tokens(ids)
            frag = p[1:].lower()
            pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                       len(ids) - 1)
            m = float(tt[2][pos] @ keys[f] / (np.linalg.norm(tt[2][pos]) + 1e-12))
            s = m ** 6 if m > 0 else 0.0
            lg, _ = fwd(prom[p], steer=(27, dirs[f], ALPHA * 6 * s))
            top = int(lg.argmax())
            rk = int((lg > lg[exp[f]]).sum()) + 1
            hold = "HOLD" if top == base[p] else "moved"
            if f == p:
                cell = f"r{rk}"
                if rk == 1:
                    diag_ok += 1
            else:
                cell = "hold" if hold == "HOLD" else "XMOVED"
                if hold == "HOLD":
                    hold_ok += 1
            n += 1
            row.append(f"{cell:>12}")
        print(f"{f:16} " + " ".join(row), flush=True)
    print(f"diagonal rank-1: {diag_ok}/{len(COUNTRIES)}; "
          f"off-diagonal holds: {hold_ok}/{n - len(COUNTRIES)}", flush=True)


if __name__ == "__main__":
    main()
