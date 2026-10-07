"""Composition sweep: all six installs live at once, who survives.

Same regime as dose_cal (vs-mean late directions, lex keys, torch
mirror): for each prompt, gate all six facts and add the SUM of
gated doses in one forward. Compares against alone-installs:
diagonals hold? new collisions? washout (shared norm budget) or
resonance (uninstalled attractors)? The catalog turn, pre-listing.
Usage: python3 research/compose_all.py
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
    print(f"{'prompt':10} {'alone':>14} {'composed':>14} {'delta'}", flush=True)
    for p in COUNTRIES:
        tt, ids = fwdH(prom[p], keep="all")
        toks = tok.convert_ids_to_tokens(ids)
        frag = p[1:].lower()
        pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
                   len(ids) - 1)
        # alone: only p's own install (reference from dose_cal regime)
        m0 = float(tt[2][pos] @ keys[p] / (np.linalg.norm(tt[2][pos]) + 1e-12))
        s0 = m0 ** 6 if m0 > 0 else 0.0
        lga, _ = fwd(prom[p], steer=(27, dirs[p], ALPHA * 6 * s0))
        ta = int(lga.argmax())
        # composed: all six gated doses summed
        tot = np.zeros(3584)
        gates = {}
        for f in COUNTRIES:
            m = float(tt[2][pos] @ keys[f] / (np.linalg.norm(tt[2][pos]) + 1e-12))
            s = m ** 6 if m > 0 else 0.0
            gates[f] = s
            tot += ALPHA * 6 * s * dirs[f]
        nrm = np.linalg.norm(tot)
        lgc, _ = fwd(prom[p], steer=(27, tot / max(nrm, 1e-12),
                                     ALPHA * 6 * (nrm / (ALPHA * 6))))
        tc = int(lgc.argmax())
        ra = int((lga > lga[exp[p]]).sum()) + 1
        rc = int((lgc > lgc[exp[p]]).sum()) + 1
        gl = ",".join("%s:%.2f" % (c[:2], gates[c]) for c in COUNTRIES)
        sa = "%r:r%d" % (tok.decode([ta])[:12], ra)
        sc = "%r:r%d" % (tok.decode([tc])[:12], rc)
        print("%-10s %-16s %-16s [%s]" % (p, sa, sc, gl), flush=True)


if __name__ == "__main__":
    main()
