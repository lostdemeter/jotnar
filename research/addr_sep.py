"""Address separability probe (listing-design input, torch mirror).

For each prompt: per-row L2 residual match vs the three bank keys.
Questions: (a) does the country-token row separate (Germany high,
others low)? (b) does the PROMPT-END row separate (pure in-graph
addressing needs this; else the listing must SLICE the country row
and route explicitly)? Also reports prompt lengths (one binary needs
uniform geometry).
Usage: python3 research/addr_sep.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

COUNTRIES = ["Germany", "Italy", "Japan"]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwdH
    from siphon_ball import early_key
    _, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    keys = {c: early_key(promp[c])[0] for c in COUNTRIES}
    for c in COUNTRIES:
        t7, gids = fwdH(promp[c], keep="all")
        toks = tok.convert_ids_to_tokens(gids)
        n = len(gids)
        print(f"{c}: len={n} toks={[t.replace(chr(9646), '') for t in toks]}",
              flush=True)
        H2 = t7[2]
        for r in range(n):
            v = H2[r] / np.linalg.norm(H2[r])
            m = {k: round(float(v @ keys[k]), 3) for k in COUNTRIES}
            print(f"  row{r}: match={m}", flush=True)


if __name__ == "__main__":
    main()
