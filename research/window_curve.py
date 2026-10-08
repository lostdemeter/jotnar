"""Window curve L20-27: install-dose vs leak-dose per layer (natives).

Editability = window between install needs (margins) and leak onset
(cross-talk), both depth-dependent. L27: 1 vs 16 (wide). L23: 4 vs 4
(empty). This maps every candidate layer with layer-native contrasts
(mine AT L, place AT L -- transplants derail, proven): gains {1,4} x
Germany/Italy/Japan. Output: per-layer (Paris rank trajectory, hold
status) = the window table. Any layer with install at gain G and
holds through >>G is a new editable address; none found = L27 stands
alone with mechanism (thin margins + disciplined leakage).
Usage: python3 research/window_curve.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank
from siphon_battery import country_pos

COUNTRIES = ["Germany", "Italy", "Japan"]
TARGET = "Germany"
LAYERS = [20, 21, 22, 23, 24, 25, 26, 27]
GAINS = [1, 4]
KEY_SCALE = 8.0


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    _, tok = load7b()
    promp = {c: f"The capital of {c} is" for c in COUNTRIES}
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]

    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(promp[c])
        base[c] = int(lg.argmax())
    print("unsteered:", {c: tok.decode([v]) for c, v in base.items()}, flush=True)

    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    keys = {}
    for c in COUNTRIES:
        _, pos = country_pos(promp[c], c, tok)
        k, _ = early_key(promp[c], pos=pos)
        keys[c] = k
    D = 3584
    for L in LAYERS:
        d = tfr[L] - tde[L]
        d /= np.linalg.norm(d)
        stores = [{"key": keys[TARGET], "value": d, "dose": 1.0,
                   "tier": "assoc",
                   "support": f"L{L} native contrast"}]
        for c in COUNTRIES:
            if c != TARGET:
                stores.append({"key": keys[c], "value": np.zeros(D),
                               "tier": "null", "support": "background"})
        Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                  stores, key_scale=KEY_SCALE)
        for gn in GAINS:
            row = []
            for c in COUNTRIES:
                t7, _ = fwdH(promp[c], keep="all")
                _, pos = country_pos(promp[c], c, tok)
                xa = t7[2][pos]
                xa = xa / np.linalg.norm(xa)
                C = xa @ Ua
                P = np.exp(C - C.max())
                P /= P.sum()
                mag = float(np.linalg.norm(t7[L][-1]))
                y = (P @ Vc) * (gn * mag)
                yn = y / (np.linalg.norm(y) + 1e-12)
                lg, _ = fwd(promp[c], steer=(L, yn,
                                             float(np.linalg.norm(y) / mag)))
                top = int(lg.argmax())
                pr = int((lg > lg[paris]).sum()) + 1
                if c == TARGET:
                    verdict = "INSTALL" if top == paris else f"r{pr}"
                else:
                    verdict = "hold" if top == base[c] else "MOVED"
                row.append(f"{c[:4]}:{tok.decode([top])[:8]}^{pr}:{verdict}")
            print(f"L{L} gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
