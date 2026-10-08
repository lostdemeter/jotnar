"""Mid-layer install ladder: does upstream injection amplify?

Same gated ball as siphon_ball.py (L27 pair-contrast value, L2 lexical
keys, nulls), but the gated add lands at L14 / L22 / L27 (baseline).
Transport map says L14 carries heavy country mass (3.17) and L22 is
the donor precedent; L26-27 ignore the country token. Question: does
14 layers of model machinery amplify a placed direction (top-1 at
LOWER dose than L27) or derail it (lin_map saturation)? Gates per
(layer, gain): Germany Paris rank + Italy/Japan holds + retrieval.
Live-mined every run.
Usage: python3 research/siphon_mid.py (needs 7B snapshot + GPU)
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
LAYERS = [14, 22, 27]
GAINS = [1, 2, 4]
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
    d = tfr[27] - tde[27]
    d /= np.linalg.norm(d)
    keys = {}
    for c in COUNTRIES:
        _, pos = country_pos(promp[c], c, tok)
        k, _ = early_key(promp[c], pos=pos)
        keys[c] = k
    D = 3584
    stores = [{"key": keys[TARGET], "value": d, "dose": 1.0,
               "tier": "assoc", "support": "L27 contrast (placed upstream)"}]
    for c in COUNTRIES:
        if c != TARGET:
            stores.append({"key": keys[c], "value": np.zeros(D),
                           "tier": "null", "support": "background"})
    Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                              stores, key_scale=KEY_SCALE)

    for L in LAYERS:
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
                ret = int(P.argmax())
                want = COUNTRIES.index(c)
                mark = "RETR" if ret == want else f"ret{ret}"
                if c == TARGET:
                    verdict = "INSTALL" if top == paris else f"Parisr{pr}"
                else:
                    verdict = "hold" if top == base[c] else "MOVED"
                row.append(f"{c[:4]}:{tok.decode([top])[:8]}^{pr}:{mark}:{verdict}")
            print(f"L{L} gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
