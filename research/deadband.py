"""Dead band + descent: bound the editable edge, price the churn.

(1) Descent: L18 {1,4} + L19 {2} (L19g1/L20g1 install, L20g2 derails;
where does the lower edge end, how wide is L19?).
(2) Churn profile: per-layer relative residual velocity
||x[L]-x[L-1]||/||x|| on the battery (prompt-end + country rows).
Hypothesis: L21-26 (dead band: moves, never installs) shows max
velocity = representations mid-formation churn placed vectors;
editable regimes (L19-20, L27) sit in low-velocity reaches
(pre-formation stability / post-formation margins).
Usage: python3 research/deadband.py (needs 7B snapshot + GPU)
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
KEY_SCALE = 8.0


def install_cell(promp, tok, tfr, tde, keys, Ua, Vc, paris, base, L, gn):
    from qwen_torch import fwd, fwdH
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
        lg, _ = fwd(promp[c], steer=(L, yn, float(np.linalg.norm(y) / mag)))
        top = int(lg.argmax())
        pr = int((lg > lg[paris]).sum()) + 1
        if c == "Germany":
            verdict = "INSTALL" if top == paris else f"r{pr}"
        else:
            verdict = "hold" if top == base[c] else "MOVED"
        row.append(f"{c[:4]}:{tok.decode([top])[:8]}^{pr}:{verdict}")
    return row


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
    # (1) descent
    for L, gains in ((18, [1, 4]), (19, [2])):
        d = tfr[L] - tde[L]
        d /= np.linalg.norm(d)
        stores = [{"key": keys["Germany"], "value": d, "dose": 1.0,
                   "tier": "assoc", "support": f"L{L} native"}]
        for c in COUNTRIES:
            if c != "Germany":
                stores.append({"key": keys[c], "value": np.zeros(D),
                               "tier": "null", "support": "bg"})
        Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                  stores, key_scale=KEY_SCALE)
        for gn in gains:
            row = install_cell(promp, tok, tfr, tde, keys, Ua, Vc,
                               paris, base, L, gn)
            print(f"L{L} gain={gn}: " + " ".join(row), flush=True)
    # (2) churn profile
    for c in COUNTRIES:
        t7, gids = fwdH(promp[c], keep="all")
        _, pos = country_pos(promp[c], c, tok)
        ve, vc = [], []
        for L in range(1, 28):
            a, b = t7[L][-1], t7[L - 1][-1]
            ve.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)))
            a, b = t7[L][pos], t7[L - 1][pos]
            vc.append(float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)))
        print(f"churn {c} end-row: " +
              " ".join(f"{v:.2f}" for v in ve), flush=True)
        print(f"churn {c} country-row: " +
              " ".join(f"{v:.2f}" for v in vc), flush=True)


if __name__ == "__main__":
    main()
