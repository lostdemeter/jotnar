"""Siphon multi-fact bank (teacher side, torch mirror).

Two installs, one bank: [Germany: France-minus-Germany Paris value |
Japan: France-minus-Japan Tokyo value | France/Italy/Spain/China:
nulls]. Question: do installs coexist through one softmax (shared
address space, disjoint content)? Gates: Germany Paris r1 AND Japan
Tokyo r1, 4 controls hold, all retrieve own. Live-mined every run.
Usage: python3 research/siphon_multi.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank
from siphon_battery import country_pos

COUNTRIES = ["Germany", "France", "Italy", "Spain", "Japan", "China"]
FACTS = {"Germany": "Paris", "Japan": "Tokyo"}
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
    tgt_ids = {t: tok(" " + t, return_tensors="pt")["input_ids"][0].tolist()[0]
               for t in ("Paris", "Tokyo")}

    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(promp[c])
        base[c] = int(lg.argmax())
    print("unsteered:", {c: tok.decode([v]) for c, v in base.items()}, flush=True)

    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    d_de = tfr[27] - tde[27]
    d_de /= np.linalg.norm(d_de)
    # Japan value: Tokyo readout row (mine_tokyo screen: pair-contrasts
    # carry the CONTRAST's capital -- Beijing/Madrid confirmed twice;
    # big-jp empty; decoder row installs Tokyo r1 ungated at all gains).
    # Readout rows are exact native content (precise-relations principle).
    g7, _ = load7b()
    Wlog = np.asarray(g7("lm_head.weight"), dtype=np.float64)
    d_jp = Wlog[tok(" Tokyo", return_tensors="pt")["input_ids"][0].tolist()[0]]
    d_jp /= np.linalg.norm(d_jp)
    print(f"value alignment: {float(d_de @ d_jp):.3f} (contrast value vs "
          f"readout-row value -- different species, shared bank)", flush=True)

    keys = {}
    for c in COUNTRIES:
        _, pos = country_pos(promp[c], c, tok)
        k, _ = early_key(promp[c], pos=pos)
        keys[c] = k
    D = 3584
    vals = {"Germany": (d_de, "L27 France-minus-Germany contrast"),
            "Japan": (d_jp, "Tokyo readout row (exact native content)")}
    stores = []
    for c in COUNTRIES:
        if c in vals:
            v, sup = vals[c]
            stores.append({"key": keys[c], "value": v, "dose": 1.0,
                           "tier": "assoc", "support": sup})
        else:
            stores.append({"key": keys[c], "value": np.zeros(D),
                           "tier": "null", "support": "background"})
    Ua, Vc, ledger = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                   stores, key_scale=KEY_SCALE)
    print(f"ball: {Ua.shape[1]} stores", flush=True)

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
            x27 = t7[27][-1]
            mag = float(np.linalg.norm(x27))
            y = (P @ Vc) * (gn * mag)
            yn = y / (np.linalg.norm(y) + 1e-12)
            lg, _ = fwd(promp[c], steer=(27, yn,
                                         float(np.linalg.norm(y) / mag)))
            top = int(lg.argmax())
            ret = int(P.argmax())
            want = COUNTRIES.index(c)
            mark = "RETR" if ret == want else f"ret{ret}"
            if c in FACTS:
                t = FACTS[c]
                r = int((lg > lg[tgt_ids[t]]).sum()) + 1
                verdict = "INSTALL" if top == tgt_ids[t] else f"{t}r{r}"
            else:
                verdict = "hold" if top == base[c] else "MOVED"
            row.append(f"{c[:4]}:{tok.decode([top])[:8]}:{mark}:{verdict}")
        print(f"gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
