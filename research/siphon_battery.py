"""Siphon generality battery (teacher side, torch mirror).

Single-fact Germany ball (contrast value + nulls, ledgered) over a
wider prompt set: target Germany + controls France/Italy/Spain/Japan/
China. Reports unsteered tops (multi-fact scouting: non-vacuous facts
are install candidates), gated tops at gains 1 and 8, retrieval per
prompt. Live-mined every run.
Usage: python3 research/siphon_battery.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

COUNTRIES = ["Germany", "France", "Italy", "Spain", "Japan", "China"]
TARGET = "Germany"
GAINS = [1, 8]
KEY_SCALE = 8.0


def country_pos(prompt, country, tok):
    """Token index of the country word via offset mapping (robust to
    subword splits: no fragile substring matching)."""
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    ids = enc["input_ids"][0].numpy()
    offs = enc["offset_mapping"][0].numpy()
    lo = prompt.find(country)
    assert lo >= 0, f"{country} not in {prompt!r}"
    hi = lo + len(country)
    for i, (a, b) in enumerate(offs):
        if a < hi and lo < b:
            return ids, i
    raise AssertionError(f"no token covers {country} in {prompt!r}")


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
        o = np.argsort(-lg)
        base[c] = (int(o[0]), int(o[1]))
        print(f"base {c:8} top={tok.decode([base[c][0]])!r} "
              f"runner={tok.decode([base[c][1]])!r}", flush=True)

    tfr, _ = fwdH("The capital of France is")
    tde, _ = fwdH("The capital of Germany is")
    d = tfr[27] - tde[27]
    d /= np.linalg.norm(d)
    from siphon_ball import early_key
    keys = {}
    for c in COUNTRIES:
        _, pos = country_pos(promp[c], c, tok)
        k, _ = early_key(promp[c], pos=pos)
        keys[c] = k
        print(f"key {c}: country-tok-pos={pos}", flush=True)
    D = 3584
    stores = [{"key": keys[TARGET], "value": d, "dose": 1.0,
               "tier": "assoc",
               "support": "L27 France-minus-Germany contrast"}]
    for c in COUNTRIES:
        if c != TARGET:
            stores.append({"key": keys[c], "value": np.zeros(D),
                           "tier": "null", "support": "background"})
    Ua, Vc, ledger = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                   stores, key_scale=KEY_SCALE)
    print(f"ball: {Ua.shape[1]} stores", flush=True)

    for gn in GAINS:
        row = []
        for c in COUNTRIES:
            t7, gids = fwdH(promp[c], keep="all")
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
            pr = int((lg > lg[paris]).sum()) + 1
            ret = int(P.argmax())
            want = COUNTRIES.index(c)
            mark = "RETR" if ret == want else f"ret{ret}"
            if c == TARGET:
                verdict = "INSTALL" if top == paris else ""
            else:
                verdict = "hold" if top == base[c][0] else "MOVED"
            row.append(f"{c[:4]}:{tok.decode([top])[:8]}^{pr}:{mark}:{verdict}")
        print(f"gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
