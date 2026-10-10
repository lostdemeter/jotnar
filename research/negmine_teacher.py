"""Teacher-side negmine: nulls keyed to what-must-not-fire.

Install Germany->Paris (contrast, gain ladder {1,4,8}) over battery +
DIVERGENTS: reversed relation ("Berlin is the capital of" -- same
entities, inverted intent: must NOT install Paris), multi-entity
("Germany and France are_potentates..." keep simple: "Germany and
France are"), chatter ("hello"). Collect leaks (non-target prompts
flipping to Paris), mine their early rows, append as nulls, re-gate:
install intact + all divergents hold. Nulls keyed to must-not-fire
(the XOR thread: absence with an address). Live-mined every run.
Usage: python3 research/negmine_teacher.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank
from siphon_battery import country_pos

TARGET = ("The capital of Germany is", "Germany")
DIVERG = [("Berlin is the capital of", "Berlin"),
          ("Germany and France are", "Germany"),
          ("hello", None)]
CONTROLS = ["The capital of Italy is", "The capital of Japan is"]
GAINS = [1, 4, 8]
KEY_SCALE = 8.0
MAXNEG = 8


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    _, tok = load7b()
    g, _ = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    D = 3584
    tprompt, tsubj = TARGET
    t7t, _ = fwdH("The capital of France is")
    t7d, _ = fwdH(tprompt)
    d = t7t[27] - t7d[27]
    d /= np.linalg.norm(d)
    _, tpos = country_pos(tprompt, tsubj, tok)
    tk, _ = early_key(tprompt, pos=tpos)
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]

    def mine(p, sub=None):
        t, gids = fwdH(p, keep="all")
        n = len(gids)
        if sub is None:
            pos = n - 1
        else:
            _, pos = country_pos(p, sub, tok)
        k, _ = early_key(p, pos=pos)
        return t, (k, pos)

    keys = {"italy": mine(CONTROLS[0], "Italy")[1][0],
            "japan": mine(CONTROLS[1], "Japan")[1][0]}
    base = {}
    for p in [tprompt] + CONTROLS + [d for d, _ in DIVERG]:
        lg, _ = fwd(p)
        base[p] = int(lg.argmax())
    print("unsteered:", {p[:24]: tok.decode([v]) for p, v in base.items()},
          flush=True)

    def bank(negs):
        stores = [{"key": tk, "value": d, "dose": 1.0, "tier": "assoc",
                   "support": "germany-contrast"}]
        for c in ("italy", "japan"):
            stores.append({"key": keys[c], "value": np.zeros(D),
                           "tier": "null", "support": f"bg-{c}"})
        for k, s in negs:
            stores.append({"key": k, "value": np.zeros(D), "tier": "null",
                           "support": f"neg-{s}"})
        return yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                             stores, key_scale=KEY_SCALE)

    def cell(p, sub, Ua, Vc, gn, want):
        t7, _ = fwdH(p, keep="all")
        if sub is None:
            n = len(tok(p, return_tensors="pt")["input_ids"][0])
            pos = n - 1
        else:
            _, pos = country_pos(p, sub, tok)
        xa = t7[2][pos]
        xa = xa / np.linalg.norm(xa)
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        mag = float(np.linalg.norm(t7[27][-1]))
        y = (P @ Vc) * (gn * mag)
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(p, steer=(27, yn, float(np.linalg.norm(y) / mag)))
        top = int(lg.argmax())
        if want == "install":
            ok = top == paris
            mark = "INSTALL" if ok else f"parisr{int((lg > lg[paris]).sum()) + 1}"
        else:
            ok = top == base[p]
            mark = "hold" if ok else f"MOVED->{tok.decode([top])[:10]!r}"
        return ok, mark, int(P.argmax())

    tests = [(tprompt, tsubj, "install"),
             ("The capital of Italy is", "Italy", "hold"),
             ("The capital of Japan is", "Japan", "hold")] + \
            [(p, s, "hold-div") for p, s in DIVERG]
    Ua, Vc, _ = bank([])
    print("== pass 1 (no negs) ==", flush=True)
    leaks = []
    for gn in GAINS:
        for p, s, want in tests:
            ok, mark, ret = cell(p, s, Ua, Vc, gn, want)
            tag = "OK " if (ok or (want == "hold-div")) else "FAIL"
            # divergents: collect LEAKS (flips to paris), not base-holds
            print(f"  gain={gn} {p[:24]!r}: {mark} ret={ret} "
                  f"{tag if want != 'hold-div' else ''}", flush=True)
            if want == "hold-div":
                leaks.append((p, s, mark))
    # mine leaks -> nulls (any divergent that moved at any gain)
    negs = []
    seen = set()
    for p, s, mark in leaks:
        if "MOVED" in mark and p not in seen and len(negs) < MAXNEG:
            t, kk = mine(p, s)
            h = t[2][kk[1]]
            negs.append((h / np.linalg.norm(h),
                         p[:20].replace(" ", "_")))
            seen.add(p)
    print(f"neg-mined: {len(negs)} ({[s for _, s in negs]})", flush=True)
    Ua2, Vc2, _ = bank(negs)
    print("== pass 2 (neg-mined) ==", flush=True)
    for gn in GAINS:
        for p, s, want in tests:
            ok, mark, ret = cell(p, s, Ua2, Vc2, gn, want)
            good = ok if want != "hold-div" else (mark == "hold")
            print(f"  gain={gn} {p[:24]!r}: {mark} ret={ret} "
                  f"{'OK' if good else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
