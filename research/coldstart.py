"""Cold-start install battery (protocol v1, docs/COLDSTART.md).

Unseen facts (everest/nile -- never installed) x unseen templates
(paraphrases never mined) + shared-template controls. Generic driver:
offset-mapped early keys, readout-row values, sibling nulls, fixed
ks=8, gains {1,2}, L27. Predict install + holds, run once per cell,
grade. No per-case tuning: any failure is a species, not a knob.
Usage: python3 research/coldstart.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

CELLS = [
    ("everest", "The tallest mountain on Earth is", "Everest", " Everest"),
    ("everest2", "Earth's tallest mountain is called", "Everest", " Everest"),
    ("nile", "The longest river in Africa is", "Nile", " Nile"),
    ("nile2", "Africa's longest river is called", "Nile", " Nile"),
]
GAINS = [1, 2]
KEY_SCALE = 8.0


def span_pos(prompt, sub, tok):
    enc = tok(prompt, return_tensors="pt", return_offsets_mapping=True)
    offs = enc["offset_mapping"][0].numpy()
    lo = prompt.find(sub)
    assert lo >= 0, (sub, prompt)
    hi = lo + len(sub)
    for i, (a, b) in enumerate(offs):
        if a < hi and lo < b:
            return i
    raise AssertionError((sub, prompt))


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)

    prompts = {name: p for name, p, _, _ in CELLS}
    subj = {name: s for name, _, s, _ in CELLS}
    base = {}
    for name, p, _, _ in CELLS:
        lg, _ = fwd(p)
        base[name] = int(lg.argmax())
    print("unsteered:", {n: tok.decode([v]) for n, v in base.items()}, flush=True)
    # PREDICT (windows doctrine: L27 wide, gain 1-2 installs + holds)
    tids = {}
    for name, _, _, t in CELLS:
        tids[name] = tok(t, return_tensors="pt")["input_ids"][0].tolist()[0]
    for name, p, subj, t in CELLS:
        lg, _ = fwd(p)
        print(f"predict {name}: {tids[name] and ''}target {t!r} rank="
              f"{int((lg > lg[tids[name]]).sum()) + 1} -> top-1 @gain1-2, "
              f"controls hold", flush=True)

    keys = {}
    for name, p, subj, _ in CELLS:
        pos = span_pos(p, subj, tok)
        k, _ = early_key(p, pos=pos)
        keys[name] = (k, pos)
        print(f"key {name}: subject-row={pos}", flush=True)
    D = 3584
    for name, p, subj, t in CELLS:
        tid = tids[name]
        v = Wlog[tid] / np.linalg.norm(Wlog[tid])
        stores = [{"key": keys[name][0], "value": v, "dose": 1.0,
                   "tier": "assoc", "support": f"readout row {t}"}]
        for other, _, _, _ in CELLS:
            if other != name:
                stores.append({"key": keys[other][0], "value": np.zeros(D),
                               "tier": "null", "support": "background"})
        Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                  stores, key_scale=KEY_SCALE)
        for gn in GAINS:
            row = []
            for c, p2, _, _ in CELLS:
                t7, _ = fwdH(p2, keep="all")
                pos = span_pos(p2, subj[c], tok)
                xa = t7[2][pos]
                xa = xa / np.linalg.norm(xa)
                C = xa @ Ua
                P = np.exp(C - C.max())
                P /= P.sum()
                # stores order = [target-c, others...]: expected index is
                # 0 when c is the target, else index of c among others + 1.
                mag = float(np.linalg.norm(t7[27][-1]))
                y = (P @ Vc) * (gn * mag)
                yn = y / (np.linalg.norm(y) + 1e-12)
                lg, _ = fwd(p2, steer=(27, yn, float(np.linalg.norm(y) / mag)))
                top = int(lg.argmax())
                order = [name] + [o for o, _, _, _ in CELLS if o != name]
                want = order.index(c)
                ret = int(P.argmax())
                mark = "RETR" if ret == want else f"ret{ret}"
                if c == name:
                    r = int((lg > lg[tid]).sum()) + 1
                    verdict = "INSTALL" if top == tid else f"r{r}"
                else:
                    verdict = "hold" if top == base[c] else "MOVED"
                row.append(f"{c[:6]}:{tok.decode([top])[:10]}^{verdict}:{mark}")
            print(f"{name} gain={gn}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
