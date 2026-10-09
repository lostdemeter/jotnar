"""Paraphrase-sharing gate: one store covers the template family.

Coldstart showed siblings flip free via shared asker-words (graded
MOVED per protocol). Reframed as design: key = the asker (question
noun present in every template), value = the answer. Install on
template A, assert sibling template B flips + cross-family holds.
Bank per fact: [asker store | sibling-asker nulls]. Fixed ks=8,
gain 1 (operating points, no search). If green: paraphrase coverage
is a property of asker-keying (one store per (asker, content)), and
the coldstart "misses" were the feature working.
Usage: python3 research/paraphrase.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank
from coldstart import span_pos

FAMS = {
    "everest": [("The tallest mountain on Earth is", "mountain", " Everest"),
                ("Earth's tallest mountain is called", "mountain", " Everest")],
    "nile": [("The longest river in Africa is", "river", " Nile"),
             ("Africa's longest river is called", "river", " Nile")],
}
KEY_SCALE = 8.0
GAIN = 1


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
    D = 3584
    for fam, members in FAMS.items():
        prompts = {m[0]: m for m in members}
        base = {}
        for p, _, _ in members:
            lg, _ = fwd(p)
            base[p] = int(lg.argmax())
        # install on template A only
        pa, aska, ta = members[0]
        tid = tok(ta, return_tensors="pt")["input_ids"][0].tolist()[0]
        posa = span_pos(pa, aska, tok)
        ka, _ = early_key(pa, pos=posa)
        v = Wlog[tid] / np.linalg.norm(Wlog[tid])
        others = [(p, s) for fam2, ms in FAMS.items() if fam2 != fam
                  for (p, s, _) in ms]
        stores = [{"key": ka, "value": v, "dose": 1.0, "tier": "assoc",
                   "support": f"asker {aska} -> {ta}"}]
        for p, s in others:
            pos = span_pos(p, s, tok)
            k, _ = early_key(p, pos=pos)
            stores.append({"key": k, "value": np.zeros(D),
                           "tier": "null", "support": "background"})
        Ua, Vc, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                                  stores, key_scale=KEY_SCALE)
        evals = [(p, s, t) for (p, s, t) in members] + \
                [(p, s, None) for (p, s) in others]
        for p, s, t in evals:
            t7, _ = fwdH(p, keep="all")
            pos = span_pos(p, s, tok)
            xa = t7[2][pos]
            xa = xa / np.linalg.norm(xa)
            C = xa @ Ua
            P = np.exp(C - C.max())
            P /= P.sum()
            mag = float(np.linalg.norm(t7[27][-1]))
            y = (P @ Vc) * (GAIN * mag)
            yn = y / (np.linalg.norm(y) + 1e-12)
            lg, _ = fwd(p, steer=(27, yn, float(np.linalg.norm(y) / mag)))
            top = int(lg.argmax())
            if p == pa:
                mark = "INSTALL" if top == tid else f"r{int((lg > lg[tid]).sum()) + 1}"
            elif t is not None:
                mark = "SHARED" if top == tid else f"r{int((lg > lg[tid]).sum()) + 1}"
            else:
                mark = "hold" if top == base[p] else "MOVED"
            print(f"{fam} A-install {p[:30]!r}: top={tok.decode([top])[:12]!r} "
                  f"{mark} ret={int(P.argmax())}", flush=True)


if __name__ == "__main__":
    main()
