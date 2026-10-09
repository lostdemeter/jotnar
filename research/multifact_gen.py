"""Multi-fact generation: shared bank, per-prompt first-only pushes.

Two installs (Hamlet 0.5, New-York 0.05 -- per-fact micro-doses from
the point doctrine) share one bank [Ham, New, nulls]; two prompts
generate with first-only pushes. Doses folded into Vc rows (gain 1.0
steer; mags measured per prompt -- dose discipline, not tuning).
Gates per prompt: target surface present + degenerate<=1 + control
(unsteered lacks it). Shared-bank coexistence THROUGH generation:
either prompt must not show the other's content (cross-talk watch).
Usage: python3 research/multifact_gen.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

PROMS = {
    "hamlet": ("Shakespeare wrote the play", "Shakespeare", "Hamlet", 0.5),
    "nycity": ("The largest city in America is", "America", "New York", 0.05),
}
N_GEN = 8
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
    D = 3584
    keys, firstrows = {}, {}
    for name, (prompt, subj, want, dose) in PROMS.items():
        tids = tok(" " + want.split()[0], return_tensors="pt")["input_ids"][0].tolist()
        firstrows[name] = Wlog[tids[0]] / np.linalg.norm(Wlog[tids[0]])
        pos = span_pos(prompt, subj, tok)
        k, _ = early_key(prompt, pos=pos)
        keys[name] = (k, pos)
    stores = []
    for name in PROMS:
        stores.append({"key": keys[name][0], "value": firstrows[name],
                       "dose": 1.0, "tier": "assoc",
                       "support": f"first-row {name}"})
    stores.append({"key": -keys["hamlet"][0], "value": np.zeros(D),
                   "tier": "null", "support": "background"})
    Ua, Vc0, _ = yarnball_bank(np.zeros((D, 0)), np.zeros((0, D)),
                               stores, key_scale=KEY_SCALE)
    for name, (prompt, subj, want, dose) in PROMS.items():
        seq = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
        # per-fact dose folded into ITS value row (index = fact order)
        idx = list(PROMS).index(name)
        mag0, _ = fwdH(prompt)
        mag = float(np.linalg.norm(mag0[27]))
        Vc = Vc0.copy()
        Vc[idx] = Vc[idx] / (np.linalg.norm(Vc[idx]) + 1e-12) * dose * mag
        outs = []
        for step in range(N_GEN):
            cur = tok.decode(seq)
            t7, _ = fwdH(cur, keep="all")
            n = len(seq)
            pos = keys[name][1]
            prow = pos if pos < n else n - 1
            xa = t7[2][prow]
            xa = xa / np.linalg.norm(xa)
            C = xa @ Ua
            P = np.exp(C - C.max())
            P /= P.sum()
            m = float(np.linalg.norm(t7[27][-1]))
            y = (P @ Vc) * m
            yn = y / (np.linalg.norm(y) + 1e-12)
            use = step == 0
            lg, _ = fwd(cur, steer=(27, yn, float(np.linalg.norm(y) / m))
                        if use else None)
            top = int(lg.argmax())
            outs.append(top)
            seq = seq + [top]
        txt = tok.decode(seq)
        deg = txt.count("�")
        for i in range(2, len(seq)):
            if seq[i] == seq[i - 1] == seq[i - 2]:
                deg += 1
        other = [v for k, v in PROMS.items() if k != name][0][2]
        print(f"{name}: {txt[:90]!r} want={want!r} hit={want.lower() in txt.lower()} "
              f"deg={deg} crosstalk={other.lower() in txt[len(prompt):].lower() if len(txt) > len(prompt) else False}",
              flush=True)


if __name__ == "__main__":
    main()
