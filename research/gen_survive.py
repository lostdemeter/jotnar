"""Generation survival: installs must live past the first token.

All install gates so far are point measurements (prompt-end rank).
Multi-token targets (Hamlet = Ham+let, 1984 = five tokens) need
TRAJECTORY methodology: steer every step, generate greedily, check
the target surface appears in order. Ball: first-token readout row
(exact native content) + lex address (subject substring) + nulls.
Metric-setting run: report steered vs unsteered trajectories +
per-step target ranks; the survival criterion (consecutive target
tokens from step 0?) is set from data, not assumed.
Live-mined every run. Usage: research/gen_survive.py (7B + GPU).
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

PROMS = {
    "hamlet": ("Shakespeare wrote the play", "Shakespeare", " Hamlet"),
    "orwell": ("George Orwell wrote the novel", "Orwell", " 1984"),
}
GAINS = [1, 2]
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
    for name, (prompt, subj, target) in PROMS.items():
        tids = tok(target, return_tensors="pt")["input_ids"][0].tolist()
        lg0, _ = fwd(prompt)
        print(f"== {name}: target={target!r}{tids} base-top="
              f"{tok.decode([int(lg0.argmax())])!r}", flush=True)
        t7, _ = fwdH(prompt, keep="all")
        toks = tok.convert_ids_to_tokens(
            tok(prompt, return_tensors="pt")["input_ids"][0].numpy())
        pos = span_pos(prompt, subj, tok)
        # address key = early subject state (prompt's own early row)
        key, _ = early_key(prompt, pos=pos)
        Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
        v = Wlog[tids[0]] / np.linalg.norm(Wlog[tids[0]])
        D = 3584
        Ua, Vc, _ = yarnball_bank(
            np.zeros((D, 0)), np.zeros((0, D)),
            [{"key": key, "value": v, "dose": 1.0, "tier": "assoc",
              "support": f"first-token readout {target}"},
             {"key": -key, "value": np.zeros(D), "tier": "null",
              "support": "background (anti-address)"}],
            key_scale=KEY_SCALE)
        for gn in GAINS:
            for tag, dose in (("off", 0.0), ("on", gn)):
                seq = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
                outs = []
                for _ in range(N_GEN):
                    cur = tok.decode(seq)
                    t7c, _ = fwdH(cur, keep="all")
                    # address follows the subject row while present; else end
                    n = len(seq)
                    prow = pos if pos < n else n - 1
                    xa = t7c[2][prow]
                    xa = xa / np.linalg.norm(xa)
                    C = xa @ Ua
                    P = np.exp(C - C.max())
                    P /= P.sum()
                    mag = float(np.linalg.norm(t7c[27][-1]))
                    y = (P @ Vc) * (dose * mag)
                    yn = y / (np.linalg.norm(y) + 1e-12)
                    lg, ids = fwd(cur, steer=(27, yn, float(np.linalg.norm(y) / mag)) if dose else None)
                    top = int(lg.argmax())
                    outs.append(top)
                    seq = seq + [top]
                txt = tok.decode(seq)
                print(f"  {name} gain={gn} {tag}: {txt[:100]!r}", flush=True)
        print("", flush=True)


if __name__ == "__main__":
    main()
