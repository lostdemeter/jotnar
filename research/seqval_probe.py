"""Sequential values: multi-token installs need per-position support.

Orwell first-only stalls at fragments (', 19, in 1', gain-invariant):
a single first-token value cannot complete a 5-token surface (later
pieces lack support). Probe: advance the VALUE per step (step-k gets
k-th target row, then hold last), same address/routing/dose. If
"1984" surfaces in order, multi-token installation = sequential
values (host-scheduled Vc per step; program unchanged) -- installs
compose over TIME as well as over stores.
Usage: python3 research/seqval_probe.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from chain.engram import yarnball_bank

PROMPT = "George Orwell wrote the novel"
SUBJ = "Orwell"
TARGET = os.environ.get("SEQ_TARGET", " 1984")
GAIN = 0.5
N_GEN = 8
KEY_SCALE = 8.0


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    from qwen_torch import fwd, fwdH
    from siphon_ball import early_key
    from siphon_battery import country_pos
    g, tok = load7b()
    Wlog = np.asarray(g("lm_head.weight"), dtype=np.float64)
    tids = tok(TARGET, return_tensors="pt")["input_ids"][0].tolist()
    print(f"target {TARGET!r} = {tids}", flush=True)
    _, pos = country_pos(PROMPT, SUBJ, tok)
    key, _ = early_key(PROMPT, pos=pos)
    D = 3584
    vrows = [Wlog[t] / np.linalg.norm(Wlog[t]) for t in tids]
    PIECE = int(os.environ.get("PIECE", 0))
    print(f"seed piece {PIECE} = {tok.decode([tids[PIECE]])!r}", flush=True)
    Ua, _, _ = yarnball_bank(
        np.zeros((D, 0)), np.zeros((0, D)),
        [{"key": key, "value": vrows[0], "dose": 1.0, "tier": "assoc",
          "support": "address only (values scheduled per step)"},
         {"key": -key, "value": np.zeros(D), "tier": "null",
          "support": "background"}],
        key_scale=KEY_SCALE)
    seq = tok(PROMPT, return_tensors="pt")["input_ids"][0].tolist()
    for step in range(N_GEN):
        cur = tok.decode(seq)
        t7, _ = fwdH(cur, keep="all")
        n = len(seq)
        prow = pos if pos < n else n - 1
        xa = t7[2][prow]
        xa = xa / np.linalg.norm(xa)
        C = xa @ Ua
        P = np.exp(C - C.max())
        P /= P.sum()
        mag = float(np.linalg.norm(t7[27][-1]))
        # Seed hypothesis (Hamlet analog): ONE contentful piece every
        # step; the model completes the rest itself. Sequential values
        # already failed (base persists + ' 1' leak): fragments compose
        # in machinery, not in rows.
        v = vrows[PIECE]
        y = (P[0] * v) * (GAIN * mag)
        yn = y / (np.linalg.norm(y) + 1e-12)
        lg, _ = fwd(cur, steer=(27, yn, float(np.linalg.norm(y) / mag)))
        top = int(lg.argmax())
        seq = seq + [top]
    txt = tok.decode(seq)
    print(f"seqval: {txt[:120]!r}", flush=True)
    print(f"1984-in-order: {all(t in seq for t in tids)}", flush=True)


if __name__ == "__main__":
    main()
