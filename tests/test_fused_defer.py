"""Fused-attention deferral gate (pattern A4): measured, not built.

Unfused composition (tabled BMMV/SOFTMAX_WIDE math) BEATS industrial
fused SDPA at student scale (676/2692 tokens: 0.31x/0.50x wall) —
card falsifier fires, A4 DEFERS with evidence. Legs (CPU except a
tiny GPU composition-agreement smoke when a GPU is present):
composition agreement (explicit composite == SDPA-math reference),
banked wall table present with ratio < 1.0 (defer condition), no
fused encoder kernel exists (tripwire). Wall methodology in
samples/marigold/fused_measure.json; gate never times (flaky).
Usage: python3 tests/test_fused_defer.py
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import numpy as np
    rng = np.random.default_rng(44)
    S, W = 64, 32
    Q = rng.normal(size=(S, W))
    K = rng.normal(size=(S, W))
    V = rng.normal(size=(S, W))
    a = Q @ K.T / 8.0
    a = np.exp(a - a.max(-1, keepdims=True))
    a = a / a.sum(-1, keepdims=True)
    ref = a @ V
    import torch
    got = (torch.softmax(torch.tensor(Q) @ torch.tensor(K).T / 8.0, -1)
           @ torch.tensor(V)).numpy()
    check("fused-composed-agrees", float(np.abs(got - ref).max()) < 1e-9,
          "explicit composite == reference (math covered)")

    bank = json.load(open(os.path.join(
        ROOT, "samples", "marigold", "fused_measure.json")))
    ratios = bank.get("ratio_unfused_over_fused", {})
    check("fused-defer-condition",
          set(ratios) == {"676", "2692"}
          and all(0 < v < 1.0 for v in ratios.values()),
          f"unfused faster both scales: {ratios} (falsifier fires)")
    cu = open(os.path.join(ROOT, "chain/emit_cuda.py")).read()
    check("fused-no-kernel", "k_joint_attn" not in cu and "k_fused_attn" not in cu,
          "no fused encoder kernel on CUDA")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
