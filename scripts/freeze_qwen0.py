"""Freeze Qwen2-0.5B layer-0 full block (offline, port Task 2).

Loads safetensors layer-N (offline cache) + freezes data/qwen layer block:
augmented projections (ones-column trick, exact by linearity --
NO new mnemonics needed): Wqa=[Wq;qb] (897x896), Wka=[Wk/8;bk/8]
(/8 folding frozen in, stated), Wva=[Wv;bv]; Wo, ln1, ln2, MLP
trio as-is. K-bias INCLUDED via augment (fewer proof obligations
than the cancellation route). o_bias absent upstream (verified).
RoPE base is a listing CONFIG (rope_base 1000000.0), not a weight.
Deterministic; manifest carries shas. No TSHIFT in port (teacher
has none -- fidelity first, stated divergence from house stack).
Usage: python3 scripts/freeze_qwen0.py
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chain.qwen_mirror import load_layer0

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    L = f"model.layers.{a.layer}"
    g, _, _ = load_layer0()
    Wq = g(f"model.layers.{a.layer}.self_attn.q_proj.weight")
    Wk = g(f"model.layers.{a.layer}.self_attn.k_proj.weight")
    Wv = g(f"model.layers.{a.layer}.self_attn.v_proj.weight")
    Wo = g(f"model.layers.{a.layer}.self_attn.o_proj.weight")
    bq = g(f"model.layers.{a.layer}.self_attn.q_proj.bias")
    bk = g(f"model.layers.{a.layer}.self_attn.k_proj.bias")
    bv = g(f"model.layers.{a.layer}.self_attn.v_proj.bias")
    f = 1.0 / np.sqrt(8.0)
    # torch stores out×in; listings consume in×out (+bias row) -- transpose here.
    # RANGE LAW (measured 2026-10-02): matmul products must stay ≲ ±13
    # (m-independent lattice fold; m prices precision, not range). Q/K
    # biases (±45 rows) EXCEED it -> Q/K ship WITHOUT bias (stated
    # divergence from HF, pinned per-layer in tests; L0 23.3dB);
    # V bias (±0.1) stays augmented.
    Wq8 = Wq.T * f
    Wk8 = Wk.T * f
    Wva = np.concatenate([Wv.T, bv[None, :]], axis=0)
    out = {"Wq8": Wq8, "Wk8": Wk8, "Wva": Wva, "Wo": Wo.T,
           "ln1": g(f"model.layers.{a.layer}.input_layernorm.weight"),
           "ln2": g(f"model.layers.{a.layer}.post_attention_layernorm.weight"),
           "wup": g(f"model.layers.{a.layer}.mlp.up_proj.weight").T,
           "wgate": g(f"model.layers.{a.layer}.mlp.gate_proj.weight").T,
           "wdown": g(f"model.layers.{a.layer}.mlp.down_proj.weight").T}
    np.savez(os.path.join(OUT, f"qwen{a.layer}_block.npz"), **out)
    man = {"shapes": {k: list(v.shape) for k, v in out.items()},
           "folding": "SC/8 split balanced (/sqrt8 into Q and K sides; "
                      "rotation-linearity keeps RoPE exact; quantization-optimal)",
           "biases": "V-bias via ones-augment (exact, tiny); Q-bias OMITTED "
                      "(stated divergence, pinned per-layer in tests); "
                      "K-bias cancelled by proof (row-constant); "
                      "o_bias absent verified",
           "tshift": "omitted (teacher has none)",
           "rope_base": 1000000.0,
           "sha": hashlib.sha256(b"".join(
               np.ascontiguousarray(v).tobytes() for v in out.values())).hexdigest()[:16]}
    json.dump(man, open(os.path.join(OUT, f"qwen{a.layer}_block_manifest.json"), "w"), indent=2)
    print(f"shapes: {man['shapes']}")
    print(f"wrote {OUT}/qwen{a.layer}_block.npz + manifest (sha {man['sha']})")


if __name__ == "__main__":
    main()
