"""Cross-layer weight similarity audit (dedup candidates, CPU statics).

For each role (wq/wk/wv/wo/wup/wgate/wdown/ln1/ln2) and each adjacent
layer pair: cosine similarity + relative Frobenius distance. High
similarity pairs share one copy (symlink) with predicted parity cost;
low everywhere = layers are distinct, dedup dead, decide in minutes
not weeks. Also reports global min/max per role (where rhyme lives,
if anywhere). No GPU, no forwards: pure weight statics.
Usage: python3 research/layer_sim.py (needs 7B snapshot; CPU only)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

ROLES = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
         "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
         "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
         "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
         "ln2": "post_attention_layernorm.weight"}


def main():
    from chain.qwen7b import load7b, snapshot_ok
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    g, _ = load7b()
    W = {}
    for L in range(28):
        for short, dotted in ROLES.items():
            W[(short, L)] = np.ascontiguousarray(
                g(f"model.layers.{L}.{dotted}"), dtype=np.float64).ravel()
    for short in ROLES:
        cos, rel = [], []
        for L in range(27):
            a, b = W[(short, L)], W[(short, L + 1)]
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)))
            rel.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)))
        print(f"{short:6} adj-cos min={min(cos):.4f} mean={float(np.mean(cos)):.4f} "
              f"relFrob max={max(rel):.3f} mean={float(np.mean(rel)):.3f}",
              flush=True)
    print("DEDUP-VERDICT: pairs with adj-cos > 0.99 below (share candidates):",
          flush=True)
    found = 0
    for short in ROLES:
        for L in range(27):
            a, b = W[(short, L)], W[(short, L + 1)]
            c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            if c > 0.99:
                print(f"  {short}@L{L}/L{L + 1}: cos={c:.4f}", flush=True)
                found += 1
    if not found:
        print("  none -- layers distinct, dedup dead by measurement", flush=True)


if __name__ == "__main__":
    main()
