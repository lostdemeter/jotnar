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
    prev = {}
    agg = {short: [[], []] for short in ROLES}
    for L in range(28):
        cur = {}
        for short, dotted in ROLES.items():
            cur[short] = np.ascontiguousarray(
                g(f"model.layers.{L}.{dotted}"), dtype=np.float64).ravel()
        if prev:
            for short in ROLES:
                a, b = prev[short], cur[short]
                c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
                r = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))
                agg[short][0].append(c)
                agg[short][1].append(r)
                if c > 0.99:
                    print(f"  SHARE-CANDIDATE {short}@L{L - 1}/L{L}: cos={c:.4f}",
                          flush=True)
        prev = cur
        del cur
        import gc as _gc
        _gc.collect()
    for short in ROLES:
        cos, rel = agg[short]
        print(f"{short:6} adj-cos min={min(cos):.4f} mean={float(np.mean(cos)):.4f} "
              f"relFrob max={max(rel):.3f} mean={float(np.mean(rel)):.3f}",
              flush=True)
    print("streaming held <=2 layers (~4GB peak); no pair table retained",
          flush=True)


if __name__ == "__main__":
    main()
