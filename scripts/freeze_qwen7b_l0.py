"""Freeze Qwen2-7B-Instruct L0 MLP as native phi triples (local cache).

Why freeze (benefit, measured in test_qwen7b_teach.py): encoding 200M
params through log/pow costs minutes per run; fread of frozen int
planes costs seconds. Frozen bytes are also dependency-free (no
phi-core/torch at load) and replay-identical. Cost, stated: triples
are 6B/elem vs bf16 2B/elem (~1.2GB for this organ); full-7B native
would run ~40GB. This file is LOCAL CACHE (never commit; the test
SKIPs gracefully without it and encodes from snapshot as fallback).
Usage: python3 scripts/freeze_qwen7b_l0.py
Writes: data/qwen7b_l0_mlp.npz + data/qwen7b_l0_mlp_manifest.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from chain.qwen7b import SNAP, load7b


def main():
    import numpy as np
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct in local HF cache)")
        sys.exit(0)
    import phi_core.lattice as S
    g, _ = load7b()
    mats = {}
    for key, src in (("wupT", "model.layers.0.mlp.up_proj.weight"),
                     ("wgateT", "model.layers.0.mlp.gate_proj.weight"),
                     ("wdownT", "model.layers.0.mlp.down_proj.weight")):
        mats[key] = g(src).T
    mats["ln"] = g("model.layers.0.post_attention_layernorm.weight")
    t0 = time.perf_counter()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    dd = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    from chain.frozen import save_shared, share_report
    logical = {}
    for k, v in mats.items():
        s, e, z = enc(v)
        logical[f"{k}_s"] = np.ascontiguousarray(s)
        logical[f"{k}_e"] = np.ascontiguousarray(e)
        logical[f"{k}_z"] = np.ascontiguousarray(z)
    enc_s = time.perf_counter() - t0
    man, saved = save_shared(os.path.join(dd, "qwen7b_l0_mlp.npz"), logical,
                             os.path.join(dd, "qwen7b_l0_mlp_manifest.json"))
    man.update({"snapshot": os.path.basename(SNAP), "shapes":
                {k: list(v.shape) for k, v in mats.items()},
                "encode_seconds": round(enc_s, 1)})
    json.dump(man, open(os.path.join(dd, "qwen7b_l0_mlp_manifest.json"), "w"),
              indent=1)
    nbytes = sum(v.nbytes for v in logical.values())
    print(f"froze L0 MLP triples: {nbytes / 1e9:.2f}GB logical in {enc_s:.0f}s "
          f"(encode once, fread forever); {share_report(man)}")


if __name__ == "__main__":
    main()
