"""Lightweight snapshot smoke: tokenizer + one weight, no GPU, no nvcc.

Gates the download before anything heavy runs. Peak RSS is one weight.
Usage: python3 scripts/smoke_qwen7b.py
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from chain.qwen7b import (snapshot_ok, load7b, HIDDEN, N_HEADS, N_KV, DH,
                              QWEN7B_VOCAB, QWEN7B_INTER)
    check("smoke-snapshot", snapshot_ok(), "pinned revision present")
    if not snapshot_ok():
        print("FAILURES:", FAIL)
        sys.exit(1)
    g, tok = load7b()
    ids = tok("The capital of France is",
              return_tensors="pt")["input_ids"][0].numpy().tolist()
    check("smoke-tok", len(ids) > 0, f"{len(ids)} ids")
    E = g("model.embed_tokens.weight")
    check("smoke-embed", tuple(E.shape) == (QWEN7B_VOCAB, HIDDEN),
          f"{tuple(E.shape)} (vocab assert doubles as geometry pin)")
    del E
    wq = g("model.layers.0.self_attn.q_proj.weight")
    check("smoke-wq", tuple(wq.shape) == (HIDDEN, HIDDEN), f"{tuple(wq.shape)}")
    del wq
    wu = g("model.layers.0.mlp.up_proj.weight")
    check("smoke-wup", tuple(wu.shape) == (QWEN7B_INTER, HIDDEN),
          f"{tuple(wu.shape)}")
    del wu
    import gc
    gc.collect()
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
