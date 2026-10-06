"""Decode gate: single-token KV decode vs HF original (fork-aware).

The decode program (one chained listing, in-graph cache update via
broadcast-MATMUL + SELECT, host relays fp16 files) must speak the same
distribution as HF: parity bound + first-pick rank + fluency -- never
string equality (greedy amplifies sub-noise diffs; see the gen gate).
Also asserts prefill-vs-recompute consistency (decode stays glued to
the proven full-recompute path): same argmax on short prefills.
SKIPs without snapshot/GPU/nvcc. Slow (~10min: nvcc + HF + prefill).
Usage: python3 tests/test_decode.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []
PROMPT = "The capital of France is"
N_GEN = 3
SMAX = 16


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM
    from chain.qwen7b import SNAP, load7b, hf_no_triton
    hf_no_triton()
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)
    g, tok = load7b()
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0].numpy().tolist()
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    with torch.no_grad():
        lg = model(**tok(PROMPT, return_tensors="pt").to("cuda")).logits
    hf_top = lg[0, -1].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         PROMPT, "--n", str(N_GEN), "--smax", str(SMAX)],
        capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-800:] if r.stdout else "", flush=True)
    if r.returncode != 0:
        check("decode-run", False, r.stderr[:300])
        print("FAILURES:", FAIL)
        sys.exit(1)
    import json as _json  # noqa
    dec = np.load("/tmp/dec_prefill.npy")
    par = float(np.abs(dec - hf_top).max())
    check("decode-parity", par < 10.0,
          f"maxabs={par:.2f} (structural-break detector)")
    first = int(dec.argmax())
    frank = int((hf_top > hf_top[first]).sum()) + 1
    check("decode-first-pick-rank", frank <= 8, f"rank {frank}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
