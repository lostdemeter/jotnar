"""Sparse-decode gate: top-k readout path vs HF + vs dense (fork charter).

The sparse layer (TOPK+GATHER+narrow-BMMV, same K/V caches, same
weights) must speak the same distribution as HF: parity bound +
first-pick rank (same bars as the dense decode gate -- decisions,
never strings), and its caches must equal the dense path's
bit-for-bit (identical Kfull/Vfull computation, deterministic relay).
SKIPs without snapshot/GPU/nvcc. Slow (~6min: two builds + runs).
Usage: python3 tests/test_sparse.py
"""
import filecmp
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
SMAX = 16
SPARSE_K = 8


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def run(prompt, n, smax, work, extra=()):
    env = dict(os.environ, QWEN_WORKDIR=work)
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         prompt, "--n", str(n), "--smax", str(smax)] + list(extra),
        capture_output=True, text=True, cwd=ROOT, env=env)
    return r


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
    model = AutoModelForCausalLM.from_pretrained(
        SNAP, dtype=torch.bfloat16, trust_remote_code=False).to("cuda").eval()
    with torch.no_grad():
        lg = model(**tok(PROMPT, return_tensors="pt").to("cuda")).logits
    hf_top = lg[0, -1].float().cpu().numpy()
    del model
    torch.cuda.empty_cache()

    wsp, wde = "/tmp/sparse_gate_sp", "/tmp/sparse_gate_de"
    rsp = run(PROMPT, 0, SMAX, wsp, ("--sparse", "--sparse-k", str(SPARSE_K)))
    print(rsp.stdout[-400:] if rsp.stdout else "", flush=True)
    check("sparse-run", rsp.returncode == 0, rsp.stderr[:300])
    if rsp.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    # serve_decode always writes /tmp/dec_prefill.npy: rescue before the
    # dense run overwrites it.
    shutil.copyfile("/tmp/dec_prefill.npy", "/tmp/sparse_gate_prefill.npy")
    rde = run(PROMPT, 0, SMAX, wde, ())
    check("dense-run", rde.returncode == 0, rde.stderr[:300])
    if rde.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)

    # sparse prefill logits vs HF (same bars as the dense decode gate)
    dec = np.load("/tmp/sparse_gate_prefill.npy")
    par = float(np.abs(dec - hf_top).max())
    check("sparse-parity", par < 10.0,
          f"maxabs={par:.2f} (structural-break detector)")
    first = int(dec.argmax())
    frank = int((hf_top > hf_top[first]).sum()) + 1
    check("sparse-first-pick-rank", frank <= 8, f"rank {frank}")

    # caches: identical computation -> bit-identical files
    same = True
    for L in range(28):
        for tag in ("ckP", "cvP"):
            a = os.path.join(wsp, f"{tag}{L}.bin")
            b = os.path.join(wde, f"{tag}{L}.bin")
            if not (os.path.isfile(a) and os.path.isfile(b)
                    and filecmp.cmp(a, b, shallow=False)):
                same = False
    check("sparse-cache-exact", same, "56/56 K/V files bit-identical")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
