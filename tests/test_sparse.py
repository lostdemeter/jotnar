"""Sparse-decode gate: band-map readout path vs HF + vs dense.

Audit band map (sharp bands top-k, broad middle dense; same K/V caches,
same weights) must speak the same distribution as HF: parity bound +
first-pick rank (same bars as the dense decode gate -- decisions,
never strings). Caches agree within eps across binaries (cublas
summation order differs); stale pad rows must be bit-invisible
same-binary (mask-floor regression gate: caught -10000 at 23.6).
SKIPs without snapshot/GPU/nvcc. Slow (~10min: builds + runs).
Usage: python3 tests/test_sparse.py
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
SMAX = 16
SPARSE_K = 8
# Shipped configuration: audit band map (sharp bands sparse).
SPARSE_BANDS = "4-11,21-27"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def run(prompt, n, smax, work, extra=()):
    env = dict(os.environ, QWEN_WORKDIR=work)
    # warm-up (untimed): first run in a fresh workdir wobbles in values
    # (GPU/fragmentation state); steady state is bit-deterministic.
    # Gates measure steady state; decisions hold in both.
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         prompt, "--n", "0", "--smax", str(smax)] + list(extra),
        capture_output=True, text=True, cwd=ROOT, env=env)
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
    spex = ("--sparse-bands", SPARSE_BANDS, "--sparse-k", str(SPARSE_K))
    rsp = run(PROMPT, 0, SMAX, wsp, spex)
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
    shutil.copyfile("/tmp/dec_prefill.npy", "/tmp/dense_gate_prefill.npy")

    # sparse prefill logits vs HF (same bars as the dense decode gate)
    dec = np.load("/tmp/sparse_gate_prefill.npy")
    par = float(np.abs(dec - hf_top).max())
    check("sparse-parity", par < 10.0,
          f"maxabs={par:.2f} (structural-break detector)")
    first = int(dec.argmax())
    frank = int((hf_top > hf_top[first]).sum()) + 1
    check("sparse-first-pick-rank", frank <= 8, f"rank {frank}")

    # caches: same Kfull/Vfull computation; cross-binary cublas
    # summation order differs at eps level (measured ~0.01), so the
    # bar is eps, not bits (bits hold same-binary: stale legs below).
    md = 0.0
    for L in range(28):
        for tag in ("ckP", "cvP"):
            a = np.fromfile(os.path.join(wsp, f"{tag}{L}.bin"),
                            dtype=np.float16).astype(np.float32)
            b = np.fromfile(os.path.join(wde, f"{tag}{L}.bin"),
                            dtype=np.float16).astype(np.float32)
            md = max(md, float(np.abs(a - b).max()))
    check("sparse-cache-agree", md < 1.0,
          f"maxabs={md:.3f} (same bar as prefill)")

    # stale-cache immunity (mask-floor regression gate): nonzero pad
    # rows must not move outputs. Same binary + steady state =>
    # bit-exact bar (this caught the -10000 mask floor: 23.6 maxabs).
    refs = {wsp: "/tmp/sparse_gate_prefill.npy",
            wde: "/tmp/dense_gate_prefill.npy"}
    for work, extra, tag in ((wsp, spex, "sparse"), (wde, (), "dense")):
        for L in range(28):
            for tg in ("ckP", "cvP"):
                p = os.path.join(work, f"{tg}{L}.bin")
                a = np.fromfile(p, dtype=np.float16).reshape(16, 512)
                a[5:] = np.float16(100.0)
                a.tofile(p)
        r = run(PROMPT, 0, SMAX, work, extra)
        check(f"{tag}-stale-run", r.returncode == 0, r.stderr[:300])
        got = np.load("/tmp/dec_prefill.npy")
        ref = np.load(refs[work])
        stale = float(np.abs(got - ref).max())
        check(f"{tag}-stale-exact", stale == 0.0,
              f"maxabs={stale:.4f} (pad rows must be invisible)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
