"""Prefill gate: parallel caches agree with sequential; TTFT is warm.

Parallel prefill (one full forward emitting rotated-K/V groups) must
match sequential decode-prefill caches within eps (different matmul
shapes => different fp order, so eps not bits), generation from
prefilled caches must equal sequential generation exactly
(deterministic binaries), and the forward must stay warm-fast (bars
the cold one-shot regression: 19s).
SKIPs without snapshot/GPU/nvcc. Slow (~8min: two builds + runs).
Usage: python3 tests/test_prefill.py
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []
PROMPT = ["The", "capital", "of", "France", "is"]
N_PROMPT = 5


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import torch
    from chain.qwen7b import SNAP
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)

    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_prefill.py"),
         "The capital of France is", "--smax", "16",
         "--outdir", "/tmp/prefill_gate"],
        capture_output=True, text=True, cwd=ROOT)
    dt_pre = time.perf_counter() - t0
    check("prefill-run", r.returncode == 0, r.stderr[:200])
    if r.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    wall = None
    for ln in r.stdout.splitlines():
        if ln.startswith("prefill forward:"):
            wall = float(ln.split()[2].replace("ms", ""))
    check("prefill-warm", wall is not None and wall < 5000,
          f"{wall}ms (cold one-shot took 19000ms)")

    # sequential caches for the same prompt (decode --n 0 prefills only).
    # Fresh build here (hermetic); later runs reuse via the stamp.
    r2 = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         "The capital of France is", "--n", "0", "--smax", "16"],
        capture_output=True, text=True, cwd=ROOT)
    check("decode-seq-run", r2.returncode == 0, r2.stderr[:200])
    md = 0.0
    for L in range(28):
        for tag in ("ckP", "cvP"):
            a = np.fromfile(f"/tmp/prefill_gate/{tag}{L}.bin",
                            dtype=np.float16).astype(np.float32).reshape(16, 512)[:N_PROMPT]
            b = np.fromfile(f"/tmp/dec7b/{tag}{L}.bin",
                            dtype=np.float16).astype(np.float32).reshape(16, 512)[:N_PROMPT]
            md = max(md, float(np.abs(a - b).max()))
    check("prefill-cache-agree", md < 1.0,
          f"maxabs={md:.3f} (fp-order eps, values ~165)")

    # generation from prefilled caches == sequential generation (exact:
    # deterministic binaries; a flip is data, investigate, don't loosen)
    re = dict(os.environ, QWEN_REUSE_BIN="1")
    g1 = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         "The capital of France is", "--n", "5", "--smax", "16",
         "--prefill-dir", "/tmp/prefill_gate"],
        capture_output=True, text=True, cwd=ROOT, env=re)
    t1 = [ln for ln in g1.stdout.splitlines() if ln.startswith("DECODED:")]
    g2 = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "serve_decode.py"),
         "The capital of France is", "--n", "5", "--smax", "16"],
        capture_output=True, text=True, cwd=ROOT, env=re)
    t2 = [ln for ln in g2.stdout.splitlines() if ln.startswith("DECODED:")]
    check("prefill-gen-run", g1.returncode == 0 and g2.returncode == 0,
          f"{g1.returncode}/{g2.returncode}")
    if t1 and t2:
        check("prefill-gen-equal", t1[0] == t2[0], f"{t1[0][:60]!r}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
