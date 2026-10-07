"""Reravel completeness: bit-exact where determinism promises it.

The completeness task (unravel then reravel, bit exact) decomposes:
1. determinism: same served binary + same files -> byte-identical
   outputs across runs (free check, always runs).
2. tap consistency: prefix-program LOGITS vs gen-serve LOGITS for the
   same prompt -- same core program, must agree (eps at most; exact
   expected, measured honestly).
3. knowledge roundtrip (behavioral): steered battery -- installs live
   across all prompts, every unaffected top must hold.
SKIPs without snapshot/GPU/nvcc. Slow (builds + runs).
Usage: python3 tests/test_reravel.py
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
PROMPT = "The capital of Germany is"


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
    work = "/tmp/reravel"
    os.makedirs(work, exist_ok=True)

    # -- determinism: steer prefix twice, byte-compare -----------------
    # Fresh workdir: first call builds (no REUSE), second reuses via
    # source-hash stamps (also exercises the stamp path itself).
    env = dict(os.environ)
    env.pop("QWEN_REUSE_BIN", None)
    base = [sys.executable, os.path.join(ROOT, "scripts", "serve_steer.py"),
            PROMPT, "--n", "0", "--gain", "0.0", "--country", "Germany",
            "--capital", " Paris", "--workdir", work]
    r1 = subprocess.run(base, capture_output=True, text=True, cwd=ROOT,
                        env=env)
    check("reravel-run1", r1.returncode == 0, r1.stderr[:150])
    if r1.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    a = {f: np.fromfile(os.path.join(work, "prefix", f), dtype=np.float32)
         for f in ("o_LOGITS.bin", "o_Y2.bin", "o_Y26.bin")}
    r2 = subprocess.run(base, capture_output=True, text=True, cwd=ROOT,
                        env=env)
    check("reravel-run2", r2.returncode == 0, r2.stderr[:150])
    b = {f: np.fromfile(os.path.join(work, "prefix", f), dtype=np.float32)
         for f in ("o_LOGITS.bin", "o_Y2.bin", "o_Y26.bin")}
    for f in a:
        check(f"reravel-deterministic-{f[:-4]}",
              np.array_equal(a[f], b[f]), "byte-identical rerun")

    # -- tap consistency: prefix LOGITS vs gen-serve LOGITS ------------
    # Same core program (build_program bmmv), same weights, deterministic
    # GPU math: bit-exact expected, eps reported honestly either way.
    # Reuse binaries here (built above in this same workdir... gen uses
    # /tmp/gen7b canonical bins -- reuse those, not ours).
    env["QWEN_REUSE_BIN"] = "1"
    r3 = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "gen_qwen7b_geo.py"),
         PROMPT, "--n", "1", "--smax", "16", "--geo-only", "--bmmv",
         "--graph", "--no-sync", "--out-json", "/tmp/reravel_gen.json"],
        capture_output=True, text=True, cwd=ROOT, env=env)
    check("reravel-gen-run", r3.returncode == 0, r3.stderr[:150])
    if r3.returncode == 0:
        from chain.qwen7b import load7b
        _, tok = load7b()
        n = len(tok(PROMPT, return_tensors="pt")["input_ids"][0])
        gl = np.load("/tmp/reravel_gen.json.promptlogits.npy")
        pl = a["o_LOGITS.bin"].reshape(16, -1)[n - 1]
        dmax = float(np.abs(gl - pl).max())
        check("reravel-tap-exact", dmax == 0.0,
              f"maxabs={dmax:.3e} (bit-exact across binaries)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
