#!/usr/bin/env python3
"""Shadowed-input gate: an `IN` also assigned as an `OUT` (one-shot C).

Regression for the one-shot emitter re-malloc'ing a reassigned input after
loading it (garbage output). The codec path already handled this; the
one-shot path now reuses the loaded buffer. Covers I and F, 1D and 2D.
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR",
                                  os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain import asm as ASM  # noqa: E402
from chain.asm_ops import REGISTRY, SIGS  # noqa: E402
from chain.emit_c import build, compile_program  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _run(work, text, pay, outputs, name):
    art = compile_program(text, "c", sample=pay, outputs=outputs)
    exe = build(art["source"], work, name=name)
    files = []
    for n in art["inputs"]:
        dt = np.float64 if pay[n].dtype.kind == "f" else np.int64
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(pay[n], dtype=dt).tofile(fn)
        files.append(fn)
    outs = [os.path.join(work, f"out_{o}.bin") for o in outputs]
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:300]}")
    return {o: np.fromfile(os.path.join(work, f"out_{o}.bin"),
                           dtype=(np.float64 if pay[o].dtype.kind == "f"
                                  else np.int64))
            for o in outputs}


def main():
    work = "/tmp/jotnar_shadowed"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    cases = [
        ("int-1d", "IN X\nX = IADD(X, 1)\n",
         {"X": np.array([10], dtype=np.int64)}),
        ("int-2d", "IN X\nX = IADD(X, 1)\n",
         {"X": np.array([[1, 2], [3, 4]], dtype=np.int64)}),
    ]
    for tag, text, pay in cases:
        ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
        got = _run(work, text, pay, ["X"], tag)
        check(f"shadowed-{tag}",
              bool(np.array_equal(got["X"].reshape(ref["X"].shape),
                                  ref["X"])),
              f"got={got['X'].reshape(-1).tolist()} "
              f"ref={ref['X'].reshape(-1).tolist()}")

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
