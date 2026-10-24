#!/usr/bin/env python3
"""BREAK gate: conditional early exit from the innermost REPEAT.

`BREAK cond` exits when cond (a 1-element I stream or literal) is nonzero.
Reference = C = non-FPU; codec runs it end to end; a literal-count REPEAT
with BREAK compiles as a loop (not an unroll); BREAK outside a loop and a
non-1-element condition fail loud.
"""
import ctypes
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
from chain.emit_c import (  # noqa: E402
    NoPattern,
    build,
    build_shared,
    compile_program,
)
from chain.emit_nonfpu import float_trap  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _run(work, text, pay, outputs, name, target="c"):
    art = compile_program(text, target, sample=pay, outputs=outputs)
    if target == "nonfpu":
        float_trap(art["source"])
        exe = build(art["source"], work, name=name, libs=[])
    else:
        exe = build(art["source"], work, name=name)
    files = []
    for n in art["inputs"]:
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(pay[n], dtype=np.int64).tofile(fn)
        files.append(fn)
    outs = [os.path.join(work, f"out_{o}.bin") for o in outputs]
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:300]}")
    return {o: np.fromfile(os.path.join(work, f"out_{o}.bin"), dtype=np.int64)
            for o in outputs}


def main():
    work = "/tmp/jotnar_break"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN X\nIN N\nIN STOP\nREPEAT N\nX = IADD(X, 1)\n"
            "BREAK STOP\nEND\n")

    def pay(stop):
        return {"X": np.array([10], dtype=np.int64),
                "N": np.array([5], dtype=np.int64),
                "STOP": np.array([stop], dtype=np.int64)}

    check("break-ref-stop",
          int(ASM.run_text(text, REGISTRY, pay(1), sigs=SIGS)["X"][0]) == 11)
    check("break-ref-continue",
          int(ASM.run_text(text, REGISTRY, pay(0), sigs=SIGS)["X"][0]) == 15)

    for target in ("c", "nonfpu"):
        got1 = _run(work, text, pay(1), ["X"], f"brk1_{target}", target)
        got0 = _run(work, text, pay(0), ["X"], f"brk0_{target}", target)
        check(f"break-{target}",
              int(got1["X"][0]) == 11 and int(got0["X"][0]) == 15,
              f"stop={got1['X'].tolist()} cont={got0['X'].tolist()}")

    # a literal-count REPEAT with BREAK must be a loop, not an unroll
    lit = "IN X\nIN STOP\nREPEAT 5\nX = IADD(X, 1)\nBREAK STOP\nEND\n"
    pl = {"X": np.array([10], dtype=np.int64),
          "STOP": np.array([1], dtype=np.int64)}
    art = compile_program(lit, "c", sample=pl, outputs=["X"])
    check("break-literal-is-loop",
          "for (int64_t _it" in art["source"] and "break;" in art["source"])
    got = _run(work, lit, pl, ["X"], "brk_lit")
    check("break-literal", int(got["X"][0]) == 11, str(got["X"].tolist()))

    # codec
    art = compile_program(lit, "c", sample=pl, outputs=["X"], mode="codec")
    so = build_shared(art["source"], work, name="brk_codec",
                      libs=art.get("libs"))
    lib = ctypes.CDLL(so)
    lib.jotnar_codec_create.restype = ctypes.c_void_p
    lib.jotnar_codec_bind.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_int]
    lib.jotnar_codec_bind.restype = ctypes.c_int
    lib.jotnar_codec_step.argtypes = [ctypes.c_void_p]
    lib.jotnar_codec_step.restype = ctypes.c_int
    lib.jotnar_codec_destroy.argtypes = [ctypes.c_void_p]
    h = lib.jotnar_codec_create()
    X = np.array([10], dtype=np.int64)
    S = np.array([1], dtype=np.int64)
    lib.jotnar_codec_bind(h, b"X", X.ctypes.data_as(ctypes.c_void_p), 8, 1)
    lib.jotnar_codec_bind(h, b"STOP", S.ctypes.data_as(ctypes.c_void_p), 8, 0)
    step = lib.jotnar_codec_step(h)
    check("break-codec", step == 0 and int(X[0]) == 11,
          f"step={step} X={X.tolist()}")
    lib.jotnar_codec_destroy(h)

    # loud: BREAK outside REPEAT
    try:
        ASM.run_text("IN C\nBREAK C\n", REGISTRY,
                     {"C": np.array([0], dtype=np.int64)}, sigs=SIGS)
        check("break-outside", False, "accepted")
    except ASM.AsmError as e:
        check("break-outside", "outside REPEAT" in str(e), str(e)[:60])

    # loud: condition must be a 1-element I
    bad = ("IN X\nIN C\nREPEAT 2\nX = IADD(X, 1)\nBREAK C\nEND\n")
    pb = {"X": np.array([0], dtype=np.int64),
          "C": np.array([0, 0], dtype=np.int64)}
    try:
        compile_program(bad, "c", sample=pb, outputs=["X"])
        check("break-cond-shape", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as e:
        check("break-cond-shape", "1-element I" in str(e), str(e)[:70])

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
