#!/usr/bin/env python3
"""Runtime REPEAT gate: `REPEAT n ... END` with n a stream (compiled loop).

Static geometry, dynamic trajectory: the body is emitted once inside a C
`for`; n is a runtime scalar; all buffer shapes stay fixed. Reference = C =
non-FPU; codec runs it end to end; a body that changes a stream's shape
fails loud.
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
        dt = np.float64 if pay[n].dtype.kind == "f" else np.int64
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(pay[n], dtype=dt).tofile(fn)
        files.append(fn)
    outs = [os.path.join(work, f"out_{o}.bin") for o in outputs]
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:300]}")
    return {o: np.fromfile(os.path.join(work, f"out_{o}.bin"), dtype=np.int64)
            for o in outputs}


def main():
    work = "/tmp/jotnar_repeat_rt"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = "IN X\nIN N\nREPEAT N\nX = IADD(X, 1)\nEND\n"
    pay = {"X": np.array([10], dtype=np.int64),
           "N": np.array([5], dtype=np.int64)}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("rtr-ref", int(ref["X"][0]) == 15, str(ref["X"].tolist()))

    for target in ("c", "nonfpu"):
        got = _run(work, text, pay, ["X"], f"rtr_{target}", target)
        check(f"rtr-{target}-match", int(got["X"][0]) == 15,
              str(got["X"].tolist()))

    # nested runtime loops: N x M increments from 0
    nested = ("IN X\nIN N\nIN M\nREPEAT N\nREPEAT M\nX = IADD(X, 1)\n"
              "END\nEND\n")
    pn = {"X": np.array([0], dtype=np.int64),
          "N": np.array([3], dtype=np.int64),
          "M": np.array([4], dtype=np.int64)}
    rn = ASM.run_text(nested, REGISTRY, pn, sigs=SIGS)
    check("rtr-nested", int(rn["X"][0]) == 12, str(rn["X"].tolist()))

    # codec: shadowed input X (read-write), N frozen
    art = compile_program(text, "c", sample=pay, outputs=["X"], mode="codec")
    so = build_shared(art["source"], work, name="rtr_codec",
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
    N = np.array([5], dtype=np.int64)
    lib.jotnar_codec_bind(h, b"X", X.ctypes.data_as(ctypes.c_void_p), 8, 1)
    lib.jotnar_codec_bind(h, b"N", N.ctypes.data_as(ctypes.c_void_p), 8, 0)
    step = lib.jotnar_codec_step(h)
    check("rtr-codec", step == 0 and int(X[0]) == 15,
          f"step={step} X={X.tolist()}")
    lib.jotnar_codec_destroy(h)

    # loud: a body that changes a stream's geometry
    bad = "IN X\nIN N\nREPEAT N\nX = IWINDOW(X, 0, 1)\nEND\n"
    pb = {"X": np.arange(4, dtype=np.int64),
          "N": np.array([2], dtype=np.int64)}
    try:
        compile_program(bad, "c", sample=pb, outputs=["X"])
        check("rtr-shape-invariant", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as e:
        check("rtr-shape-invariant", "geometry" in str(e), str(e)[:80])

    # loud: negative runtime count
    neg = "IN X\nIN N\nREPEAT N\nX = IADD(X, 1)\nEND\n"
    pneg = {"X": np.array([0], dtype=np.int64),
            "N": np.array([-1], dtype=np.int64)}
    try:
        ASM.run_text(neg, REGISTRY, pneg, sigs=SIGS)
        check("rtr-negative", False, "accepted")
    except ASM.AsmError as e:
        check("rtr-negative", "negative" in str(e), str(e)[:60])

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
