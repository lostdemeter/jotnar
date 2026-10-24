#!/usr/bin/env python3
"""REPEAT gate: static `REPEAT n ... END` unroll.

Static geometry: the body is copied n times, so shapes, the codec ABI, and
the emitters are unchanged. Reference = C = non-FPU; codec mode runs it end
to end; non-integral / missing-END / stray-END / negative counts fail loud.
Runtime trip counts use `ASM.repeat()` (host-driven STATE threading).
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
from chain.emit_c import build, build_shared, compile_program  # noqa: E402
from chain.emit_nonfpu import float_trap  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _run_art(work, art, payload, name, target):
    if target == "nonfpu":
        float_trap(art["source"])
        exe = build(art["source"], work, name=name, libs=[])
    else:
        exe = build(art["source"], work, name=name)
    files = []
    for n in art["inputs"]:
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
        files.append(fn)
    outs = [os.path.join(work, f"out_{o}.bin") for o in art["outputs"]]
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:400]}")
    return {o: np.fromfile(os.path.join(work, f"out_{o}.bin"), dtype=np.int64)
            for o in art["outputs"]}


def main():
    work = "/tmp/jotnar_repeat"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN X\nY = IADD(X, 0)\nREPEAT 3\nY = IADD(Y, 1)\nEND\n")
    pay = {"X": np.array([10], dtype=np.int64)}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("repeat-ref", int(ref["Y"][0]) == 13, str(ref["Y"].tolist()))

    names = ["Y"]
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"rep_{target}", target)
        check(f"repeat-{target}-match", int(got["Y"][0]) == 13,
              str(got["Y"].tolist()))
    art = compile_program(text, "c", sample=pay, outputs=names)
    check("repeat-unrolled", art["source"].count("IADD") >= 4,
          str(art["source"].count("IADD")))

    # nested loops: 2 x 3 increments from 0 -> 6
    nested = ("IN X\nY = IADD(X, 0)\n"
              "REPEAT 2\nREPEAT 3\nY = IADD(Y, 1)\nEND\nEND\n")
    pn = {"X": np.array([0], dtype=np.int64)}
    rn = ASM.run_text(nested, REGISTRY, pn, sigs=SIGS)
    check("repeat-nested", int(rn["Y"][0]) == 6, str(rn["Y"].tolist()))

    # codec: X in, Y out
    ctext = ("IN X\nY = IADD(X, 0)\nREPEAT 3\nY = IADD(Y, 1)\nEND\n")
    cpay = {"X": np.array([10], dtype=np.int64)}
    art = compile_program(ctext, "c", sample=cpay, outputs=["Y"],
                          mode="codec")
    so = build_shared(art["source"], work, name="rep_codec",
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
    Y = np.zeros(1, dtype=np.int64)
    lib.jotnar_codec_bind(h, b"X", X.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(X.nbytes), 0)
    lib.jotnar_codec_bind(h, b"Y", Y.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(Y.nbytes), 2)
    step = lib.jotnar_codec_step(h)
    check("repeat-codec", step == 0 and int(Y[0]) == 13,
          f"step={step} Y={Y.tolist()}")
    lib.jotnar_codec_destroy(h)

    # loud
    for tag, prog, want in (
            ("loud-frac", "IN X\nREPEAT 3.5\nX = IADD(X, 1)\nEND\n",
             "integral literal"),
            ("loud-missing-end", "IN X\nREPEAT 3\nX = IADD(X, 1)\n",
             "without END"),
            ("loud-stray-end", "IN X\nX = IADD(X, 1)\nEND\n",
             "END without REPEAT"),
            ("loud-negative", "IN X\nREPEAT -1\nX = IADD(X, 1)\nEND\n",
             ">= 0")):
        try:
            ASM.run_text(prog, REGISTRY, {"X": np.array([0], np.int64)},
                         sigs=SIGS)
            check(tag, False, "accepted")
        except ASM.AsmError as e:
            check(tag, want in str(e), str(e)[:70])

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
