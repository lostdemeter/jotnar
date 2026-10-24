#!/usr/bin/env python3
"""ISLICE2 gate: 2D integer slice (SQLite pages / bitmaps).

`SLICE` now handles 2D `I` as well as 2D `F`/`T`; 1D `I` windows stay on
`IWINDOW`. Reference = C = non-FPU; codec mode slices pages end to end;
bad rank/axis/bounds fail loud.
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
from chain.asm_ops import REGISTRY, SIGS, op_slice  # noqa: E402
from chain.emit_c import NoPattern, build, build_shared, compile_program  # noqa: E402
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
    work = "/tmp/jotnar_islice2"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    T = np.arange(20, dtype=np.int64).reshape(4, 5)
    text = ("IN T\n"
            "A = SLICE(T, 0, 1, 3)\n"
            "B = SLICE(T, 1, 1, 4)\n")
    pay = {"T": T}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("islice2-ref-ax0", bool((ref["A"] == T[1:3, :]).all()),
          str(ref["A"].tolist()))
    check("islice2-ref-ax1", bool((ref["B"] == T[:, 1:4]).all()),
          str(ref["B"].tolist()))

    names = ["A", "B"]
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"isl2_{target}", target)
        check(f"islice2-{target}-match",
              bool((got["A"].reshape(ref["A"].shape) == ref["A"]).all())
              and bool((got["B"].reshape(ref["B"].shape) == ref["B"]).all()),
              f"{got['A'].size}/{got['B'].size}")

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("islice2-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("islice2-nonfpu-trap-clean", False, str(e)[:140])

    # loud: reference (bad axis / out of range)
    for tag, args, want in (
            ("loud-ref-axis", (T, 2, 0, 1), "bad axis"),
            ("loud-ref-oob", (T, 0, 0, 9), "out of range")):
        try:
            op_slice(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time (rank / bounds)
    for tag, prog, sample, want in (
            ("loud-c-3d", "IN T\nA = SLICE(T, 0, 0, 1)\n",
             {"T": np.zeros((2, 2, 2), dtype=np.int64)}, "2D"),
            ("loud-c-1d", "IN T\nA = SLICE(T, 0, 0, 1)\n",
             {"T": np.arange(4, dtype=np.int64)}, "2D"),
            ("loud-c-oob", "IN T\nA = SLICE(T, 0, 0, 9)\n",
             {"T": T}, "of dim")):
        try:
            compile_program(prog, "c", sample=sample, outputs=["A"])
            check(tag, False, "accepted")
        except (NoPattern, ASM.AsmError, ValueError) as e:
            check(tag, want in str(e), str(e)[:70])

    # codec: slice integer pages end to end
    art = compile_program(text, "c", sample=pay, outputs=names, mode="codec")
    so = build_shared(art["source"], work, name="isl2_codec",
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

    def bind(name, arr, mode):
        return lib.jotnar_codec_bind(
            h, name.encode(), arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(arr.nbytes), ctypes.c_int(mode))

    A = np.zeros((2, 5), dtype=np.int64)
    B = np.zeros((4, 3), dtype=np.int64)
    bind("T", T, 0)
    bind("A", A, 2)
    bind("B", B, 2)
    step = lib.jotnar_codec_step(h)
    check("islice2-codec",
          step == 0 and bool((A == T[1:3, :]).all())
          and bool((B == T[:, 1:4]).all()),
          f"step={step} A={A.tolist()} B={B.tolist()}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
