#!/usr/bin/env python3
"""IGATHER1 gate: 1D integer gather (non-contiguous format reads).

The companion to IWINDOW:

  * reference eval is an exact numpy gather, ids bounds-checked loud;
  * C and non-FPU one-shot agree with the reference bit-for-bit;
  * non-FPU source stays float-free;
  * codec mode drives a non-contiguous byte gather end to end, and a
    bad index fails loud at runtime (step returns 4);
  * bad table/ids kind, rank, and out-of-range ids fail loud.
Usage: python3 tests/test_igather1.py (needs cc).
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
from chain.asm_ops import REGISTRY, SIGS, op_igather1  # noqa: E402
from chain.emit_c import NoPattern, build, build_shared, compile_program  # noqa: E402
from chain.emit_nonfpu import float_trap  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _write_inputs(work, art, payload):
    files = []
    for n in art["inputs"]:
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
        files.append(fn)
    return files


def _run_art(work, art, payload, name, target):
    if target == "nonfpu":
        float_trap(art["source"])
        exe = build(art["source"], work, name=name, libs=[])
    else:
        exe = build(art["source"], work, name=name)
    files = _write_inputs(work, art, payload)
    outs = [os.path.join(work, f"out_{o}.bin") for o in art["outputs"]]
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:400]}")
    out = {}
    for o in art["outputs"]:
        out[o] = np.fromfile(os.path.join(work, f"out_{o}.bin"),
                             dtype=np.int64)
    return out


def main():
    work = "/tmp/jotnar_igather1"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN T\nIN OFF\n"
            "B = IGATHER1(T, OFF)\n"
            "C = IADD(B, 1)\n")
    T = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)
    off = np.array([5, 0, 3, 3], dtype=np.int64)
    pay = {"T": T, "OFF": off}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    names = ["B", "C"]
    check("igather1-ref", bool((ref["B"] == T[off]).all())
          and bool((ref["C"] == T[off] + 1).all()),
          f"B={ref['B'].tolist()}")

    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"ig_{target}", target)
        check(f"igather1-{target}-match",
              all(bool((got[n] == ref[n]).all()) for n in names),
              ",".join(str(got[n].tolist()) for n in names))

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("igather1-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("igather1-nonfpu-trap-clean", False, str(e)[:140])

    # loud: reference kind/rank/bounds
    for tag, args, want in (
            ("loud-ref-oob", (T, np.array([6], dtype=np.int64)), "out of range"),
            ("loud-ref-negative", (T, np.array([-1], dtype=np.int64)), "out of range"),
            ("loud-ref-2d-table", (T.reshape(2, 3), off), "must be 1D"),
            ("loud-ref-2d-ids", (T, off.reshape(2, 2)), "must be 1D"),
            ("loud-ref-float", (T.astype(np.float64), off), "must be integer")):
        try:
            op_igather1(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time rank
    try:
        compile_program("IN T\nIN OFF\nB = IGATHER1(T, OFF)\n", "c",
                        sample={"T": T.reshape(2, 3), "OFF": off},
                        outputs=["B"])
        check("loud-c-2d-table", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as e:
        check("loud-c-2d-table", "1D I table" in str(e), str(e)[:70])
    try:
        compile_program("IN T\nIN OFF\nB = IGATHER1(T, OFF)\n", "c",
                        sample={"T": T, "OFF": off.reshape(2, 2)},
                        outputs=["B"])
        check("loud-c-2d-ids", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as e:
        check("loud-c-2d-ids", "1D I ids" in str(e), str(e)[:70])

    # codec mode: gather non-contiguous bytes end to end.
    art = compile_program(text, "c", sample={"T": np.zeros(8, dtype=np.int64),
                                             "OFF": np.zeros(4, dtype=np.int64)},
                          outputs=["B"], mode="codec")
    so = build_shared(art["source"], work, name="ig_codec", libs=art.get("libs"))
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

    E = np.arange(8, dtype=np.int64)
    OFF = np.array([7, 2, 5, 0], dtype=np.int64)
    B = np.zeros(4, dtype=np.int64)
    rc = [bind("T", E, 0), bind("OFF", OFF, 0), bind("B", B, 2)]
    step = lib.jotnar_codec_step(h)
    check("igather1-codec-gather",
          step == 0 and bool((B == E[OFF]).all()),
          f"rc={rc} step={step} B={B.tolist()} want={E[OFF].tolist()}")
    # runtime bounds: bad index must fail loud (rc 4)
    OFF[:] = [0, 1, 2, 8]
    rc_bad = lib.jotnar_codec_step(h)
    check("igather1-codec-oob-loud", rc_bad == 4, f"rc={rc_bad}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
