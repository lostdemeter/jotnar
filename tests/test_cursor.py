#!/usr/bin/env python3
"""Cursor gate: static-capacity output + runtime cursor (decoder pattern).

A loop writes bytes into a fixed-capacity buffer at a moving cursor. When
`ISCATTER1`'s output aliases its base, the emitter writes **in place**
(no full-buffer copy), so the write is O(1). Reference = C = non-FPU;
codec-proven. The loop + BREAK + cursor is the decoder skeleton.
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
            for o in outputs}, art


def main():
    work = "/tmp/jotnar_cursor"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN BUF\nIN CUR\nIN N\nREPEAT N\n"
            "B = IADD(CUR, 65)\n"
            "BUF = ISCATTER1(BUF, CUR, B)\n"
            "CUR = IADD(CUR, 1)\n"
            "END\n")
    pay = {"BUF": np.zeros(8, dtype=np.int64),
           "CUR": np.array([0], dtype=np.int64),
           "N": np.array([5], dtype=np.int64)}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("cursor-ref-buf", ref["BUF"].tolist() == [65, 66, 67, 68, 69, 0, 0, 0],
          str(ref["BUF"].tolist()))
    check("cursor-ref-cur", int(ref["CUR"][0]) == 5, str(ref["CUR"].tolist()))

    for target in ("c", "nonfpu"):
        got, _ = _run(work, text, pay, ["BUF", "CUR"], f"cur_{target}", target)
        check(f"cursor-{target}",
              got["BUF"].tolist() == [65, 66, 67, 68, 69, 0, 0, 0]
              and int(got["CUR"][0]) == 5,
              f"{got['BUF'].tolist()} cur={got['CUR'].tolist()}")

    got, art = _run(work, text, pay, ["BUF", "CUR"], "cur_inplace")
    check("cursor-inplace", "in-place" in art["source"],
          "ISCATTER1 out aliases base")

    # loop + BREAK + cursor: write one byte, then stop
    stop_text = ("IN BUF\nIN CUR\nIN N\nIN STOP\nREPEAT N\n"
                 "B = IADD(CUR, 65)\n"
                 "BUF = ISCATTER1(BUF, CUR, B)\n"
                 "CUR = IADD(CUR, 1)\n"
                 "BREAK STOP\nEND\n")
    spay = {"BUF": np.zeros(8, dtype=np.int64),
            "CUR": np.array([0], dtype=np.int64),
            "N": np.array([5], dtype=np.int64),
            "STOP": np.array([1], dtype=np.int64)}
    rs = ASM.run_text(stop_text, REGISTRY, spay, sigs=SIGS)
    check("cursor-break", rs["BUF"].tolist() == [65, 0, 0, 0, 0, 0, 0, 0]
          and int(rs["CUR"][0]) == 1, str(rs["BUF"].tolist()))

    # codec: BUF and CUR are shadowed inputs (read-write)
    art = compile_program(text, "c", sample=pay, outputs=["BUF", "CUR"],
                          mode="codec")
    so = build_shared(art["source"], work, name="cur_codec",
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
    BUF = np.zeros(8, dtype=np.int64)
    CUR = np.array([0], dtype=np.int64)
    N = np.array([5], dtype=np.int64)
    lib.jotnar_codec_bind(h, b"BUF", BUF.ctypes.data_as(ctypes.c_void_p), 64, 1)
    lib.jotnar_codec_bind(h, b"CUR", CUR.ctypes.data_as(ctypes.c_void_p), 8, 1)
    lib.jotnar_codec_bind(h, b"N", N.ctypes.data_as(ctypes.c_void_p), 8, 0)
    step = lib.jotnar_codec_step(h)
    check("cursor-codec",
          step == 0 and BUF.tolist() == [65, 66, 67, 68, 69, 0, 0, 0]
          and int(CUR[0]) == 5,
          f"step={step} BUF={BUF.tolist()} CUR={CUR.tolist()}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
