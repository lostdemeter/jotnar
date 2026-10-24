#!/usr/bin/env python3
"""IBITWISE gate: integer bitwise family (IXOR/IAND/IOR/ISHL/ISHR).

The arithmetic half of the algebraic path:
  * reference eval exact, 64-bit wrap, logical (zero-fill) right shift;
  * C and non-FPU agree bit-for-bit, non-FPU stays float-free;
  * codec mode decodes a deflate block header and runs a CRC-32 step
    (IGATHER1 table + XOR/AND/SHR) end to end;
  * shift range, kind, rank, and literal failures are loud.
Usage: python3 tests/test_ibitwise.py (needs cc).
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
from chain.asm_ops import (  # noqa: E402
    REGISTRY,
    SIGS,
    op_iand,
    op_ior,
    op_ishl,
    op_ishr,
    op_ixor,
)
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


def _crc_table():
    poly = 0xEDB88320
    table = []
    for n in range(256):
        c = n
        for _ in range(8):
            c = (c >> 1) ^ poly if (c & 1) else c >> 1
        table.append(c)
    return table


def main():
    work = "/tmp/jotnar_ibitwise"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN A\nIN B\n"
            "X = IXOR(A, B)\n"
            "N = IAND(A, 255)\n"
            "O = IOR(A, B)\n"
            "L = ISHL(A, 3)\n"
            "R = ISHR(A, 2)\n")
    A = np.array([0b10110011, 255, 0, 1], dtype=np.int64)
    B = np.array([0b01001100, 15, 255, 0], dtype=np.int64)
    pay = {"A": A, "B": B}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("ibitwise-ref-xor", bool((ref["X"] == (A ^ B)).all()),
          str(ref["X"].tolist()))
    check("ibitwise-ref-and", bool((ref["N"] == (A & 255)).all()))
    check("ibitwise-ref-or", bool((ref["O"] == (A | B)).all()))
    check("ibitwise-ref-shl", bool((ref["L"] == (A << 3)).all()))
    check("ibitwise-ref-shr", bool((ref["R"] == (A >> 2)).all()))

    names = ["X", "N", "O", "L", "R"]
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"ibw_{target}", target)
        check(f"ibitwise-{target}-match",
              all(bool((got[n] == ref[n]).all()) for n in names),
              ",".join(str(got[n].tolist()) for n in names))

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("ibitwise-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("ibitwise-nonfpu-trap-clean", False, str(e)[:140])

    # loud: reference
    for tag, fn, args, want in (
            ("loud-ref-shl-range", op_ishl, (A, 64), "0..63"),
            ("loud-ref-shr-negative", op_ishr, (A, -1), "0..63"),
            ("loud-ref-xor-float", op_ixor, (A.astype(np.float64), B),
             "must be integer"),
            ("loud-ref-and-shape", op_iand, (A.reshape(2, 2), B), "shape"),
            ("loud-ref-or-float", op_ior, (A, 1.5), "integer literal")):
        try:
            fn(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time
    for tag, prog, want in (
            ("loud-c-shl-range", "IN A\nS = ISHL(A, 64)\n", "0..63"),
            ("loud-c-shr-range", "IN A\nS = ISHR(A, 99)\n", "0..63"),
            ("loud-c-xor-float", "IN A\nS = IXOR(A, 1.5)\n", "literal"),
            ("loud-c-xor-2d", "IN A\nIN B\nS = IXOR(A, B)\n", "shape")):
        sample = {"A": A} if "IN B" not in prog else {
            "A": A, "B": B.reshape(2, 2)}
        try:
            compile_program(prog, "c", sample=sample, outputs=["S"])
            check(tag, False, "accepted")
        except (NoPattern, ASM.AsmError, ValueError) as e:
            check(tag, want in str(e), str(e)[:70])

    # codec: deflate block header (BFINAL, BTYPE) from the first byte
    art = compile_program(
        "IN E\nB0 = IASSEMBLE(E, 0, 1)\n"
        "F = IAND(B0, 1)\nS = ISHR(B0, 1)\nT = IAND(S, 3)\n",
        "c", sample={"E": np.zeros(8, dtype=np.int64)},
        outputs=["F", "T"], mode="codec")
    so = build_shared(art["source"], work, name="ibw_deflate",
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

    E = np.zeros(8, dtype=np.int64)
    E[0] = 0x03  # BFINAL=1, BTYPE=01 (fixed Huffman)
    F = np.zeros(1, dtype=np.int64)
    T = np.zeros(1, dtype=np.int64)
    bind("E", E, 0)
    bind("F", F, 2)
    bind("T", T, 2)
    step = lib.jotnar_codec_step(h)
    check("ibitwise-codec-deflate-header",
          step == 0 and int(F[0]) == 1 and int(T[0]) == 1,
          f"step={step} F={F.tolist()} T={T.tolist()}")
    lib.jotnar_codec_destroy(h)

    # codec: one CRC-32 step (table + XOR/AND/SHR) -- the compressed-data kernel
    table = np.array(_crc_table(), dtype=np.int64)
    art = compile_program(
        "IN T\nIN C\nIN B\n"
        "XB = IXOR(C, B)\n"
        "IDX = IAND(XB, 255)\n"
        "T0 = IGATHER1(T, IDX)\n"
        "SH = ISHR(C, 8)\n"
        "C2 = IXOR(T0, SH)\n",
        "c", sample={"T": table, "C": np.zeros(1, dtype=np.int64),
                     "B": np.zeros(1, dtype=np.int64)},
        outputs=["C2"], mode="codec")
    so = build_shared(art["source"], work, name="ibw_crc", libs=art.get("libs"))
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

    def bind2(name, arr, mode):
        return lib.jotnar_codec_bind(
            h, name.encode(), arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(arr.nbytes), ctypes.c_int(mode))

    crc0 = 0xFFFFFFFF
    byte = 0x41
    exp = _crc_table()[(crc0 ^ byte) & 0xFF] ^ (crc0 >> 8)
    C = np.array([crc0], dtype=np.int64)
    Bv = np.array([byte], dtype=np.int64)
    C2 = np.zeros(1, dtype=np.int64)
    bind2("T", table, 0)
    bind2("C", C, 0)
    bind2("B", Bv, 0)
    bind2("C2", C2, 2)
    step = lib.jotnar_codec_step(h)
    check("ibitwise-codec-crc32-step",
          step == 0 and int(C2[0]) == exp,
          f"step={step} C2={C2.tolist()} want={exp}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
