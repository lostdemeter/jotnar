#!/usr/bin/env python3
"""IASSEMBLE / ISUM / ISCATTER1 gate: the rest of the byte-index family.

  * reference eval exact (LE assembly, window sum, 1D scatter write);
  * C and non-FPU one-shot agree bit-for-bit, non-FPU stays float-free;
  * codec mode drives a LE field decode and a header rebuild end to end,
    and a bad scatter index fails loud (step returns 4);
  * kind/rank/bounds failures are loud.
Usage: python3 tests/test_ibyte.py (needs cc).
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
    op_iassemble,
    op_iscatter1,
    op_isum,
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


def main():
    work = "/tmp/jotnar_ibyte"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN E\nIN OFF\nIN V\n"
            "A = IASSEMBLE(E, 12, 4)\n"
            "S = ISUM(E, 0, 4)\n"
            "H = ISCATTER1(E, OFF, V)\n")
    E = np.arange(16, dtype=np.int64)
    OFF = np.array([7, 2, 5, 0], dtype=np.int64)
    V = np.array([77, 22, 55, 0], dtype=np.int64)
    pay = {"E": E, "OFF": OFF, "V": V}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    want_a = int(E[12]) + (int(E[13]) << 8) + (int(E[14]) << 16) + (int(E[15]) << 24)
    want_h = E.copy()
    want_h[OFF] = V
    check("ibyte-ref-assemble", int(ref["A"][0]) == want_a,
          f"{ref['A'].tolist()} want {want_a}")
    check("ibyte-ref-sum", int(ref["S"][0]) == int(E[:4].sum()),
          str(ref["S"].tolist()))
    check("ibyte-ref-scatter", bool((ref["H"] == want_h).all()),
          str(ref["H"].tolist()))

    names = ["A", "S", "H"]
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"ib_{target}", target)
        check(f"ibyte-{target}-match",
              all(bool((got[n] == ref[n]).all()) for n in names),
              ",".join(str(got[n].tolist()) for n in names))

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("ibyte-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("ibyte-nonfpu-trap-clean", False, str(e)[:140])

    # loud: reference kind/rank/bounds
    for tag, fn, args, want in (
            ("loud-ref-assemble-len", op_iassemble, (E, 0, 9), "len must be"),
            ("loud-ref-assemble-oob", op_iassemble, (E, 12, 8), "out of range"),
            ("loud-ref-assemble-2d", op_iassemble, (E.reshape(4, 4), 0, 2),
             "expected 1D"),
            ("loud-ref-sum-oob", op_isum, (E, 14, 8), "out of range"),
            ("loud-ref-scatter-oob", op_iscatter1,
             (E, np.array([16], dtype=np.int64),
              np.array([1], dtype=np.int64)), "out of range"),
            ("loud-ref-scatter-mismatch", op_iscatter1,
             (E, np.array([1, 2], dtype=np.int64),
              np.array([1], dtype=np.int64)), "IDS"),
            ("loud-ref-scatter-2d", op_iscatter1,
             (E.reshape(4, 4), OFF, V), "must be 1D")):
        try:
            fn(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time
    for tag, prog, sample, out, want in (
            ("loud-c-assemble-len",
             "IN E\nA = IASSEMBLE(E, 0, 9)\n", {"E": E}, ["A"],
             "len must be 1..8"),
            ("loud-c-assemble-oob",
             "IN E\nA = IASSEMBLE(E, 12, 8)\n", {"E": E}, ["A"], "of dim"),
            ("loud-c-assemble-2d",
             "IN E\nA = IASSEMBLE(E, 0, 2)\n", {"E": E.reshape(4, 4)},
             ["A"], "1D I"),
            ("loud-c-scatter-2d",
             "IN E\nIN OFF\nIN V\nH = ISCATTER1(E, OFF, V)\n",
             {"E": E, "OFF": OFF.reshape(2, 2), "V": V}, ["H"], "1D I IDS")):
        try:
            compile_program(prog, "c", sample=sample, outputs=out)
            check(tag, False, "accepted")
        except (NoPattern, ASM.AsmError, ValueError) as e:
            check(tag, want in str(e), str(e)[:70])

    # codec: LE field decode (assemble + sum) end to end
    art = compile_program("IN E\nA = IASSEMBLE(E, 12, 4)\nS = ISUM(E, 0, 4)\n",
                          "c", sample={"E": np.zeros(16, dtype=np.int64)},
                          outputs=["A", "S"], mode="codec")
    so = build_shared(art["source"], work, name="ib_codec", libs=art.get("libs"))
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

    E8 = np.arange(16, dtype=np.int64)
    A = np.zeros(1, dtype=np.int64)
    S = np.zeros(1, dtype=np.int64)
    bind("E", E8, 0)
    bind("A", A, 2)
    bind("S", S, 2)
    step = lib.jotnar_codec_step(h)
    check("ibyte-codec-assemble",
          step == 0 and int(A[0]) == want_a and int(S[0]) == 6,
          f"step={step} A={A.tolist()} S={S.tolist()}")
    lib.jotnar_codec_destroy(h)

    # codec: header rebuild via scatter, then a bad index fails loud
    art = compile_program("IN E\nIN OFF\nIN V\nH = ISCATTER1(E, OFF, V)\n",
                          "c", sample={"E": np.zeros(8, dtype=np.int64),
                                       "OFF": np.zeros(4, dtype=np.int64),
                                       "V": np.zeros(4, dtype=np.int64)},
                          outputs=["H"], mode="codec")
    so = build_shared(art["source"], work, name="ib_scat", libs=art.get("libs"))
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
    E0 = np.arange(8, dtype=np.int64)
    OFF2 = np.array([7, 2, 5, 0], dtype=np.int64)
    V2 = np.array([70, 20, 50, 0], dtype=np.int64)
    H = np.zeros(8, dtype=np.int64)
    bind("E", E0, 0)
    bind("OFF", OFF2, 0)
    bind("V", V2, 0)
    bind("H", H, 2)
    step = lib.jotnar_codec_step(h)
    want = E0.copy()
    want[OFF2] = V2
    check("ibyte-codec-scatter", step == 0 and bool((H == want).all()),
          f"step={step} H={H.tolist()}")
    OFF2[:] = [0, 1, 2, 8]
    rc_bad = lib.jotnar_codec_step(h)
    check("ibyte-codec-oob-loud", rc_bad == 4, f"rc={rc_bad}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
