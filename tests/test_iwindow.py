#!/usr/bin/env python3
"""IWINDOW gate: 1D integer window (the format-kernel byte primitive).

Covers the gap that SLICE (2D F/T only) and GATHER ((V,C) table) leave:

  * reference eval is an exact numpy slice, bounds-checked loud;
  * C and non-FPU one-shot agree with the reference bit-for-bit;
  * non-FPU source stays float-free;
  * codec mode drives the little-endian EOCD field decode end to end;
  * bad kind / rank / literal / window fail loud.
Usage: python3 tests/test_iwindow.py (needs cc).
"""
import ctypes
import os
import shutil
import struct
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR",
                                  os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain import asm as ASM  # noqa: E402
from chain.asm_ops import REGISTRY, SIGS, op_iwindow  # noqa: E402
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
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"in_{n}.bin")
        np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
        files.append(fn)
    return files


def _read_outputs(work, art):
    out = {}
    for o in art["outputs"]:
        fn = os.path.join(work, f"out_{o}.bin")
        out[o] = np.fromfile(fn, dtype=np.int64)
    return out


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
    return _read_outputs(work, art)


def main():
    work = "/tmp/jotnar_iwindow"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = ("IN E\n"
            "A = IWINDOW(E, 3, 4)\n"
            "B = IWINDOW(E, 0, 2)\n"
            "M = IMUL(A, 7)\n"
            "C = IADD(M, 1)\n")
    E = np.arange(16, dtype=np.int64)
    pay = {"E": E}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    names = ["A", "B", "C"]

    # reference correctness
    check("iwindow-ref-slice",
          bool((ref["A"] == E[3:7]).all())
          and bool((ref["B"] == E[0:2]).all())
          and bool((ref["C"] == E[3:7] * 7 + 1).all()),
          f"A={ref['A'].tolist()} C={ref['C'].tolist()}")

    # one-shot C and non-FPU match the reference
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"iw_{target}", target)
        check(f"iwindow-{target}-match",
              all(bool((got[n] == ref[n]).all()) for n in names),
              ",".join(str(got[n].tolist()) for n in names))

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("iwindow-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("iwindow-nonfpu-trap-clean", False, str(e)[:140])

    # loud: reference bounds and kind
    for tag, args, want in (
            ("loud-ref-bounds", (E, 12, 8), "out of range"),
            ("loud-ref-negative", (E, -1, 2), "out of range"),
            ("loud-ref-2d", (E.reshape(4, 4), 0, 2), "expected 1D"),
            ("loud-ref-float", (E.astype(np.float64), 0, 2), "must be integer"),
            ("loud-ref-bad-literal", (E, 1.5, 2), "must be integral")):
        try:
            op_iwindow(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time type/rank
    for tag, prog, want in (
            ("loud-c-2d", "IN E\nA = IWINDOW(E, 0, 2)\n", "1D I"),
            ("loud-c-float", "IN E\nA = IWINDOW(E, 0, 2)\n", "1D I"),
            ("loud-c-range", "IN E\nA = IWINDOW(E, 0, 99)\n", "of dim")):
        E2 = (E.reshape(4, 4) if tag == "loud-c-2d"
              else E.astype(np.float64) if tag == "loud-c-float" else E)
        try:
            compile_program(prog, "c", sample={"E": E2}, outputs=["A"])
            check(tag, False, "accepted")
        except (NoPattern, ASM.AsmError, ValueError) as e:
            check(tag, want in str(e), str(e)[:70])

    # codec mode: decode the little-endian EOCD count/size/offset fields.
    def field(p, base, nbytes, out):
        lines = [f"{p}{k} = IWINDOW(E, {base + k}, 1)" for k in range(nbytes)]
        terms = [f"{p}0"]
        for k in range(1, nbytes):
            lines.append(f"{p}m{k} = IMUL({p}{k}, {1 << (8 * k)})")
            terms.append(f"{p}m{k}")
        acc = terms[0]
        for i, t in enumerate(terms[1:-1], start=1):
            nxt = f"{p}c{i}"
            lines.append(f"{nxt} = IADD({acc}, {t})")
            acc = nxt
        lines.append(f"{out} = IADD({acc}, {terms[-1]})")
        return lines

    prog = ["IN E"]
    prog += field("n", 10, 2, "OUTN")
    prog += field("s", 12, 4, "OUTS")
    prog += field("o", 16, 4, "OUTO")
    kern = "\n".join(prog) + "\n"
    art = compile_program(kern, "c", sample={"E": np.zeros(22, dtype=np.int64)},
                          outputs=["OUTN", "OUTS", "OUTO"], mode="codec")
    so = build_shared(art["source"], work, name="iw_codec", libs=art.get("libs"))
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

    E22 = np.arange(22, dtype=np.int64)
    E22[:] = list(struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 7, 7,
                              123456, 987654, 0))
    N = np.zeros(1, dtype=np.int64)
    S_ = np.zeros(1, dtype=np.int64)
    O = np.zeros(1, dtype=np.int64)
    rc = [bind("E", E22, 0), bind("OUTN", N, 2), bind("OUTS", S_, 2),
          bind("OUTO", O, 2)]
    step = lib.jotnar_codec_step(h)
    check("iwindow-codec-eocd",
          step == 0 and int(N[0]) == 7 and int(S_[0]) == 123456
          and int(O[0]) == 987654,
          f"rc={rc} step={step} N={int(N[0])} S={int(S_[0])} "
          f"O={int(O[0])}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
