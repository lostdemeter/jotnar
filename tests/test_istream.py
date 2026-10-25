#!/usr/bin/env python3
"""ISTREAM gate: dynamic IWINDOW + streaming ops (ICRC32, IFIND).

The control-half primitives for compressed data:
  * IWINDOW with a 1-element I start (runtime offset, bounds-checked);
  * ICRC32: full CRC-32 over a byte stream (internal loop), zlib-matching;
  * IFIND: first pattern offset (region scan, internal loop).
Reference = C = non-FPU; codec mode drives all three end to end, and a
dynamic window out of range fails loud (step returns 4).
Usage: python3 tests/test_istream.py (needs cc).
"""
import ctypes
import os
import shutil
import subprocess
import sys
import zlib

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR",
                                  os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain import asm as ASM  # noqa: E402
from chain.asm_ops import (  # noqa: E402
    REGISTRY,
    SIGS,
    op_icrc32,
    op_icrc32c,
    op_ifind,
    op_iwindow,
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
    work = "/tmp/jotnar_istream"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    E = np.array([0x50, 0x4B, 0x03, 0x04, 1, 2, 3, 4,
                  0x50, 0x4B, 0x03, 0x04], dtype=np.int64)
    PAT = np.array([0x50, 0x4B, 0x03, 0x04], dtype=np.int64)
    OFF = np.array([8], dtype=np.int64)
    text = ("IN E\nIN OFF\nIN PAT\n"
            "W = IWINDOW(E, OFF, 4)\n"
            "C = ICRC32(E)\n"
            "O = IFIND(E, PAT)\n")
    pay = {"E": E, "OFF": OFF, "PAT": PAT}
    ref = ASM.run_text(text, REGISTRY, pay, sigs=SIGS)
    check("istream-ref-window", bool((ref["W"] == E[8:12]).all()),
          str(ref["W"].tolist()))
    check("istream-ref-crc",
          int(ref["C"][0]) == (zlib.crc32(bytes(int(b) & 0xFF for b in E))
                               & 0xFFFFFFFF),
          str(ref["C"].tolist()))
    check("istream-ref-find", int(ref["O"][0]) == 0, str(ref["O"].tolist()))

    names = ["W", "C", "O"]
    for target in ("c", "nonfpu"):
        art = compile_program(text, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"ist_{target}", target)
        check(f"istream-{target}-match",
              all(bool((got[n] == ref[n]).all()) for n in names),
              ",".join(str(got[n].tolist()) for n in names))

    art = compile_program(text, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("istream-nonfpu-trap-clean", True, "int64 only")
    except Exception as e:  # noqa: BLE001
        check("istream-nonfpu-trap-clean", False, str(e)[:140])

    # ICRC32C: the Castagnoli (the ext4 crc32c), seeded + raw (no final xor)
    def crc32c_raw(bs, crc):
        for b in bs:
            crc ^= int(b) & 0xFF
            for _ in range(8):
                crc = (crc >> 1) ^ (0x82F63B78 if (crc & 1) else 0)
        return crc & 0xFFFFFFFF

    A = np.array([0x31, 0x32, 0x33, 0x34], dtype=np.int64)
    B = np.array([0x35, 0x36, 0x37, 0x38, 0x39], dtype=np.int64)
    AB = np.concatenate([A, B])
    SEED = np.array([0xFFFFFFFF], dtype=np.int64)
    text2 = ("IN A\nIN B\nIN AB\nIN SEED\n"
             "C1 = ICRC32C(A, SEED)\nC2 = ICRC32C(B, C1)\n"
             "C3 = ICRC32C(AB, SEED)\n")
    pay2 = {"A": A, "B": B, "AB": AB, "SEED": SEED}
    ref2 = ASM.run_text(text2, REGISTRY, pay2, sigs=SIGS)
    check("icrc32c-ref",
          int(ref2["C1"][0]) == crc32c_raw(A.tolist(), 0xFFFFFFFF),
          hex(int(ref2["C1"][0])))
    check("icrc32c-chain",
          int(ref2["C2"][0]) == int(ref2["C3"][0])
          == crc32c_raw(AB.tolist(), 0xFFFFFFFF),
          hex(int(ref2["C2"][0])))
    for target in ("c", "nonfpu"):
        art2 = compile_program(text2, target, sample=pay2,
                               outputs=["C1", "C2", "C3"])
        got2 = _run_art(work, art2, pay2, f"icrc32c_{target}", target)
        check(f"icrc32c-{target}-match",
              all(bool((got2[n] == ref2[n]).all())
                  for n in ("C1", "C2", "C3")),
              ",".join(hex(int(got2[n][0])) for n in ("C1", "C2", "C3")))

    # loud: reference
    for tag, fn, args, want in (
            ("loud-ref-window-oob", op_iwindow,
             (E, np.array([9], dtype=np.int64), 4), "out of range"),
            ("loud-ref-window-start-2d", op_iwindow,
             (E, np.array([[0]], dtype=np.int64), 4), "1-elem"),
            ("loud-ref-crc-2d", op_icrc32, (E.reshape(3, 4),), "1D"),
            ("loud-ref-icrc32c-2d", op_icrc32c,
             (E.reshape(3, 4), SEED), "1D"),
            ("loud-ref-find-empty", op_ifind,
             (E, np.array([], dtype=np.int64)), "empty pattern")):
        try:
            fn(args, {}, {})
            check(tag, False, "accepted")
        except ValueError as e:
            check(tag, want in str(e), str(e)[:60])

    # loud: compile-time
    try:
        compile_program("IN E\nIN OFF\nW = IWINDOW(E, OFF, 4)\n", "c",
                        sample={"E": E, "OFF": np.zeros((2, 1),
                                                        dtype=np.int64)},
                        outputs=["W"])
        check("loud-c-window-start-2d", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as e:
        check("loud-c-window-start-2d", "1-element" in str(e), str(e)[:70])

    # codec: dynamic window + full CRC + scan
    art = compile_program(text, "c", sample=pay, outputs=names, mode="codec")
    so = build_shared(art["source"], work, name="ist_codec",
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

    W = np.zeros(4, dtype=np.int64)
    C = np.zeros(1, dtype=np.int64)
    O = np.zeros(1, dtype=np.int64)
    bind("E", E, 0)
    bind("OFF", OFF, 0)
    bind("PAT", PAT, 0)
    bind("W", W, 2)
    bind("C", C, 2)
    bind("O", O, 2)
    step = lib.jotnar_codec_step(h)
    want_crc = zlib.crc32(bytes(int(b) & 0xFF for b in E)) & 0xFFFFFFFF
    check("istream-codec-run",
          step == 0 and bool((W == E[8:12]).all()) and int(C[0]) == want_crc
          and int(O[0]) == 0,
          f"step={step} W={W.tolist()} C={C.tolist()} O={O.tolist()}")
    # dynamic window out of range must fail loud (rc 4)
    OFF[0] = 9
    rc_bad = lib.jotnar_codec_step(h)
    check("istream-codec-window-oob", rc_bad == 4, f"rc={rc_bad}")
    lib.jotnar_codec_destroy(h)

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
