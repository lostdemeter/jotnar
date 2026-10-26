#!/usr/bin/env python3
"""IVARINT gate: the SQLite varint decoded in the kernel.

IVARINT(X, START) -> (value, length).  Big-endian 7-bit groups, the high
bit = "more", the 9th byte full-width.  Reference = C = non-FPU; loud on
a bad start / truncation; CUDA refuses.
"""
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
from chain.asm_ops import REGISTRY, SIGS, op_ivarint  # noqa: E402
from chain.emit_c import NoPattern, build, compile_program  # noqa: E402
from chain.emit_nonfpu import float_trap  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def enc(v):
    """SQLite varint encoder (the inverse of the op)."""
    if v < 0 or v > (1 << 64) - 1:
        raise ValueError("range")
    if v > (1 << 56) - 1:
        out = [v & 0xFF]
        v >>= 8
        for _ in range(8):
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(out))
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


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
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:300]}")
    return {o: np.fromfile(os.path.join(work, f"out_{o}.bin"), dtype=np.int64)
            for o in art["outputs"]}


def main():
    work = "/tmp/jotnar_ivarint"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    values = [0, 1, 127, 128, 129, 16383, 16384, 1 << 20, 1 << 31,
              (1 << 32) - 1, 1 << 40, (1 << 56) - 1, 1 << 56, (1 << 63) - 1]
    for v in values:
        blob = enc(v)
        got = op_ivarint((np.frombuffer(blob, dtype=np.uint8).astype(np.int64),
                          0), None, None)
        check(f"ivarint-ref-{v}",
              int(got[0][0]) == v and int(got[1][0]) == len(blob),
              f"{blob.hex()} -> {int(got[0][0])},{int(got[1][0])}")

    # an offset start: a prefix of junk, then the varint
    blob = b"\xaa\xbb" + enc(300)
    got = op_ivarint((np.frombuffer(blob, dtype=np.uint8).astype(np.int64),
                      2), None, None)
    check("ivarint-ref-offset", int(got[0][0]) == 300 and int(got[1][0]) == 2)

    # the compiled kernels: a dynamic start (the loop shape)
    page = np.zeros(64, dtype=np.int64)
    page[10:12] = [0x82, 0x2c]          # 300 at offset 10
    start = np.array([10], dtype=np.int64)
    pay = {"P": page, "S": start}
    src = "IN P\nIN S\nV, L = IVARINT(P, S)\n"
    ref = ASM.run_text(src, REGISTRY, pay, sigs=SIGS)
    check("ivarint-ref-dynamic",
          int(ref["V"][0]) == 300 and int(ref["L"][0]) == 2,
          f"{int(ref['V'][0])},{int(ref['L'][0])}")
    for target in ("c", "nonfpu"):
        art = compile_program(src, target, sample=pay, outputs=["V", "L"])
        got = _run_art(work, art, pay, f"ivarint_{target}", target)
        check(f"ivarint-{target}",
              int(got["V"][0]) == 300 and int(got["L"][0]) == 2,
              f"{int(got['V'][0])},{int(got['L'][0])}")

    # a literal start, and the 9-byte form
    big = (1 << 63) - 1
    blob = enc(big)
    pay = {"P": np.frombuffer(blob, dtype=np.uint8).astype(np.int64)}
    src = "IN P\nV, L = IVARINT(P, 0)\n"
    for target in ("c", "nonfpu"):
        art = compile_program(src, target, sample=pay, outputs=["V", "L"])
        got = _run_art(work, art, pay, f"ivarint9_{target}", target)
        check(f"ivarint-9byte-{target}",
              int(got["V"][0]) == big and int(got["L"][0]) == 9,
              f"{int(got['V'][0])},{int(got['L'][0])}")

    # loud: a start out of range (runtime rc 32) and a truncated varint
    page = np.zeros(8, dtype=np.int64)
    start = np.array([99], dtype=np.int64)
    pay = {"P": page, "S": start}
    src_dyn = "IN P\nIN S\nV, L = IVARINT(P, S)\n"
    art = compile_program(src_dyn, "c", sample=pay, outputs=["V", "L"])
    exe = build(art["source"], work, name="ivarint_oob")
    fn = os.path.join(work, "in_P.bin")
    np.ascontiguousarray(page, dtype=np.int64).tofile(fn)
    sn = os.path.join(work, "in_S.bin")
    np.ascontiguousarray(start, dtype=np.int64).tofile(sn)
    r = subprocess.run([exe, fn, sn, os.path.join(work, "o1"),
                        os.path.join(work, "o2")], capture_output=True)
    check("ivarint-oob-loud", r.returncode == 32, f"rc={r.returncode}")
    try:
        op_ivarint((np.frombuffer(b"\x80", dtype=np.uint8).astype(np.int64),
                    0), None, None)
        check("ivarint-truncated-loud", False, "accepted")
    except ValueError as e:
        check("ivarint-truncated-loud", "truncated" in str(e), str(e)[:50])

    # CUDA refuses
    pay = {"P": page, "S": start}
    try:
        compile_program(src_dyn, "cuda", sample=pay, outputs=["V", "L"])
        check("ivarint-cuda-loud", False, "cuda compiled IVARINT")
    except NoPattern as e:
        check("ivarint-cuda-loud", "IVARINT" in str(e), str(e)[:60])

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
