#!/usr/bin/env python3
"""INFLATE gate: raw deflate (RFC 1951) decoded in the kernel.

IINFLATE(X, N) decodes a raw deflate stream into exactly N bytes, fail
loud otherwise.  The reference (a pure-Python decoder) is checked against
zlib; the C and non-FPU backends are checked against the reference.

Covers stored / fixed / dynamic blocks, multiple blocks (a sync flush in
the middle), the empty stream, random bytes, zeros, text, and the loud
cases (wrong N, truncated input, CUDA refusal).
"""
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
from chain.asm_ops import REGISTRY, SIGS, op_iinflate  # noqa: E402
from chain.emit_c import NoPattern, build, compile_program  # noqa: E402
from chain.emit_nonfpu import float_trap  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def raw_deflate(data, level=6, strategy=zlib.Z_DEFAULT_STRATEGY,
                split=None):
    c = zlib.compressobj(level, zlib.DEFLATED, -15, 8, strategy)
    if split is None:
        return c.compress(data) + c.flush()
    out = c.compress(data[:split]) + c.flush(zlib.Z_SYNC_FLUSH)
    out += c.compress(data[split:]) + c.flush()
    return out


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
    work = "/tmp/jotnar_inflate"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    text = (b"the quick brown fox jumps over the lazy dog\n" * 200)[:5000]
    cases = {
        "empty": (b"", 6, zlib.Z_DEFAULT_STRATEGY, None),
        "one": (b"a", 6, zlib.Z_DEFAULT_STRATEGY, None),
        "stored": (bytes(range(256)) * 4, 0, zlib.Z_DEFAULT_STRATEGY, None),
        "fixed": (text, 6, zlib.Z_FIXED, None),
        "dynamic": (text, 9, zlib.Z_DEFAULT_STRATEGY, None),
        "zeros": (bytes(8192), 6, zlib.Z_DEFAULT_STRATEGY, None),
        "random": (np.random.default_rng(7).integers(
            0, 256, 3000, dtype=np.uint8).tobytes(), 6,
            zlib.Z_DEFAULT_STRATEGY, None),
        "multiblock": (text, 6, zlib.Z_DEFAULT_STRATEGY, 2500),
        "huffman-only": (text, 6, zlib.Z_HUFFMAN_ONLY, None),
    }

    for name, (data, level, strategy, split) in cases.items():
        comp = raw_deflate(data, level, strategy, split)
        ref = op_iinflate(
            (np.frombuffer(comp, dtype=np.uint8).astype(np.int64),
             len(data)), None, None)
        check(f"inflate-ref-{name}",
              bytes(int(v) & 0xFF for v in ref) == data
              and bytes(int(v) & 0xFF for v in ref)
              == zlib.decompress(comp, -15),
              f"{len(comp)} -> {len(data)}")

    # the compiled kernels, on three representative streams
    for name in ("stored", "dynamic", "multiblock"):
        data, level, strategy, split = cases[name]
        comp = raw_deflate(data, level, strategy, split)
        X = np.frombuffer(comp, dtype=np.uint8).astype(np.int64)
        pay = {"X": X}
        src = f"IN X\nD = IINFLATE(X, {len(data)})\n"
        ref = ASM.run_text(src, REGISTRY, pay, sigs=SIGS)["D"]
        for target in ("c", "nonfpu"):
            art = compile_program(src, target, sample=pay, outputs=["D"])
            got = _run_art(work, art, pay, f"inf_{name}_{target}", target)
            check(f"inflate-{name}-{target}",
                  bool((got["D"] == ref).all()),
                  f"{len(comp)} -> {len(got['D'])}")

    # loud: a wrong N fails at runtime (return 31)
    data, level, strategy, split = cases["dynamic"]
    comp = raw_deflate(data, level, strategy, split)
    X = np.frombuffer(comp, dtype=np.uint8).astype(np.int64)
    pay = {"X": X}
    src = f"IN X\nD = IINFLATE(X, {len(data) - 1})\n"
    art = compile_program(src, "c", sample=pay, outputs=["D"])
    exe = build(art["source"], work, name="inf_wrongn")
    fn = os.path.join(work, "in_X.bin")
    np.ascontiguousarray(X, dtype=np.int64).tofile(fn)
    r = subprocess.run([exe, fn, os.path.join(work, "out_D.bin")],
                       capture_output=True, text=True)
    check("inflate-wrong-n-loud", r.returncode == 31, f"rc={r.returncode}")

    # loud: the reference refuses a truncated stream
    try:
        op_iinflate((np.frombuffer(comp[:4], dtype=np.uint8).astype(np.int64),
                     len(data)), None, None)
        check("inflate-truncated-loud", False, "accepted")
    except ValueError as e:
        check("inflate-truncated-loud", "truncated" in str(e), str(e)[:60])

    # CUDA is not a target for the integer decode
    try:
        compile_program(src, "cuda", sample=pay, outputs=["D"])
        check("inflate-cuda-loud", False, "cuda compiled IINFLATE")
    except NoPattern as e:
        check("inflate-cuda-loud", "IINFLATE" in str(e), str(e)[:60])

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
