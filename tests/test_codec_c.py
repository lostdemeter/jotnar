#!/usr/bin/env python3
"""C codec ABI gate: create/bind/step/destroy matches one-shot.

Compiles a program with mode="codec" into a shared library, drives it
through ctypes, and compares each step with the one-shot binary on the
same inputs.  Also checks the loud bind contracts (unknown name, wrong
mode, wrong byte count, unbound slot).
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain.emit_c import build, build_shared, compile_program

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _write_input(path, kind, arr):
    if kind == "F":
        np.ascontiguousarray(arr, dtype=np.float64).tofile(path)
    else:
        np.ascontiguousarray(arr, dtype=np.int64).tofile(path)


def oneshot(art, payload, work, name):
    exe = build(art["source"], work, name=name, libs=art["libs"])
    argv = []
    for n in art["inputs"]:
        k = art["streams"][n][0]
        p = os.path.join(work, f"{name}_{n}.bin")
        _write_input(p, k, payload[n])
        argv.append(p)
    outs = []
    for o in art["outputs"]:
        k = art["streams"][o][0]
        p = os.path.join(work, f"{name}out_{o}.bin")
        outs.append((o, k, p))
        argv.append(p)
    r = subprocess.run([exe] + argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:500]
    got = {}
    for o, k, p in outs:
        if k == "F":
            got[o] = np.fromfile(p, dtype=np.float64)
        else:
            got[o] = np.fromfile(p, dtype=np.int64)
    return got


class Codec:
    def __init__(self, so):
        self.lib = ctypes.CDLL(so)
        self.lib.jotnar_codec_create.restype = ctypes.c_void_p
        self.lib.jotnar_codec_bind.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_int]
        self.lib.jotnar_codec_bind.restype = ctypes.c_int
        self.lib.jotnar_codec_step.argtypes = [ctypes.c_void_p]
        self.lib.jotnar_codec_step.restype = ctypes.c_int
        self.lib.jotnar_codec_destroy.argtypes = [ctypes.c_void_p]
        self.handle = self.lib.jotnar_codec_create()
        assert self.handle

    def bind(self, name, arr, mode):
        ptr = arr.ctypes.data_as(ctypes.c_void_p)
        return self.lib.jotnar_codec_bind(
            self.handle, name.encode(), ptr,
            ctypes.c_size_t(arr.nbytes), ctypes.c_int(mode))

    def step(self):
        return self.lib.jotnar_codec_step(self.handle)

    def destroy(self):
        self.lib.jotnar_codec_destroy(self.handle)


def test_matmul(work):
    text = "IN X\nIN W\nY = MATMUL(X, W)\n"
    rng = np.random.default_rng(2)
    X = np.ascontiguousarray(rng.normal(size=(4, 8)))
    W = np.ascontiguousarray(rng.normal(size=(8, 5)))
    X2 = np.ascontiguousarray(rng.normal(size=(4, 8)))
    art_one = compile_program(text, "c", sample={"X": X, "W": W},
                              outputs=["Y"])
    art = compile_program(text, "c", sample={"X": X, "W": W},
                          outputs=["Y"], live=["X"], mode="codec")
    check("codec-matmul-meta", art["codec"]["symbol"] == "jotnar_codec")
    ref = oneshot(art_one, {"X": X, "W": W}, work, "codec_mm_one")
    so = build_shared(art["source"], work, name="codec_mm",
                      libs=art["libs"])
    c = Codec(so)
    try:
        check("codec-bind-W", c.bind("W", W, 0) >= 0)
        check("codec-bind-X", c.bind("X", X, 1) >= 0)
        out = np.zeros((4, 5), dtype=np.float64)
        check("codec-bind-Y", c.bind("Y", out, 2) >= 0)
        check("codec-step0", c.step() == 0)
        check("codec-matmul-step0",
              np.allclose(out, ref["Y"].reshape(4, 5), atol=1e-12),
              f"maxdiff {np.abs(out - ref['Y'].reshape(4,5)).max():.2e}")
        X[:] = X2
        ref2 = oneshot(art_one, {"X": X2, "W": W}, work, "codec_mm_one2")
        check("codec-step1", c.step() == 0)
        check("codec-matmul-step1",
              np.allclose(out, ref2["Y"].reshape(4, 5), atol=1e-12))
        check("codec-bind-unknown", c.bind("nope", X, 1) == -4)
        check("codec-bind-wrong-mode", c.bind("X", X, 0) == -2)
        check("codec-bind-wrong-size",
              c.bind("X", np.zeros((4, 9)), 1) == -3)
    finally:
        c.destroy()


def test_unbound(work):
    text = "IN X\nIN W\nY = MATMUL(X, W)\n"
    rng = np.random.default_rng(4)
    X = np.ascontiguousarray(rng.normal(size=(2, 3)))
    W = np.ascontiguousarray(rng.normal(size=(3, 2)))
    art = compile_program(text, "c", sample={"X": X, "W": W},
                          outputs=["Y"], live=["X"], mode="codec")
    so = build_shared(art["source"], work, name="codec_unbound",
                      libs=art["libs"])
    c = Codec(so)
    try:
        check("codec-unbound-step", c.step() >= 10)
    finally:
        c.destroy()


def test_gather_argmax(work):
    text = ("IN bank\nIN tok\nROW = GATHER(bank, tok)\n"
            "OUT = ARGMAX(ROW, 1)\n")
    rng = np.random.default_rng(7)
    bank = np.ascontiguousarray(rng.normal(size=(32, 16)))
    toks = np.array([3], dtype=np.int64)
    art_one = compile_program(text, "c", sample={"bank": bank, "tok": toks},
                              outputs=["OUT"])
    ref = oneshot(art_one, {"bank": bank, "tok": toks}, work,
                  "codec_gather_one")
    art = compile_program(text, "c", sample={"bank": bank, "tok": toks},
                          outputs=["OUT"], live=["tok"], mode="codec")
    so = build_shared(art["source"], work, name="codec_gather",
                      libs=art["libs"])
    c = Codec(so)
    try:
        check("codec-gather-bind-bank", c.bind("bank", bank, 0) >= 0)
        check("codec-gather-bind-tok", c.bind("tok", toks, 1) >= 0)
        out = np.zeros(1, dtype=np.int64)
        check("codec-gather-bind-out", c.bind("OUT", out, 2) >= 0)
        check("codec-gather-step0", c.step() == 0)
        check("codec-gather-match", bool((out == ref["OUT"]).all()),
              f"{out.tolist()} vs {ref['OUT'].tolist()}")
        toks[0] = 9
        ref2 = oneshot(art_one, {"bank": bank, "tok": toks}, work,
                       "codec_gather_one2")
        check("codec-gather-step1", c.step() == 0)
        check("codec-gather-match1", bool((out == ref2["OUT"]).all()))
    finally:
        c.destroy()


def main():
    work = "/tmp/emit_codec_c"
    os.makedirs(work, exist_ok=True)
    test_matmul(work)
    test_unbound(work)
    test_gather_argmax(work)
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
