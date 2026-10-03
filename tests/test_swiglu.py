"""SwiGLU block gate: Qwen MLP forward on all backends.

UP/GATE matmuls + SILU + MUL + DOWN matmul (+ residual): the exact
compute shape of a Qwen2 MLP block (norms stay host-side boundary, per
the assay doctrine). C/CUDA run decoded floats (eps gates); nonfpu runs
triples (bit-exact); lattice is the reference everywhere.
Usage: python3 tests/test_swiglu.py (needs cc + nvcc + GPU for cuda).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import build, compile_program
from chain.emit_nonfpu import float_trap

FAIL = []
# Operating point: m_cov 37300 covers the ~+-25 DOWN signals in ambient
# (U_m ~= 70; m_cov 35048 folds above ~9.4 -- the lattice ADD saturates
# while float does not, same species as the image-demo exposure find).
# Quantum at 37300 (~3e-4) stays far under every gate below.
PROG = ("CONFIG m_acc 36118\nCONFIG m_cov 37300\n"
        "IN hn\nIN wup\nIN wgate\nIN wdown\n"
        "UP = MATMUL(hn, wup)\nGATE = MATMUL(hn, wgate)\nGS = SILU(GATE)\n"
        "MID = MUL(GS, UP)\nDOWN = MATMUL(MID, wdown)\nOUT = ADD(hn, DOWN)\n")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    rng = np.random.default_rng(0)
    S_dim, D, Dff = 3, 16, 32
    fpay = {"hn": rng.normal(size=(S_dim, D)),
            "wup": rng.normal(size=(D, Dff)) * 0.5,
            "wgate": rng.normal(size=(D, Dff)) * 0.5,
            "wdown": rng.normal(size=(Dff, D)) * 0.5}
    fpay = {k: np.ascontiguousarray(v) for k, v in fpay.items()}
    tpay = {k: enc(v) for k, v in fpay.items()}
    ref = ASM.run_text(PROG, REGISTRY, tpay, sigs=SIGS, basedir="programs")
    ref_out = dec(ref["OUT"])
    work = "/tmp/swiglu"
    os.makedirs(work, exist_ok=True)
    # -- C float64 ----------------------------------------------------
    art = compile_program(PROG, "c", sample=dict(fpay), outputs=["OUT"],
                          basedir="programs")
    exe = build(art["source"], work, name="sw")
    argv = [exe]
    for n in art["inputs"]:
        fn = os.path.join(work, f"c_{n}.bin")
        fpay[n].astype(np.float64).tofile(fn)
        argv.append(fn)
    fo = os.path.join(work, "c_out.bin")
    argv.append(fo)
    r = subprocess.run(argv, capture_output=True, text=True)
    check("swiglu-c-run", r.returncode == 0, f"rc={r.returncode}")
    got = np.fromfile(fo, dtype=np.float64).reshape(ref_out.shape)
    dmax = float(np.abs(got - ref_out).max())
    print(f"swiglu-c-logits: maxabs={dmax:.3e} (cal 6e-02, lattice-quantum floor)")
    check("swiglu-c-eps", dmax < 6e-2, f"maxabs={dmax:.3e}")
    # -- CUDA FP32 -----------------------------------------------------
    if shutil.which("nvcc") is None:
        print("SKIP swiglu-cuda (no nvcc)")
    else:
        from chain.emit_cuda import build_cu
        art2 = compile_program(PROG, "cuda", sample=dict(fpay),
                               outputs=["OUT"], basedir="programs")
        exe2 = build_cu(art2["source"], work, name="swu")
        argv = [exe2]
        for n in art2["inputs"]:
            fn = os.path.join(work, f"u_{n}.bin")
            fpay[n].astype(np.float32).tofile(fn)
            argv.append(fn)
        fo2 = os.path.join(work, "u_out.bin")
        argv.append(fo2)
        r = subprocess.run(argv, capture_output=True, text=True)
        check("swiglu-cu-run", r.returncode == 0, f"rc={r.returncode}")
        if r.returncode == 0:
            got2 = np.fromfile(fo2, dtype=np.float32).reshape(ref_out.shape)
            dmax2 = float(np.abs(got2 - ref_out).max())
            check("swiglu-cu-eps", dmax2 < 6e-2, f"maxabs={dmax2:.3e}")
    # -- nonfpu triples, bit-exact --------------------------------------
    art3 = compile_program(PROG, "nonfpu", sample=tpay, outputs=["OUT"],
                           basedir="programs")
    try:
        float_trap(art3["source"])
        check("swiglu-nf-trap", True, "")
    except Exception as e:  # noqa: BLE001
        check("swiglu-nf-trap", False, str(e)[:120])
    exe3 = build(art3["source"], work, name="swn", libs=[])
    argv = [exe3]
    for n in art3["inputs"]:
        for comp, arr in zip("sez", tpay[n]):
            fn = os.path.join(work, f"n_{n}_{comp}.bin")
            np.ascontiguousarray(arr).tofile(fn)
            argv.append(fn)
    fos = []
    for c in "sez":
        fn = os.path.join(work, f"n_out_{c}.bin")
        fos.append(fn)
        argv.append(fn)
    r = subprocess.run(argv, capture_output=True, text=True)
    check("swiglu-nf-run", r.returncode == 0, f"rc={r.returncode}")
    if r.returncode == 0:
        ok = True
        for c, dt, fn in zip("sez", (np.int8, np.int32, np.uint8), fos):
            gotp = np.fromfile(fn, dtype=dt).reshape(ref_out.shape)
            same = bool((gotp == np.ascontiguousarray(
                ref["OUT"]["sez".index(c)])).all())
            ok = ok and same
        check("swiglu-nf-exact", ok, "bit-identical to lattice")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
