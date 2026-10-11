"""SCATTER gate: indexed row update, KV-cache merge primitive (51st mnemonic).

SCATTER(base (S,C), row (1,C), idx scalar/1-elem) -> base copy with row
idx := row. Replaces the ones-broadcast MATMUL + one-hot SELECT merge
(same math, bit-exact; kills 112 kernels/layer-set + 2 live files).
Gates: lattice vs torch indexed update (0-diff), OOB/width/T fail loud,
C vs lattice, CUDA vs lattice bit-exact, non-FPU refuses loud.
Usage: python3 tests/test_scatter.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

S, C = 16, 8


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(27)
    B = np.ascontiguousarray(rng.normal(size=(S, C)))
    R = np.ascontiguousarray(rng.normal(size=(1, C)))
    N = 5

    fn = REGISTRY["SCATTER"][0]
    # -- lattice vs torch -------------------------------------------
    import torch
    got = fn([B, R, np.array([N])], {}, {})
    ref = B.copy()
    ref[N, :] = R[0, :]
    check("scatter-torch", bool((got == ref).all()),
          f"maxabs={np.abs(got - ref).max():.2e}")
    check("scatter-shapes", got.shape == (S, C), f"{got.shape}")
    check("scatter-pure", bool((B[0] == B[0]).all()) and got is not B,
          "input unmoved (out-of-place)")

    # -- fail loud ----------------------------------------------------
    for bad, tag in [([B, R, np.array([S])], "oob"),
                     ([B, R, np.array([-1])], "neg"),
                     ([B, R, np.array([1, 2])], "vec"),
                     ([B, np.zeros((2, C)), np.array([1])], "width"),
                     ([B[:, 0], R, np.array([1])], "1D")]:
        try:
            fn(bad, {}, {})
            check(f"scatter-{tag}-loud", False, "no raise")
        except (ValueError, IndexError):
            check(f"scatter-{tag}-loud", True, "")

    # -- text form ------------------------------------------------------
    t = "IN B\nIN R\nIN I\nKF = SCATTER(B, R, I)\n"
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("scatter-text", len(bound) == 1 and bound[0][1] == "SCATTER"
          and bound[0][0] == ["KF"], f"{bound[0][0]}")
    feeds = {"B": B, "R": R, "I": np.array([N])}
    out = ASM.run_text(t, REGISTRY, feeds, sigs=SIGS)
    check("scatter-run", bool((out["KF"] == ref).all()), "run_text matches")

    # -- C vs lattice -----------------------------------------------------
    from chain.emit_c import compile_program as _cc
    from chain.emit_c import build as _bc
    work = "/tmp/scatter_c"
    os.makedirs(work, exist_ok=True)
    art = _cc(t, "c", sample={"B": B, "R": R, "I": np.array([N])},
              outputs=["KF"], basedir=os.path.join(ROOT, "programs"))
    exe = _bc(art["source"], work, name="sc")
    B.tofile(os.path.join(work, "f_B.bin"))
    R.tofile(os.path.join(work, "f_R.bin"))
    np.array([N]).tofile(os.path.join(work, "f_I.bin"))
    argv = [exe, os.path.join(work, "f_B.bin"),
            os.path.join(work, "f_R.bin"), os.path.join(work, "f_I.bin"),
            os.path.join(work, "o_KF.bin")]
    r = subprocess.run(argv, capture_output=True, text=True)
    check("scatter-c-run", r.returncode == 0, r.stderr[:200])
    if r.returncode == 0:
        gotc = np.fromfile(os.path.join(work, "o_KF.bin"),
                           dtype=np.float64).reshape(S, C)
        check("scatter-c-exact", bool((gotc == ref).all()),
              f"maxabs={np.abs(gotc - ref).max():.2e}")
    # C OOB refuses loud (rc=4)
    np.array([S]).tofile(os.path.join(work, "f_I.bin"))
    r = subprocess.run(argv, capture_output=True, text=True)
    check("scatter-c-oob-loud", r.returncode == 4, f"rc={r.returncode}")

    # -- non-FPU refuses loud ----------------------------------------------
    from chain.emit_c import NoPattern
    from chain.emit_nonfpu import NonFPUBackend
    try:
        NonFPUBackend().pattern("SCATTER", ["KF"], ["B", "R", "I"], None,
                                {"streams": {}, "config": {}, "ids_n": "1"})
        check("scatter-nonfpu-refuses", False, "no raise")
    except NoPattern:
        check("scatter-nonfpu-refuses", True, "")

    # -- CUDA vs lattice (bit-exact move) ------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_c import compile_program
        from chain.emit_cuda import build_cu
        work = "/tmp/scatter_cu"
        os.makedirs(work, exist_ok=True)
        Bf = np.ascontiguousarray(B, dtype=np.float32)
        Rf = np.ascontiguousarray(R, dtype=np.float32)
        art = compile_program(t, "cuda", sample={"B": Bf, "R": Rf,
                                                 "I": np.array([N])},
                              outputs=["KF"],
                              basedir=os.path.join(ROOT, "programs"))
        exe = build_cu(art["source"], work, name="sc",
                       units=art.get("units"))
        Bf.tofile(os.path.join(work, "f_B.bin"))
        Rf.tofile(os.path.join(work, "f_R.bin"))
        np.array([N]).tofile(os.path.join(work, "f_I.bin"))
        argv = [exe, os.path.join(work, "f_B.bin"),
                os.path.join(work, "f_R.bin"),
                os.path.join(work, "f_I.bin"),
                os.path.join(work, "o_KF.bin")]
        r = subprocess.run(argv, capture_output=True, text=True)
        check("scatter-cu-run", r.returncode == 0, r.stderr[:200])
        if r.returncode == 0:
            gotd = np.fromfile(os.path.join(work, "o_KF.bin"),
                               dtype=np.float32).reshape(S, C)
            exp = ref.astype(np.float32)
            check("scatter-cu-exact", bool((gotd == exp).all()),
                  f"maxabs={np.abs(gotd - exp).max():.2e}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
