"""GATHER batch gate: 2D ids (R,k) -> (R,k,C), same op, no new kernel.

Same bytes as R*k flat fibers (k_gather_f32 over the flattened view);
the 2D shape keeps downstream per-head BMMV strides expressible
(TOPK idx feeding V-gather in the sparse decode layer). Gates:
lattice triples 2D vs torch (quantum-bounded move), CUDA F-table 2D
0-diff vs torch, 1D path untouched, dynamic/3D ids fail loud,
shape inference (R,k,C).
Usage: python3 tests/test_gather2d.py
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

V, C, R, K = 16, 8, 7, 8


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as LS
    rng = np.random.default_rng(21)
    T = np.ascontiguousarray(rng.normal(size=(V, C)))
    ids = np.ascontiguousarray(rng.integers(0, V, size=(R, K)))

    def enc(a):
        s, e, z = LS.encode(a.ravel())
        return (s.reshape(a.shape), e.reshape(a.shape), z.reshape(a.shape))

    def dec(t):
        return LS.decode(t[0], t[1]) * (1 - t[2].astype(np.float64))

    # -- lattice triples 2D vs torch ----------------------------------
    fn = REGISTRY["GATHER"][0]
    got = dec(fn([enc(T), ids], {"m_cov": "128"}, {}))
    ref = T[ids]
    err = float(np.abs(got - ref).max())
    sig = float(np.abs(ref).max())
    db = 20 * np.log10(sig / max(err, 1e-300))
    check("gather2d-lattice", db > 50, f"{db:.1f}dB (move quantum)")

    # -- shape inference (R,k,C) ---------------------------------------
    from chain.emit_c import compile_program
    t = "IN V\nIN I\nG = GATHER(V, I)\n"
    art = compile_program(
        t, "cuda",
        sample={"V": np.zeros((V, C), dtype=np.float16),
                "I": np.arange(R * K, dtype=np.int64).reshape(R, K) % V},
        outputs=["G"], basedir=os.path.join(ROOT, "programs"),
        use_fp16=True)
    check("gather2d-shapes", art["streams"]["G"] == ("F", "double", (R, K, C)),
          f"{art['streams']['G']}")

    # -- dynamic / 3D ids fail loud --------------------------------------
    from chain.asm import AsmError
    try:
        compile_program("IN V\nIN I\nG = GATHER(V, I)\n", "cuda",
                        sample={"V": np.zeros((V, C)),
                                "I": np.zeros((R, K, 2), dtype=np.int64)},
                        outputs=["G"],
                        basedir=os.path.join(ROOT, "programs"))
        check("gather2d-3d-loud", False, "no raise")
    except AsmError:
        check("gather2d-3d-loud", True, "")

    # -- CUDA F-table 2D: bit-exact move ----------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_cuda import build_cu
        work = "/tmp/gather2d_cu"
        os.makedirs(work, exist_ok=True)
        Tf = np.ascontiguousarray(T, dtype=np.float32)
        art = compile_program(t, "cuda", sample={"V": Tf, "I": ids},
                              outputs=["G"],
                              basedir=os.path.join(ROOT, "programs"))
        exe = build_cu(art["source"], work, name="g2d",
                       units=art.get("units"))
        Tf.tofile(os.path.join(work, "f_V.bin"))
        ids.tofile(os.path.join(work, "f_I.bin"))
        argv = [exe, os.path.join(work, "f_V.bin"),
                os.path.join(work, "f_I.bin"),
                os.path.join(work, "o_G.bin")]
        r = subprocess.run(argv, capture_output=True, text=True)
        check("gather2d-cu-run", r.returncode == 0, r.stderr[:200])
        if r.returncode == 0:
            got = np.fromfile(os.path.join(work, "o_G.bin"),
                              dtype=np.float32).reshape(R, K, C)
            exp = Tf[ids]
            check("gather2d-cu-exact", bool((got == exp).all()),
                  f"maxabs={np.abs(got - exp).max():.2e} (move is exact)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
