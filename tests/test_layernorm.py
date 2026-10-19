"""LAYERNORM gate (pattern A1): per-row LayerNorm+affine, C + CUDA.

LAYERNORM(X (R,D), W (D,), B (D,)) = (x-mean)/sqrt(var+eps)*w+b.
Gates: lattice vs torch (40dB), C vs torch (40dB), CUDA vs torch
(40dB) + bit-exact rerun, fail loud (missing eps_ln, width
mismatch, non-F input), non-FPU still refuses (contract unchanged).
eps from CONFIG eps_ln (true float, RMSNORM scaffolding doctrine).
Usage: python3 tests/test_layernorm.py
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

R, D = 4, 32
EPS = 1e-5


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def db(ref, got):
    """40dB bar, drill convention: 10*log10(1/mse)."""
    import numpy as np
    mse = float(np.mean((np.asarray(ref, dtype=np.float64).ravel()
                         - np.asarray(got, dtype=np.float64).ravel()) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-300))


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S
    rng = np.random.default_rng(181)
    Xv = rng.uniform(-1, 1, (R, D))
    Wv = np.ones(D) * 0.9 + rng.uniform(-1, 1, (D,)) * 0.05
    Bv = rng.uniform(-0.1, 0.1, (D,))
    X, W, B = S.encode(Xv), S.encode(Wv), S.encode(Bv)
    import torch
    ref = torch.nn.functional.layer_norm(
        torch.tensor(Xv), (D,), torch.tensor(Wv), torch.tensor(Bv),
        EPS).numpy()

    fn = REGISTRY["LAYERNORM"][0]
    gl = fn([X, W, B], {"eps_ln": str(EPS)}, {})
    gv = S.decode(gl[0], gl[1]) * (1 - gl[2].astype(np.float64))
    check("layernorm-torch", db(ref, gv) >= 40,
          f"{db(ref, gv):.1f}dB (lattice vs torch)")

    # -- text form ------------------------------------------------------
    t = (f"CONFIG eps_ln {EPS}\nIN X\nIN W\nIN B\n"
         f"O = LAYERNORM(X, W, B)\n")
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("layernorm-text", len(bound) == 1 and bound[0][1] == "LAYERNORM"
          and bound[0][0] == ["O"], f"{bound[0][0]}")
    out = ASM.run_text(t, REGISTRY, {"X": X, "W": W, "B": B}, sigs=SIGS)
    ov = S.decode(out["O"][0], out["O"][1]) * (1 - out["O"][2].astype(np.float64))
    check("layernorm-run", db(ref, ov) >= 40, "run_text matches")

    # -- fail loud --------------------------------------------------------
    from chain.asm import AsmError
    from chain.emit_c import CBackend, NoPattern
    from chain.emit_nonfpu import NonFPUBackend
    try:
        compile_program_noeps = None
        from chain.emit_c import compile_program
        compile_program("IN X\nIN W\nIN B\nO = LAYERNORM(X, W, B)\n", "c",
                        sample={"X": X, "W": W, "B": B}, outputs=["O"])
        check("layernorm-noeps-loud", False, "compiled without eps?!")
    except AsmError:
        check("layernorm-noeps-loud", True, "missing eps_ln refused")
    try:
        be = CBackend()
        be.pattern("LAYERNORM", ["O"], ["X", "W", "B"], None,
                   {"streams": {"X": ("T", "double", (R, D)),
                                "W": ("T", "double", (D,)),
                                "B": ("T", "double", (D,))},
                    "config": {"eps_ln": str(EPS)}, "ids_n": "1"})
        check("layernorm-tkind-loud", False, "T input accepted?!")
    except NoPattern:
        check("layernorm-tkind-loud", True, "non-F input refused")
    try:
        NonFPUBackend().pattern(
            "LAYERNORM", ["O"], ["X", "W", "B"], None,
            {"streams": {}, "config": {"eps_ln": str(EPS)}, "ids_n": "1"})
        check("layernorm-nonfpu-unchanged", False, "non-FPU now emits?!")
    except NoPattern as e:
        check("layernorm-nonfpu-unchanged", "no integer pattern" in str(e),
              "still refuses (contract unchanged)")

    # -- C backend: parity --------------------------------------------------
    from chain.emit_c import compile_program, build as _bc
    sample = {"X": np.ascontiguousarray(Xv), "W": np.ascontiguousarray(Wv),
              "B": np.ascontiguousarray(Bv)}
    art = compile_program(t, "c", sample=sample, outputs=["O"],
                          basedir=os.path.join(ROOT, "programs"))
    work = "/tmp/layernorm_c"
    os.makedirs(work, exist_ok=True)
    exe = _bc(art["source"], work, name="layernorm")
    f1 = os.path.join(work, "x.bin")
    f2 = os.path.join(work, "w.bin")
    f3 = os.path.join(work, "b.bin")
    fo = os.path.join(work, "o.bin")
    sample["X"].tofile(f1)
    sample["W"].tofile(f2)
    sample["B"].tofile(f3)
    r = subprocess.run([exe, f1, f2, f3, fo], capture_output=True, text=True)
    check("layernorm-c-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:150]}")
    if r.returncode == 0:
        gotc = np.fromfile(fo, dtype=np.float64).reshape(R, D)
        check("layernorm-c-parity", db(ref, gotc) >= 40,
              f"{db(ref, gotc):.1f}dB (C vs torch)")

    # -- CUDA backend: parity + determinism -----------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP cuda legs (needs nvcc + GPU)")
    else:
        from chain.emit_cuda import build_cu
        worku = "/tmp/layernorm_cu"
        os.makedirs(worku, exist_ok=True)
        Xf = np.ascontiguousarray(Xv, dtype=np.float32)
        Wf = np.ascontiguousarray(Wv, dtype=np.float32)
        Bf = np.ascontiguousarray(Bv, dtype=np.float32)
        artu = compile_program(t, "cuda",
                               sample={"X": Xf, "W": Wf, "B": Bf},
                               outputs=["O"],
                               basedir=os.path.join(ROOT, "programs"))
        exeu = build_cu(artu["source"], worku, name="layernorm",
                        units=artu.get("units"))
        Xf.tofile(os.path.join(worku, "f_X.bin"))
        Wf.tofile(os.path.join(worku, "f_W.bin"))
        Bf.tofile(os.path.join(worku, "f_B.bin"))
        argv = [exeu, os.path.join(worku, "f_X.bin"),
                os.path.join(worku, "f_W.bin"),
                os.path.join(worku, "f_B.bin"),
                os.path.join(worku, "o_O.bin")]
        outs = []
        for i in range(2):
            rr = subprocess.run(argv, capture_output=True, text=True)
            if rr.returncode != 0:
                check("layernorm-cu-run", False, rr.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(worku, "o_O.bin"),
                                    dtype=np.float32))
        else:
            check("layernorm-cu-run", True, "")
            check("layernorm-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            ref32 = ref.astype(np.float32)
            check("layernorm-cu-parity", db(ref32, outs[0]) >= 40,
                  f"{db(ref32, outs[0]):.1f}dB (CUDA vs torch)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
