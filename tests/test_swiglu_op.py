"""SWIGLU gate: fused SwiGLU MLP gate (54th mnemonic).

SWIGLU(ug (R,2I), inter) = silu(gate)*up with gate as second half, one
pass (replaces UG-split x2 + SILU + MUL: 4 kernels + 3 extra passes).
Gates: lattice vs torch (exact), CUDA vs lattice (eps) +
bit-determinism (elementwise), fail loud (width), C + non-FPU refuse
loud (contract).
Usage: python3 tests/test_swiglu_op.py
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

R, INTER = 3, 64


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(54)
    UG = np.ascontiguousarray(rng.normal(size=(R, 2 * INTER)) * 2)

    fn = REGISTRY["SWIGLU"][0]
    got = fn([UG, INTER], {}, {})
    import torch
    ref = torch.nn.functional.silu(
        torch.tensor(UG[:, INTER:])) * torch.tensor(UG[:, :INTER])
    ref = ref.numpy()
    check("swiglu-torch", float(np.abs(got - ref).max()) < 1e-12,
          f"maxabs={np.abs(got - ref).max():.2e} (1-ulp eval order)")

    # -- composition equivalence -------------------------------------------
    up = UG[:, :INTER]
    gs = UG[:, INTER:] / (1.0 + np.exp(-UG[:, INTER:]))
    check("swiglu-composed",
          float(np.abs(got - gs * up).max()) < 1e-12,
          "== slice/slice/silu/mul")

    # -- fail loud ------------------------------------------------------------
    for bad, tag in [([UG, 0], "zero"), ([UG, INTER + 1], "width"),
                     ([UG[:, 0], INTER], "1D"),
                     ([np.zeros((2, 2, 2 * INTER)), INTER], "3D")]:
        try:
            fn(bad, {}, {})
            check(f"swiglu-{tag}-loud", False, "no raise")
        except (ValueError, IndexError):
            check(f"swiglu-{tag}-loud", True, "")

    # -- text form --------------------------------------------------------------
    t = f"IN UG\nMID = SWIGLU(UG, {INTER})\n"
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("swiglu-text", len(bound) == 1 and bound[0][1] == "SWIGLU"
          and bound[0][0] == ["MID"], f"{bound[0][0]}")
    out = ASM.run_text(t, REGISTRY, {"UG": UG}, sigs=SIGS)
    check("swiglu-run", float(np.abs(out["MID"] - ref).max()) < 1e-12,
          "run_text matches")

    # -- backend refusal contract --------------------------------------------------
    from chain.emit_c import CBackend, NoPattern
    from chain.emit_nonfpu import NonFPUBackend
    for be, tag in ((CBackend(), "c"), (NonFPUBackend(), "nonfpu")):
        try:
            be.pattern("SWIGLU", ["MID"], ["UG", str(INTER)], None,
                       {"streams": {}, "config": {}, "ids_n": "1"})
            check(f"swiglu-{tag}-refuses", False, "no raise")
        except NoPattern:
            check(f"swiglu-{tag}-refuses", True, "")

    # -- CUDA: parity + determinism ---------------------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_c import compile_program
        from chain.emit_cuda import build_cu
        work = "/tmp/swiglu_op_cu"
        os.makedirs(work, exist_ok=True)
        UGf = np.ascontiguousarray(UG, dtype=np.float32)
        art = compile_program(t, "cuda", sample={"UG": UGf},
                              outputs=["MID"],
                              basedir=os.path.join(ROOT, "programs"))
        exe = build_cu(art["source"], work, name="sw",
                       units=art.get("units"))
        UGf.tofile(os.path.join(work, "f_UG.bin"))
        argv = [exe, os.path.join(work, "f_UG.bin"),
                os.path.join(work, "o_MID.bin")]
        outs = []
        for i in range(2):
            r = subprocess.run(argv, capture_output=True, text=True)
            if r.returncode != 0:
                check("swiglu-cu-run", False, r.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(work, "o_MID.bin"),
                                    dtype=np.float32))
        else:
            check("swiglu-cu-run", True, "")
            check("swiglu-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            exp = ref.astype(np.float32).reshape(-1)
            check("swiglu-cu-parity",
                  float(np.abs(outs[0] - exp).max()) < 1e-5,
                  f"maxabs={np.abs(outs[0] - exp).max():.2e}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
