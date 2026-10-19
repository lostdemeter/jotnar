"""CONV gate (pattern P1): single-channel 2D, edge-replicate, C + CUDA.

CONV(X (H,W), K (kH,kW)) odd kernel within image, same-size output,
stride 1, dilation 1, no bias (v0.1 scope; pad/stride/dilation are
NOT parameters — misuse fails loudly). Audit first, per the card:
edge-replicate == torch replicate_pad EXACT; torch zero-pad
diverges at borders (interior identical) — the falsifier (zero-pad
equivalence) does NOT fire. Gates: audit record legs, C vs torch
(40dB), CUDA vs torch (40dB) + bit-exact rerun, fail loud (even
kernel, oversize kernel, non-2D, non-F kind, extra args at
assemble), non-FPU still refuses (contract unchanged).
Usage: python3 tests/test_conv.py
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

H, W, KH, KW = 10, 12, 3, 3


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def db(ref, got):
    import math
    mse = float(np.mean((np.asarray(ref, dtype=np.float64).ravel()
                         - np.asarray(got, dtype=np.float64).ravel()) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-300))


def rep_conv(x, k):
    p = np.pad(x, 1, mode="edge")
    o = np.zeros_like(x)
    for dy in range(3):
        for dx in range(3):
            o += k[dy, dx] * p[dy:dy + x.shape[0], dx:dx + x.shape[1]]
    return o


def cases():
    rng = np.random.default_rng(71)
    d = {}
    d["gradient"] = np.tile(np.linspace(0, 1, W), (H, 1))
    d["step"] = (np.mgrid[0:H, 0:W][0] >= H // 2).astype(float)
    d["impulse"] = np.zeros((H, W))
    d["impulse"][H // 2, W // 2] = 1.0
    d["flat"] = np.full((H, W), 0.37)
    d["corner"] = np.zeros((H, W))
    d["corner"][0, 0] = 1.0
    d["noise"] = rng.random((H, W))
    return d


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import torch
    import torch.nn.functional as F
    K = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0

    # -- audit: boundary semantics vs torch --------------------------------
    rep_ok, zerodiv, inteq = True, 0.0, True
    for name, x in cases().items():
        t = torch.tensor(x, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        kk = torch.tensor(K, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        zero = F.conv2d(t, kk, padding=1).numpy()[0, 0]
        rp = F.conv2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), kk).numpy()[0, 0]
        ours = rep_conv(x, K)
        rep_ok &= bool((ours == rp).all())
        zerodiv = max(zerodiv, float(np.abs(ours - zero).max()))
        inteq &= bool((ours[1:-1, 1:-1] == zero[1:-1, 1:-1]).all())
    check("conv-audit-replicate", bool(rep_ok),
          "edge-replicate == torch replicate_pad bit-exact x6")
    check("conv-audit-zero-diverges", zerodiv > 1e-3,
          f"max border div vs zero-pad {zerodiv:.2e} (falsifier: equivalence)")
    check("conv-audit-interior", bool(inteq), "interior identical, all cases")

    # -- text form ------------------------------------------------------------
    t = "IN X\nIN K\nO = CONV(X, K)\n"
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("conv-text", len(bound) == 1 and bound[0][1] == "CONV"
          and bound[0][0] == ["O"], f"{bound[0][0]}")

    # -- fail loud --------------------------------------------------------------
    from chain.asm import AsmError
    from chain.emit_c import CBackend, NoPattern, compile_program
    from chain.emit_nonfpu import NonFPUBackend
    X = np.zeros((H, W))
    Kev = np.zeros((2, 3))
    Kbig = np.zeros((H + 2, W + 2))
    X1 = np.zeros((W,))
    try:
        compile_program(t, "c",
                        sample={"X": X, "K": Kev}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("conv-even-loud", False, "even kernel accepted?!")
    except AsmError:
        check("conv-even-loud", True, "even kernel refused")
    try:
        compile_program(t, "c",
                        sample={"X": X, "K": Kbig}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("conv-oversize-loud", False, "oversize kernel accepted?!")
    except AsmError:
        check("conv-oversize-loud", True, "oversize kernel refused")
    try:
        compile_program(t, "c",
                        sample={"X": X1, "K": K}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("conv-rank-loud", False, "1D input accepted?!")
    except (AsmError, NoPattern):
        check("conv-rank-loud", True, "non-2D refused")
    try:
        be = CBackend()
        be.pattern("CONV", ["O"], ["X", "K"], None,
                   {"streams": {"X": ("T", "double", (H, W)),
                                "K": ("T", "double", (KH, KW))},
                    "config": {}, "ids_n": "1"})
        check("conv-tkind-loud", False, "T input accepted?!")
    except NoPattern:
        check("conv-tkind-loud", True, "non-F input refused")
    try:
        ASM.assemble("IN X\nIN K\nO = CONV(X, K, 1)\n", REGISTRY, SIGS)
        check("conv-arity-loud", False, "extra args accepted?!")
    except (AsmError, ValueError, IndexError):
        check("conv-arity-loud", True, "pad/stride/dilation not parameters")
    try:
        NonFPUBackend().pattern(
            "CONV", ["O"], ["X", "K"], None,
            {"streams": {}, "config": {}, "ids_n": "1"})
        check("conv-nonfpu-unchanged", False, "non-FPU now emits?!")
    except NoPattern as e:
        check("conv-nonfpu-unchanged", "no integer pattern" in str(e),
              "still refuses (contract unchanged)")

    # -- C backend: parity --------------------------------------------------------
    sample = {"X": cases()["noise"].astype(np.float64),
              "K": K.astype(np.float64)}
    art = compile_program(t, "c", sample=sample, outputs=["O"],
                          basedir=os.path.join(ROOT, "programs"))
    from chain.emit_c import build as _bc
    work = "/tmp/conv_c"
    os.makedirs(work, exist_ok=True)
    exe = _bc(art["source"], work, name="conv")
    f1 = os.path.join(work, "x.bin")
    f2 = os.path.join(work, "k.bin")
    fo = os.path.join(work, "o.bin")
    sample["X"].tofile(f1)
    sample["K"].tofile(f2)
    r = subprocess.run([exe, f1, f2, fo], capture_output=True, text=True)
    check("conv-c-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:150]}")
    if r.returncode == 0:
        gotc = np.fromfile(fo, dtype=np.float64).reshape(H, W)
        ref = rep_conv(sample["X"], sample["K"])
        check("conv-c-parity", db(ref, gotc) >= 40,
              f"{db(ref, gotc):.1f}dB (C vs replicate)")

    # -- CUDA backend: parity + determinism -----------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP cuda legs (needs nvcc + GPU)")
    else:
        from chain.emit_cuda import build_cu
        worku = "/tmp/conv_cu"
        os.makedirs(worku, exist_ok=True)
        Xf = np.ascontiguousarray(sample["X"], dtype=np.float32)
        Kf = np.ascontiguousarray(sample["K"], dtype=np.float32)
        artu = compile_program(t, "cuda",
                               sample={"X": Xf, "K": Kf},
                               outputs=["O"],
                               basedir=os.path.join(ROOT, "programs"))
        exeu = build_cu(artu["source"], worku, name="conv",
                        units=artu.get("units"))
        Xf.tofile(os.path.join(worku, "f_X.bin"))
        Kf.tofile(os.path.join(worku, "f_K.bin"))
        argv = [exeu, os.path.join(worku, "f_X.bin"),
                os.path.join(worku, "f_K.bin"),
                os.path.join(worku, "o_O.bin")]
        outs = []
        for i in range(2):
            rr = subprocess.run(argv, capture_output=True, text=True)
            if rr.returncode != 0:
                check("conv-cu-run", False, rr.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(worku, "o_O.bin"),
                                    dtype=np.float32))
        else:
            check("conv-cu-run", True, "")
            check("conv-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            ref32 = rep_conv(Xf.astype(np.float64), Kf.astype(np.float64))
            check("conv-cu-parity", db(ref32, outs[0].reshape(H, W)) >= 40,
                  f"{db(ref32, outs[0].reshape(H, W)):.1f}dB (CUDA vs replicate)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
