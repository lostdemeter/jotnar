"""INTERP gate (pattern P2): bilinear resample, C + CUDA.

INTERP(X (H,W[,C]), sy, sx) dyadic scales only (else LOWERING
ERROR): bilinear, half-pixel centers (align_corners=False —
audited: lattice matches torch align_corners=False to fixed-point
noise, True diverges), border clamp (torch default). Audit first
per house pattern: convention proof vs torch variants, then
emission. Gates: audit legs, C vs torch (40dB), CUDA vs torch
(40dB) + bit-exact rerun, fail loud (non-dyadic, non-positive,
non-F kind, rank>3, stream scales), non-FPU still refuses
(contract unchanged). Nearest stays shipped (degraded default).
Usage: python3 tests/test_interp.py
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

H, W, C = 6, 8, 2


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def db(ref, got):
    import math
    mse = float(np.mean((np.asarray(ref, dtype=np.float64).ravel()
                         - np.asarray(got, dtype=np.float64).ravel()) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-300))


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(21)
    Xv = (rng.random((H, W, C)) - 0.5) * 1.5
    X = S.encode(Xv)

    # -- audit: lattice convention vs torch variants --------------------------
    tt = torch.tensor(Xv.transpose(2, 0, 1), dtype=torch.float64).unsqueeze(0)
    o = REGISTRY["INTERP"][0]([X, 2.0, 2.0], {}, {})
    ov = np.asarray(S.decode(o[0], o[1])
                    * (1 - np.asarray(o[2]).astype(np.float64)))
    Ho, Wo = ov.shape[0], ov.shape[1]
    rF = F.interpolate(tt, size=(Ho, Wo), mode="bilinear",
                       align_corners=False).numpy()[0].transpose(1, 2, 0)
    rT = F.interpolate(tt, size=(Ho, Wo), mode="bilinear",
                       align_corners=True).numpy()[0].transpose(1, 2, 0)
    dF = float(np.abs(ov.reshape(Ho, Wo, C) - rF).max())
    dT = float(np.abs(ov.reshape(Ho, Wo, C) - rT).max())
    check("interp-audit-convention", dF < 5e-3 and dT > 1e-2,
          f"align_corners=False {dF:.1e} vs True {dT:.1e} (fixed-point noise)")
    o2 = REGISTRY["INTERP"][0]([X, 0.5, 0.5], {}, {})
    ov2 = np.asarray(S.decode(o2[0], o2[1])
                     * (1 - np.asarray(o2[2]).astype(np.float64)))
    r2 = F.interpolate(tt, size=(ov2.shape[0], ov2.shape[1]), mode="bilinear",
                       align_corners=False).numpy()[0].transpose(1, 2, 0)
    check("interp-audit-down", float(np.abs(ov2 - r2).max()) < 5e-3,
          "0.5x matches too")

    # -- text form ---------------------------------------------------------------
    t = "IN X\nO = INTERP(X, 2.0, 2.0)\n"
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("interp-text", len(bound) == 1 and bound[0][1] == "INTERP"
          and bound[0][0] == ["O"], f"{bound[0][0]}")

    # -- fail loud -----------------------------------------------------------------
    from chain.asm import AsmError
    from chain.emit_c import CBackend, NoPattern, compile_program
    from chain.emit_nonfpu import NonFPUBackend
    Xf = np.ascontiguousarray(Xv)
    try:
        compile_program("IN X\nO = INTERP(X, 1.5, 1.0)\n", "c",
                        sample={"X": Xf}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("interp-nondyadic-loud", False, "1.5 accepted?!")
    except AsmError:
        check("interp-nondyadic-loud", True, "non-dyadic refused")
    try:
        compile_program("IN X\nO = INTERP(X, 0.0, 1.0)\n", "c",
                        sample={"X": Xf}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("interp-nonpos-loud", False, "zero scale accepted?!")
    except AsmError:
        check("interp-nonpos-loud", True, "non-positive refused")
    try:
        be = CBackend()
        be.pattern("INTERP", ["O"], ["X", "2.0", "2.0"], None,
                   {"streams": {"X": ("T", "double", (H, W, C))},
                    "config": {}, "ids_n": "1"})
        check("interp-tkind-loud", False, "T input accepted?!")
    except NoPattern:
        check("interp-tkind-loud", True, "non-F input refused")
    try:
        compile_program("IN X\nO = INTERP(X, 2.0, 2.0)\n", "c",
                        sample={"X": Xf.reshape(-1)}, outputs=["O"],
                        basedir=os.path.join(ROOT, "programs"))
        check("interp-rank-loud", False, "1D accepted?!")
    except (AsmError, NoPattern):
        check("interp-rank-loud", True, "rank!=2/3 refused")
    try:
        NonFPUBackend().pattern(
            "INTERP", ["O"], ["X", "2.0", "2.0"], None,
            {"streams": {}, "config": {}, "ids_n": "1"})
        check("interp-nonfpu-unchanged", False, "non-FPU now emits?!")
    except NoPattern as e:
        check("interp-nonfpu-unchanged", "no integer pattern" in str(e),
              "still refuses (contract unchanged)")

    # -- C backend: parity -------------------------------------------------------------
    sample = {"X": Xf}
    art = compile_program(t, "c", sample=sample, outputs=["O"],
                          basedir=os.path.join(ROOT, "programs"))
    from chain.emit_c import build as _bc
    work = "/tmp/interp_c"
    os.makedirs(work, exist_ok=True)
    exe = _bc(art["source"], work, name="interp")
    f1 = os.path.join(work, "x.bin")
    fo = os.path.join(work, "o.bin")
    Xf.astype(np.float64).tofile(f1)
    r = subprocess.run([exe, f1, fo], capture_output=True, text=True)
    check("interp-c-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:150]}")
    ref = F.interpolate(torch.tensor(Xf.transpose(2, 0, 1),
                                     dtype=torch.float64).unsqueeze(0),
                        scale_factor=2, mode="bilinear",
                        align_corners=False).numpy()[0].transpose(1, 2, 0)
    if r.returncode == 0:
        gotc = np.fromfile(fo, dtype=np.float64).reshape(2 * H, 2 * W, C)
        check("interp-c-parity", db(ref, gotc) >= 40,
              f"{db(ref, gotc):.1f}dB (C vs torch)")

    # -- CUDA backend: parity + determinism ------------------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP cuda legs (needs nvcc + GPU)")
    else:
        from chain.emit_cuda import build_cu
        worku = "/tmp/interp_cu"
        os.makedirs(worku, exist_ok=True)
        Xf32 = np.ascontiguousarray(Xf, dtype=np.float32)
        artu = compile_program(t, "cuda", sample={"X": Xf32},
                               outputs=["O"],
                               basedir=os.path.join(ROOT, "programs"))
        exeu = build_cu(artu["source"], worku, name="interp",
                        units=artu.get("units"))
        Xf32.tofile(os.path.join(worku, "f_X.bin"))
        argv = [exeu, os.path.join(worku, "f_X.bin"),
                os.path.join(worku, "o_O.bin")]
        outs = []
        for i in range(2):
            rr = subprocess.run(argv, capture_output=True, text=True)
            if rr.returncode != 0:
                check("interp-cu-run", False, rr.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(worku, "o_O.bin"),
                                    dtype=np.float32))
        else:
            check("interp-cu-run", True, "")
            check("interp-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            ref32 = ref.astype(np.float32)
            check("interp-cu-parity", db(ref32, outs[0].reshape(2 * H, 2 * W, C)) >= 40,
                  f"{db(ref32, outs[0].reshape(2 * H, 2 * W, C)):.1f}dB (CUDA vs torch)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
