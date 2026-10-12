"""ROPE gate: strided per-head rotary (53rd mnemonic).

ROPE(x (R,nh*dh), pos, nh, dh) rotates each head in place: same math
as nh per-head ROTARY ops + CONCAT rejoin, one pass, no split/copy.
Kills ~90 copy ops per decode layer. Gates: lattice vs explicit
per-head reference (exact), vs torch rotary (eps), CUDA vs lattice
(eps) + bit-determinism (elementwise), fail loud (width/pos/dh),
C + non-FPU refuse loud (contract).
Usage: python3 tests/test_rope.py
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

R, NH, DH = 3, 4, 8
HID = NH * DH
BASE = 1000000.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def ref_rope(x, pos, nh, dh, base=BASE):
    th = np.power(float(base), -2.0 * np.arange(dh // 2) / dh)
    xr = x.reshape(-1, nh, dh)
    ang = pos.reshape(-1, 1, 1) * th.reshape(1, 1, -1)
    c, s = np.cos(ang), np.sin(ang)
    xe, xo = xr[..., 0::2], xr[..., 1::2]
    o = np.empty_like(xr)
    o[..., 0::2] = xe * c - xo * s
    o[..., 1::2] = xe * s + xo * c
    return o.reshape(x.shape)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(53)
    X = np.ascontiguousarray(rng.normal(size=(R, HID)))
    P = np.ascontiguousarray(np.array([0, 3, 7]))

    fn = REGISTRY["ROPE"][0]
    got = fn([X, P, NH, DH], {"rope_base": str(BASE)}, {})
    exp = ref_rope(X, P, NH, DH)
    check("rope-ref", bool((got == exp).all()),
          f"maxabs={np.abs(got - exp).max():.2e} (exact)")

    # -- vs torch rotary --------------------------------------------------
    import torch
    th = torch.pow(torch.tensor(float(BASE), dtype=torch.float64),
                   -2.0 * torch.arange(DH // 2, dtype=torch.float64)
                   / float(DH))
    ang = torch.tensor(P).double().unsqueeze(1) * th.unsqueeze(0)
    ang = ang.unsqueeze(1).expand(R, NH, DH // 2)
    c, s = torch.cos(ang), torch.sin(ang)
    xr = torch.tensor(X, dtype=torch.float64).reshape(R, NH, DH)
    xe, xo = xr[..., 0::2], xr[..., 1::2]
    te = torch.empty_like(xr)
    te[..., 0::2] = xe * c - xo * s
    te[..., 1::2] = xe * s + xo * c
    check("rope-torch", bool((got == te.numpy().reshape(R, HID)).all()),
          "independent implementation agrees")

    # -- fail loud ----------------------------------------------------------
    for bad, tag in [([X, P, NH, 7], "odd-dh"),
                     ([X[:, :8], P, NH, DH], "width"),
                     ([X, np.array([0, 1]), NH, DH], "pos"),
                     ([X.reshape(2, 8, 6), P, NH, DH], "3D")]:
        try:
            fn(bad, {"rope_base": str(BASE)}, {})
            check(f"rope-{tag}-loud", False, "no raise")
        except (ValueError, IndexError):
            check(f"rope-{tag}-loud", True, "")

    # -- text form ------------------------------------------------------------
    t = (f"CONFIG rope_base {BASE}\nIN X\nIN P\nQR = ROPE(X, P, {NH}, "
         f"{DH})\n")
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("rope-text", len(bound) == 1 and bound[0][1] == "ROPE"
          and bound[0][0] == ["QR"], f"{bound[0][0]}")
    out = ASM.run_text(t, REGISTRY, {"X": X, "P": P}, sigs=SIGS)
    check("rope-run", bool((out["QR"] == exp).all()), "run_text matches")

    # -- backend refusal contract ----------------------------------------------
    from chain.emit_c import CBackend, NoPattern
    from chain.emit_nonfpu import NonFPUBackend
    for be, tag in ((CBackend(), "c"), (NonFPUBackend(), "nonfpu")):
        try:
            be.pattern("ROPE", ["QR"], ["X", "P", str(NH), str(DH)], None,
                       {"streams": {}, "config": {}, "ids_n": "1"})
            check(f"rope-{tag}-refuses", False, "no raise")
        except NoPattern:
            check(f"rope-{tag}-refuses", True, "")

    # -- CUDA: parity + determinism ----------------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_c import compile_program
        from chain.emit_cuda import build_cu
        work = "/tmp/rope_cu"
        os.makedirs(work, exist_ok=True)
        Xf = np.ascontiguousarray(X, dtype=np.float32)
        art = compile_program(t, "cuda", sample={"X": Xf, "P": P},
                              outputs=["QR"],
                              basedir=os.path.join(ROOT, "programs"))
        exe = build_cu(art["source"], work, name="rope",
                       units=art.get("units"))
        Xf.tofile(os.path.join(work, "f_X.bin"))
        P.tofile(os.path.join(work, "f_P.bin"))
        argv = [exe, os.path.join(work, "f_X.bin"),
                os.path.join(work, "f_P.bin"),
                os.path.join(work, "o_QR.bin")]
        outs = []
        for i in range(2):
            r = subprocess.run(argv, capture_output=True, text=True)
            if r.returncode != 0:
                check("rope-cu-run", False, r.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(work, "o_QR.bin"),
                                    dtype=np.float32))
        else:
            check("rope-cu-run", True, "")
            check("rope-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            check("rope-cu-parity",
                  float(np.abs(outs[0] - exp.astype(np.float32)
                               .reshape(-1)).max()) < 1e-4,
                  f"maxabs={np.abs(outs[0] - exp.astype(np.float32).reshape(-1)).max():.2e}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
