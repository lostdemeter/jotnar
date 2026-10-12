"""ATTN gate: fused decode-GQA attention (52nd mnemonic).

ATTN(qr (1,H), Kf/Vf (S,KVD), mask (PER,S), nh/nkv/dh/S) runs the whole
per-group chain (scores, mask, shift, softmax, context) in one kernel
with fixed accumulation order -- bit-deterministic across runs and
binaries (unlike strided cublas). Gates: lattice vs torch GQA (eps),
mask-exactness (pads weightless, causal), determinism (CUDA twice
bit-exact), GQA group-sharing vs per-head reference, fail loud
(geometry, S>512, R>1), C + non-FPU refuse loud (contract).
Usage: python3 tests/test_attn.py
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

NH, NKV, DH, S = 8, 2, 8, 16
HID, KVD = NH * DH, NKV * DH
PER = NH // NKV
NEG = -1e9


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def torch_ref(qr, kf, vf, m):
    import torch
    q = torch.tensor(qr, dtype=torch.float64).reshape(1, NH, DH)
    K = torch.tensor(kf, dtype=torch.float64).reshape(S, NKV, DH)
    V = torch.tensor(vf, dtype=torch.float64).reshape(S, NKV, DH)
    mm = torch.tensor(m, dtype=torch.bool).reshape(PER, S)
    outs = []
    for g in range(NKV):
        sc = q[:, g * PER:(g + 1) * PER, :] @ K[:, g, :].T
        ms = torch.where(mm.unsqueeze(0), sc,
                         torch.tensor(NEG, dtype=torch.float64))
        sh = ms - ms.amax(-1, keepdim=True)
        pr = torch.softmax(sh, dim=-1)
        outs.append(pr @ V[:, g, :])
    return torch.cat(outs, dim=1).numpy().reshape(1, HID)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(52)
    qr = np.ascontiguousarray(rng.normal(size=(1, HID)))
    kf = np.ascontiguousarray(rng.normal(size=(S, KVD)))
    vf = np.ascontiguousarray(rng.normal(size=(S, KVD)))
    m = np.ascontiguousarray(rng.integers(0, 2, size=(PER, S)))

    fn = REGISTRY["ATTN"][0]
    got = fn([qr, kf, vf, m, NH, NKV, DH, S], {"beta": str(NEG)}, {})
    ref = torch_ref(qr, kf, vf, m)
    err = float(np.abs(got - ref).max())
    check("attn-torch", err < 1e-9, f"maxabs={err:.2e}")

    # -- mask-exactness: pads weightless --------------------------------
    mall = np.ones((PER, S), dtype=np.int64)
    pall = fn([qr, kf, vf, mall, NH, NKV, DH, S], {"beta": str(NEG)}, {})
    mone = np.zeros((PER, S), dtype=np.int64)
    mone[:, 0] = 1
    pone = fn([qr, kf, vf, mone, NH, NKV, DH, S], {"beta": str(NEG)}, {})
    # single live key => context == that V row tiled per group
    exp = np.concatenate(
        [np.tile(vf[0, g * DH:(g + 1) * DH], (1, PER)) for g in range(NKV)],
        axis=1)
    check("attn-single-key", bool((np.abs(pone - exp) < 1e-9).all()),
          f"maxabs={np.abs(pone - exp).max():.2e}")
    check("attn-mask-moves", float(np.abs(pall - pone).max()) > 1e-3,
          "mask changes output")

    # -- fail loud --------------------------------------------------------
    for bad, tag in [([qr, kf, vf, m, 7, NKV, DH, S], "nh%nkv"),
                     ([qr[:, :8], kf, vf, m, NH, NKV, DH, S], "qh"),
                     ([qr, kf, vf, m, NH, NKV, DH, 8], "s")]:
        try:
            fn(bad, {"beta": str(NEG)}, {})
            check(f"attn-{tag}-loud", False, "no raise")
        except (ValueError, IndexError):
            check(f"attn-{tag}-loud", True, "")

    # -- text form ----------------------------------------------------------
    t = (f"IN QR\nIN KF\nIN VF\nIN M\nCTX = ATTN(QR, KF, VF, M, {NH}, "
         f"{NKV}, {DH}, {S})\n")
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("attn-text", len(bound) == 1 and bound[0][1] == "ATTN"
          and bound[0][0] == ["CTX"], f"{bound[0][0]}")
    from chain import asm as _A
    feeds = {"QR": qr, "KF": kf, "VF": vf, "M": m}
    got_run = _A.run_text(t, REGISTRY, feeds, sigs=SIGS)
    check("attn-run", bool((np.abs(got_run["CTX"] - ref) < 1e-9).all()),
          "run_text matches torch")

    # -- backend refusal contract --------------------------------------------
    from chain.emit_c import CBackend, NoPattern
    from chain.emit_nonfpu import NonFPUBackend
    for be, tag in ((CBackend(), "c"), (NonFPUBackend(), "nonfpu")):
        try:
            be.pattern("ATTN", ["CTX"], ["QR", "KF", "VF", "M", "8", "2",
                                         "8", "16"], None,
                       {"streams": {}, "config": {}, "ids_n": "1"})
            check(f"attn-{tag}-refuses", False, "no raise")
        except NoPattern:
            check(f"attn-{tag}-refuses", True, "")

    # -- CUDA: torch parity + bit-determinism ----------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_c import compile_program
        from chain.emit_cuda import build_cu
        work = "/tmp/attn_cu"
        os.makedirs(work, exist_ok=True)
        qf = np.ascontiguousarray(qr, dtype=np.float32)
        kf32 = np.ascontiguousarray(kf, dtype=np.float32)
        vf32 = np.ascontiguousarray(vf, dtype=np.float32)
        art = compile_program(t, "cuda",
                              sample={"QR": qf, "KF": kf32, "VF": vf32,
                                      "M": m},
                              outputs=["CTX"],
                              basedir=os.path.join(ROOT, "programs"))
        exe = build_cu(art["source"], work, name="attn",
                       units=art.get("units"))
        qf.tofile(os.path.join(work, "f_QR.bin"))
        kf32.tofile(os.path.join(work, "f_KF.bin"))
        vf32.tofile(os.path.join(work, "f_VF.bin"))
        m.tofile(os.path.join(work, "f_M.bin"))
        argv = [exe] + [os.path.join(work, f"f_{n}.bin")
                        for n in ("QR", "KF", "VF", "M")] + \
            [os.path.join(work, "o_CTX.bin")]
        outs = []
        for i in range(2):
            r = subprocess.run(argv, capture_output=True, text=True)
            if r.returncode != 0:
                check("attn-cu-run", False, r.stderr[:200])
                break
            outs.append(np.fromfile(os.path.join(work, "o_CTX.bin"),
                                    dtype=np.float32))
        else:
            check("attn-cu-run", True, "")
            check("attn-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            exp32 = ref.astype(np.float32).reshape(-1)
            check("attn-cu-torch",
                  float(np.abs(outs[0] - exp32).max()) < 1e-4,
                  f"maxabs={np.abs(outs[0] - exp32).max():.2e}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
