"""TOPK gate: row-wise top-k over 2D F scores (50th mnemonic).

Sparse-attention primitive (phase 1): scores (R,S) -> (vals (R,k),
idx (R,k) int64), k int literal, ties-first deterministic. Gates:
lattice-ref vs torch (values 0-diff, indices gather-valid), ties
fixture, k=S identity, k=1 == argmax, k>S/k=0/1D/T fail loud,
builder tuple-out roundtrip, C vs lattice 0-diff (values + ties-first
indices), non-FPU refuses loud, CUDA vs lattice 0-diff (SKIPs without
nvcc/GPU).
Usage: python3 tests/test_topk.py
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

R, S, K = 7, 16, 8


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.normal(size=(R, S)) * 3)

    # -- lattice-ref vs torch --------------------------------------
    import torch
    fn, _, _ = REGISTRY["TOPK"]
    v, ix = fn([X, K], {}, {})
    ref_v, ref_i = torch.topk(torch.tensor(X), K, dim=1)
    ref_v, ref_i = ref_v.numpy(), ref_i.numpy()
    check("topk-vals", bool((v == ref_v).all()),
          f"maxabs={np.abs(v - ref_v).max():.2e}")
    back = X[np.arange(R)[:, None], ix]
    check("topk-idx-valid", bool((back == v).all()),
          "gather(idx) reproduces vals")
    check("topk-idx-sorted", bool((np.diff(v, axis=1) <= 0).all()),
          "descending")
    check("topk-shapes", v.shape == (R, K) and ix.shape == (R, K)
          and ix.dtype == np.int64, f"{v.shape} {ix.shape} {ix.dtype}")

    # -- ties-first deterministic -----------------------------------
    T = np.zeros((3, 6))
    T[:, 2] = 5.0
    T[:, 5] = 5.0
    T[1, 0] = 9.0
    tv, tix = fn([T, 3], {}, {})
    check("topk-ties", bool((tix[0] == [2, 5, 0]).all()
                             and (tix[2] == [2, 5, 0]).all()
                             and (tix[1] == [0, 2, 5]).all()),
          f"{tix.tolist()}")

    # -- k=S identity / k=1 == argmax -------------------------------
    vs, ixs = fn([X, S], {}, {})
    check("topk-full", bool((vs == np.sort(X, axis=1)[:, ::-1]).all()),
          "k=S sorts descending")
    v1, ix1 = fn([X, 1], {}, {})
    check("topk-one", bool((ix1[:, 0] == X.argmax(axis=1)).all())
          and bool((v1[:, 0] == X.max(axis=1)).all()), "k=1 == argmax")

    # -- fail loud ----------------------------------------------------
    for bad, tag in [([X, 0], "k=0"), ([X, S + 1], "k>S"),
                     ([X[:, 0], 2], "1D"),
                     ([np.zeros((2, 2, 2)), 1], "3D")]:
        try:
            fn(bad, {}, {})
            check(f"topk-{tag}-loud", False, "no raise")
        except (ValueError, IndexError):
            check(f"topk-{tag}-loud", True, "")

    # -- text form + builder roundtrip --------------------------------
    t = f"IN X\nTV, TI = TOPK(X, {K})\n"
    cfg, inp, bound, _ = ASM.assemble(t, REGISTRY, SIGS)
    check("topk-text", len(bound) == 1 and bound[0][1] == "TOPK"
          and bound[0][0] == ["TV", "TI"], f"{bound[0][0]}")
    from chain.builder import Prog
    p = Prog("topk-b")
    p.inp("X")
    p.op("TOPK", "X", K, out=("TV", "TI"))
    check("topk-builder", p.text() == t, repr(p.text()))
    got = ASM.run_text(t, REGISTRY, {"X": X}, sigs=SIGS)
    check("topk-run", bool((got["TV"] == v).all())
          and bool((got["TI"] == ix).all()), "run_text matches ref")

    # -- C backend: exact values + ties-first indices -----------------
    from chain.emit_c import NoPattern, build, compile_program
    from chain.emit_nonfpu import NonFPUBackend
    work = "/tmp/topk_c"
    os.makedirs(work, exist_ok=True)
    sdir = os.path.join(ROOT, "programs")
    art = compile_program(t, "c", sample={"X": X},
                          outputs=["TV", "TI"], basedir=sdir)
    exe = build(art["source"], work, name="topk")
    fx = os.path.join(work, "x.bin")
    fv = os.path.join(work, "tv.bin")
    fi = os.path.join(work, "ti.bin")
    X.tofile(fx)
    r = subprocess.run([exe, fx, fv, fi], capture_output=True, text=True)
    check("topk-c-run", r.returncode == 0, r.stderr[:200])
    if r.returncode == 0:
        dv = np.fromfile(fv, dtype=np.float64).reshape(R, K)
        di = np.fromfile(fi, dtype=np.int64).reshape(R, K)
        check("topk-c-vals", bool((dv == v).all()),
              f"maxabs={np.abs(dv - v).max():.2e}")
        check("topk-c-idx", bool((di == ix).all()),
              "C indices match lattice (ties-first)")

    # Ties fixture: C selection order must equal lattice stable order.
    tt = "IN X\nTV, TI = TOPK(X, 3)\n"
    artt = compile_program(tt, "c", sample={"X": T},
                           outputs=["TV", "TI"], basedir=sdir)
    exet = build(artt["source"], work, name="topkt")
    T.tofile(fx)
    r = subprocess.run([exet, fx, fv, fi], capture_output=True, text=True)
    check("topk-c-ties-run", r.returncode == 0, r.stderr[:200])
    if r.returncode == 0:
        dtv = np.fromfile(fv, dtype=np.float64).reshape(3, 3)
        dti = np.fromfile(fi, dtype=np.int64).reshape(3, 3)
        check("topk-c-ties", bool((dti == tix).all())
              and bool((dtv == tv).all()), f"{dti.tolist()}")

    # -- scale: the word vocabulary (S=2936, k=64) ---------------------
    # The grounded cost case: fillin's word bank is V=2936 and the
    # budgeted frontier asks top-k per row.  Reference = the ties-first
    # RULE (a stable descending argsort), not torch: the contract is
    # "lower index wins a tie", and torch's tie order is not the rule.
    S2, K2, R2 = 2936, 64, 6
    rng2 = np.random.default_rng(11)
    X2 = np.ascontiguousarray(
        rng2.integers(0, 40, size=(R2, S2)).astype(np.float64))
    X2[0] = np.round(X2[0] / 8.0) * 8.0     # a quantized row: many ties
    ord2 = np.argsort(-X2, axis=1, kind="stable")[:, :K2]
    ref2_v = X2[np.arange(R2)[:, None], ord2]
    t2 = f"IN X\nTV, TI = TOPK(X, {K2})\n"
    art2 = compile_program(t2, "c", sample={"X": X2},
                           outputs=["TV", "TI"], basedir=sdir)
    exe2 = build(art2["source"], work, name="topkscale")
    X2.tofile(fx)
    r = subprocess.run([exe2, fx, fv, fi], capture_output=True, text=True)
    check("topk-scale-run", r.returncode == 0, r.stderr[:200])
    if r.returncode == 0:
        dv2 = np.fromfile(fv, dtype=np.float64).reshape(R2, K2)
        di2 = np.fromfile(fi, dtype=np.int64).reshape(R2, K2)
        check("topk-scale-vals", bool((dv2 == ref2_v).all()),
              f"maxabs={np.abs(dv2 - ref2_v).max():.2e}")
        check("topk-scale-idx", bool((di2 == ord2).all()),
              f"S={S2} k={K2} ties-first")
        check("topk-scale-ties", bool((di2[0] == ord2[0]).all()),
              f"{di2[0, :8].tolist()}")

    # -- non-FPU refusal contract (integer-only target) ----------------
    try:
        NonFPUBackend().pattern("TOPK", ["TV", "TI"], ["X", str(K)], None,
                                {"streams": {}, "config": {}, "ids_n": "1"})
        check("topk-nonfpu-refuses", False, "no raise")
    except NoPattern:
        check("topk-nonfpu-refuses", True, "")

    # -- CUDA vs lattice -----------------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
    else:
        from chain.emit_c import compile_program
        from chain.emit_cuda import build_cu
        work = "/tmp/topk_cu"
        os.makedirs(work, exist_ok=True)
        # one (7,16) fixture incl. an embedded tie row (row 0: top pair
        # tied at 5.0 -> indices [0, 1] first, deterministic)
        Xf = np.ascontiguousarray(X, dtype=np.float16)
        Xf[0, :] = np.float16(-100.0)
        Xf[0, 0] = np.float16(5.0)
        Xf[0, 1] = np.float16(5.0)
        ev, eix = fn([np.ascontiguousarray(Xf, dtype=np.float64), K],
                       {}, {})
        art = compile_program(t, "cuda",
                              sample={"X": Xf}, outputs=["TV", "TI"],
                              basedir=os.path.join(ROOT, "programs"),
                              use_fp16=True)
        exe = build_cu(art["source"], work, name="topk",
                       units=art.get("units"))
        Xf.tofile(os.path.join(work, "f_X.bin"))
        argv = [exe, os.path.join(work, "f_X.bin"),
                os.path.join(work, "o_TV.bin"),
                os.path.join(work, "o_TI.bin")]
        r = subprocess.run(argv, capture_output=True, text=True)
        check("topk-cu-run", r.returncode == 0, r.stderr[:200])
        if r.returncode == 0:
            dv = np.fromfile(os.path.join(work, "o_TV.bin"),
                             dtype=np.float32)
            di = np.fromfile(os.path.join(work, "o_TI.bin"),
                             dtype=np.int64)
            check("topk-cu-shapes", dv.shape == (R * K,)
                  and di.shape == (R * K,), f"{dv.shape} {di.shape}")
            dv = dv.reshape(R, K).astype(np.float64)
            di = di.reshape(R, K)
            # fp16 storage rounds the input: lattice reference runs on
            # the same rounded values; fp32 device compute is then exact.
            expv = ev.astype(np.float32).astype(np.float64)
            check("topk-cu-vals", bool((dv == expv).all()),
                  f"maxabs={np.abs(dv - expv).max():.2e}")
            check("topk-cu-idx", bool((di == eix).all()),
                  "bit-exact indices incl. ties")
            check("topk-cu-ties", bool((di[0, :2] == [0, 1]).all()),
                  f"{di[0, :4].tolist()}")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
