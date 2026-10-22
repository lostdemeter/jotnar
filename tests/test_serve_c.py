"""C serve gate: resident C binaries, no CUDA required.

This is the CPU counterpart to tests/test_serve.py.  The C emitter's
one-shot path is the authoritative integer/float path for the testbed,
so residency must be proven there first:

  - live=["tok"] compiles and reports its mode
  - step 0 matches a one-shot binary
  - a live rewrite is followed on the next step
  - a frozen bank poisoned mid-run does not move outputs (load-once)
  - same live bytes replay byte-identically
  - a wrong-size live file fails loud (fixed geometry)
  - MATMUL steps do not accumulate (beta=0 contract)
  - served TOPK returns the lattice top-k (scores + indices)
  - packed outputs carry top-k values and indices in one file
  - graph/sync_each stay CUDA-only; live inputs are validated

Usage: python3 tests/test_serve_c.py
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.builder import Prog
from chain.emit_c import NoPattern, build, compile_program
from chain.serve import ServeError, ServedExe

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def dump_payload(work, art, payload, tag):
    """Write C argv inputs/outputs; F is binary64, I/T components int64."""
    argv, files = [], {}
    for n in art["inputs"]:
        k = art["streams"][n][0]
        fn = os.path.join(work, f"{tag}_{n}.bin")
        if k == "F":
            np.ascontiguousarray(payload[n], dtype=np.float64).tofile(fn)
        elif k == "I":
            np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
        else:
            raise AssertionError(f"dump_payload cannot write T input {n}")
        argv.append(fn)
        files[n] = fn
    outs = []
    for o in art["outputs"]:
        ok = art["streams"][o][0]
        if ok == "T":
            raise AssertionError(f"dump_payload cannot read T output {o}")
        fn = os.path.join(work, f"{tag}out_{o}.bin")
        outs.append(fn)
        argv.append(fn)
    return argv, files, outs


def oneshot(art, payload, work, name):
    exe = build(art["source"], work, name=name, libs=art["libs"])
    argv, _, outs = dump_payload(work, art, payload, name)
    r = subprocess.run([exe] + argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:500]
    vals = []
    for o, fn in zip(art["outputs"], outs):
        dtype = np.float64 if art["streams"][o][0] == "F" else np.int64
        vals.append(np.fromfile(fn, dtype=dtype))
    return vals


def test_bigram(work):
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    rng = np.random.default_rng(7)
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(sdir, "bigram_lm.asm")).read()
    bank = np.ascontiguousarray(counts, dtype=np.float64)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks1 = np.array(sorted(rng.choice(nz, size=50, replace=False).tolist()),
                     np.int64)
    toks2 = np.array(sorted(rng.choice(nz, size=50, replace=False).tolist()),
                     np.int64)

    art = compile_program(text, "c",
                          sample={"tok": toks1, "bank": bank},
                          outputs=["OUT"], basedir=sdir)
    ref1, = oneshot(art, {"tok": toks1, "bank": bank}, work, "c_one1")
    ref2, = oneshot(art, {"tok": toks2, "bank": bank}, work, "c_one2")

    arts = compile_program(text, "c",
                           sample={"tok": toks1, "bank": bank},
                           outputs=["OUT"], basedir=sdir, live=["tok"])
    check("c-serve-live-flag", arts.get("live") == ["tok"],
          f"{arts.get('live')}")
    check("c-serve-no-graph", arts.get("graph") is False, "graph=False")
    exe = build(arts["source"], work, name="c_serve", libs=arts["libs"])
    argv, files, outs = dump_payload(work, arts,
                                     {"tok": toks1, "bank": bank}, "c_sv")

    sv = ServedExe(exe, argv, err_path=os.path.join(work, "c_serve.err"))
    try:
        t0 = time.perf_counter()
        sv.start()
        t_load = time.perf_counter() - t0
        print(f"c-serve-load-first-step: {t_load:.4f}s", flush=True)
        got1 = np.fromfile(outs[0], dtype=np.int64)
        check("c-serve-step0-match", bool((got1 == ref1).all()),
              f"{int((got1 == ref1).sum())}/{len(ref1)} vs one-shot")

        toks2.astype(np.int64).tofile(files["tok"])
        t0 = time.perf_counter()
        sv.step()
        t_step = time.perf_counter() - t0
        got2 = np.fromfile(outs[0], dtype=np.int64)
        check("c-serve-step1-match", bool((got2 == ref2).all()),
              f"{t_step:.4f}s/step "
              f"({int((got2 == ref2).sum())}/{len(ref2)})")

        # Frozen-proof: poison the bank file; outputs must NOT move.
        rng.standard_normal(bank.shape).astype(np.float64).tofile(files["bank"])
        toks1.astype(np.int64).tofile(files["tok"])
        sv.step()
        got3 = np.fromfile(outs[0], dtype=np.int64)
        check("c-serve-frozen-proof", bool((got3 == ref1).all()),
              "bank poisoned mid-run, outputs unmoved (read-once proven)")

        sv.step()
        got4 = np.fromfile(outs[0], dtype=np.int64)
        check("c-serve-deterministic", bool((got4 == got3).all()),
              "replay-identical")

        # Fixed geometry: wrong-size live file must fail loud.
        np.array([1, 2], dtype=np.int64).tofile(files["tok"])
        try:
            sv.step()
            check("c-serve-fixed-shape-loud", False, "wrong size accepted")
        except ServeError as e:
            check("c-serve-fixed-shape-loud", "rc=10" in str(e),
                  str(e)[:160])
    except ServeError as e:
        check("c-serve-protocol", False, str(e)[:200])
    finally:
        sv.close()

    # -- MATMUL overwrite gate (beta=0): loop steps must not accumulate.
    p = Prog("c-serve-mm")
    p.inp("x", "w")
    y = p.op("MATMUL", "x", "w", out="Y")
    rng2 = np.random.default_rng(3)
    mx = np.ascontiguousarray(rng2.normal(size=(4, 8)))
    mw = np.ascontiguousarray(rng2.normal(size=(8, 6)))
    artm = compile_program(p.text(), "c", sample={"x": mx, "w": mw},
                           outputs=[y], basedir=sdir)
    refm, = oneshot(artm, {"x": mx, "w": mw}, work, "c_onemm")
    artml = compile_program(p.text(), "c", sample={"x": mx, "w": mw},
                            outputs=[y], basedir=sdir, live=["x"])
    exem = build(artml["source"], work, name="c_servemm",
                 libs=artml["libs"])
    argv, files, outs = dump_payload(work, artml,
                                     {"x": mx, "w": mw}, "c_svmm")
    svm = ServedExe(exem, argv, err_path=os.path.join(work, "c_servemm.err"))
    try:
        svm.start()
        m0 = np.fromfile(outs[0], dtype=np.float64).reshape(4, 6)
        check("c-serve-mm-step0", np.allclose(m0, refm.reshape(4, 6),
                                              atol=1e-12),
              "loop step0 == one-shot")
        svm.step()
        m1 = np.fromfile(outs[0], dtype=np.float64).reshape(4, 6)
        check("c-serve-mm-no-accum", np.array_equal(m0, m1),
              f"maxdiff {np.abs(m0 - m1).max():.2e} (beta=1 drifts here)")
        mx2 = np.ascontiguousarray(rng2.normal(size=(4, 8)))
        mx2.astype(np.float64).tofile(files["x"])
        svm.step()
        m2 = np.fromfile(outs[0], dtype=np.float64).reshape(4, 6)
        refm2, = oneshot(artm, {"x": mx2, "w": mw}, work, "c_onemm2")
        check("c-serve-mm-follows",
              np.allclose(m2, refm2.reshape(4, 6), atol=1e-12),
              "loop tracks new live inputs")
    except ServeError as e:
        check("c-serve-mm-protocol", False, str(e)[:200])
    finally:
        svm.close()


def test_topk(work):
    R, S, K = 7, 16, 4
    rng = np.random.default_rng(11)
    X = np.ascontiguousarray(rng.normal(size=(R, S)) * 3)
    X2 = np.ascontiguousarray(rng.normal(size=(R, S)) * 3)
    fn, _, _ = REGISTRY["TOPK"]
    ref_v, ref_i = fn([X, K], {}, {})
    ref_v2, ref_i2 = fn([X2, K], {}, {})
    text = f"IN X\nTV, TI = TOPK(X, {K})\n"
    sdir = os.path.join(ROOT, "programs")
    art = compile_program(text, "c", sample={"X": X},
                          outputs=["TV", "TI"], basedir=sdir)
    rv, ri = oneshot(art, {"X": X}, work, "c_topk1")
    check("c-topk-oneshot-vals", bool((rv.reshape(R, K) == ref_v).all()),
          "C == lattice values")
    check("c-topk-oneshot-idx", bool((ri.reshape(R, K) == ref_i).all()),
          "C == lattice indices (ties-first)")

    arts = compile_program(text, "c", sample={"X": X},
                           outputs=["TV", "TI"], basedir=sdir, live=["X"])
    exe = build(arts["source"], work, name="c_servetopk",
                libs=arts["libs"])
    argv, files, outs = dump_payload(work, arts, {"X": X}, "c_svtk")
    sv = ServedExe(exe, argv, err_path=os.path.join(work, "c_servetopk.err"))
    try:
        sv.start()
        gv = np.fromfile(outs[0], dtype=np.float64).reshape(R, K)
        gi = np.fromfile(outs[1], dtype=np.int64).reshape(R, K)
        check("c-serve-topk-step0",
              bool((gv == ref_v).all()) and bool((gi == ref_i).all()),
              "served top-k == lattice")
        X2.astype(np.float64).tofile(files["X"])
        sv.step()
        gv2 = np.fromfile(outs[0], dtype=np.float64).reshape(R, K)
        gi2 = np.fromfile(outs[1], dtype=np.int64).reshape(R, K)
        check("c-serve-topk-step1",
              bool((gv2 == ref_v2).all()) and bool((gi2 == ref_i2).all()),
              "served top-k tracks live X")
    except ServeError as e:
        check("c-serve-topk-protocol", False, str(e)[:200])
    finally:
        sv.close()


def test_pack_outputs(work):
    """Packed serve outputs: one contiguous argv file per step."""
    rng = np.random.default_rng(13)
    X = np.ascontiguousarray(rng.normal(size=(4, 8)))
    X2 = np.ascontiguousarray(rng.normal(size=(4, 8)))
    text = "IN X\nY = ARGMAX(X, 1)\n"
    sdir = os.path.join(ROOT, "programs")
    art = compile_program(text, "c", sample={"X": X}, outputs=["Y"],
                          basedir=sdir, live=["X"], pack_outputs=True)
    check("c-pack-flag", art.get("pack_outputs") is True, "packed")
    check("c-pack-argv", art.get("n_argv") == 2,
          f"inputs+one packed output = {art.get('n_argv')}")
    exe = build(art["source"], work, name="c_pack", libs=art["libs"])
    fx = os.path.join(work, "c_pack_x.bin")
    fp = os.path.join(work, "c_pack_out.bin")
    X.tofile(fx)
    sv = ServedExe(exe, [fx, fp],
                   err_path=os.path.join(work, "c_pack.err"))
    try:
        sv.start()
        g0 = np.fromfile(fp, dtype=np.int64)
        check("c-pack-step0", bool((g0 == X.argmax(axis=1)).all()),
              f"{g0.tolist()}")
        X2.tofile(fx)
        sv.step()
        g1 = np.fromfile(fp, dtype=np.int64)
        check("c-pack-step1", bool((g1 == X2.argmax(axis=1)).all()),
              f"{g1.tolist()}")
    except ServeError as e:
        check("c-pack-protocol", False, str(e)[:200])
    finally:
        sv.close()


def test_pack_topk(work):
    """Packed top-k rows share one file per step, values then indices."""
    R, S, K = 5, 12, 3
    rng = np.random.default_rng(17)
    X = np.ascontiguousarray(rng.normal(size=(R, S)) * 2)
    X2 = np.ascontiguousarray(rng.normal(size=(R, S)) * 2)
    fn, _, _ = REGISTRY["TOPK"]
    ref_v, ref_i = fn([X, K], {}, {})
    ref_v2, ref_i2 = fn([X2, K], {}, {})
    text = f"IN X\nTV, TI = TOPK(X, {K})\n"
    sdir = os.path.join(ROOT, "programs")
    art = compile_program(text, "c", sample={"X": X},
                          outputs=["TV", "TI"], basedir=sdir,
                          live=["X"], pack_outputs=True)
    check("c-pack-topk-flag", art.get("pack_outputs") is True, "packed")
    check("c-pack-topk-argv", art.get("n_argv") == 2,
          f"inputs+one packed output = {art.get('n_argv')}")
    exe = build(art["source"], work, name="c_pack_topk",
                libs=art["libs"])
    fx = os.path.join(work, "c_pack_topk_x.bin")
    fp = os.path.join(work, "c_pack_topk_out.bin")
    X.tofile(fx)

    def read_packed():
        raw = open(fp, "rb").read()
        want = R * K * 8 * 2
        check("c-pack-topk-size", len(raw) == want,
              f"{len(raw)} bytes, want {want}")
        vals = np.frombuffer(raw[: R * K * 8], dtype=np.float64).reshape(R, K)
        idx = np.frombuffer(raw[R * K * 8 :], dtype=np.int64).reshape(R, K)
        return vals, idx

    sv = ServedExe(exe, [fx, fp],
                   err_path=os.path.join(work, "c_pack_topk.err"))
    try:
        sv.start()
        gv, gi = read_packed()
        check("c-pack-topk-step0",
              bool((gv == ref_v).all()) and bool((gi == ref_i).all()),
              "packed top-k values+indices == lattice")
        X2.tofile(fx)
        sv.step()
        gv2, gi2 = read_packed()
        check("c-pack-topk-step1",
              bool((gv2 == ref_v2).all()) and bool((gi2 == ref_i2).all()),
              "packed top-k tracks live X")
    except ServeError as e:
        check("c-pack-topk-protocol", False, str(e)[:200])
    finally:
        sv.close()


def test_loud_contracts():
    # graph/sync_each are CUDA execution modes.
    for kw, tag in (({"graph": True}, "graph"),
                    ({"sync_each": False}, "sync_each")):
        try:
            compile_program("IN X\nY = ARGMAX(X, 1)\n", "c",
                            sample={"X": np.zeros((2, 4))},
                            outputs=["Y"], **kw)
            check(f"c-serve-{tag}-loud", False, "accepted")
        except NoPattern as e:
            check(f"c-serve-{tag}-loud", True, str(e)[:80])
    # live name must be a real, non-const, non-output input.
    try:
        compile_program("IN X\nY = ARGMAX(X, 1)\n", "c",
                        sample={"X": np.zeros((2, 4))},
                        outputs=["Y"], live=["nope"])
        check("c-serve-live-name-loud", False, "accepted")
    except NoPattern as e:
        check("c-serve-live-name-loud", True, str(e)[:80])
    # pack_outputs is a resident mode, not a one-shot mode.
    try:
        compile_program("IN X\nY = ARGMAX(X, 1)\n", "c",
                        sample={"X": np.zeros((2, 4))},
                        outputs=["Y"], pack_outputs=True)
        check("c-pack-needs-live-loud", False, "accepted")
    except NoPattern as e:
        check("c-pack-needs-live-loud", True, str(e)[:80])


def main():
    work = "/tmp/emit_c_serve"
    os.makedirs(work, exist_ok=True)
    test_bigram(work)
    test_topk(work)
    test_pack_outputs(work)
    test_pack_topk(work)
    test_loud_contracts()
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
