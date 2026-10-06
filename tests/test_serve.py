"""Persistence gate: loop binary serves steps without reloading.

Bigram program compiled with live=["tok"] (bank frozen): served steps
must match one-shot binaries on identical inputs, repeat steps must be
byte-identical, and -- the money gate -- overwriting the bank file
mid-run must NOT change outputs (frozen inputs are read once, which is
the whole reload gap in docs/PERF.md).
SKIPs without nvcc/GPU. Usage: python3 tests/test_serve.py
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain.emit_c import compile_program
from chain.emit_cuda import build_cu
from chain.serve import ServedExe, ServeError

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def dump_payload(work, art, payload):
    argv, files = [], {}
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"sv_{n}.bin")
        if k == "F":
            payload[n].astype(np.float32).tofile(fn)
        else:
            payload[n].astype(np.int64).tofile(fn)
        argv.append(fn)
        files[n] = fn
    outs = []
    for o in art["outputs"]:
        fn = os.path.join(work, f"svout_{o}.bin")
        outs.append(fn)
        argv.append(fn)
    return argv, files, outs


def oneshot(art, payload, work, name, dtype=np.int64):
    from chain.emit_cuda import build_cu as _b
    exe = _b(art["source"], work, name=name)
    argv, _, outs = dump_payload(work, art, payload)
    r = subprocess.run([exe] + argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:300]
    return np.fromfile(outs[0], dtype=dtype)


def main():
    if shutil.which("nvcc") is None:
        print("SKIP (no nvcc)")
        sys.exit(0)
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    rng = np.random.default_rng(7)
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(sdir, "bigram_lm.asm")).read()
    bank_f = np.ascontiguousarray(counts, dtype=np.float64)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks1 = np.array(sorted(rng.choice(nz, size=50, replace=False).tolist()),
                     np.int64)
    toks2 = np.array(sorted(rng.choice(nz, size=50, replace=False).tolist()),
                     np.int64)
    work = "/tmp/emit_cuda_serve"
    os.makedirs(work, exist_ok=True)

    art1 = compile_program(text, "cuda", sample={"tok": toks1, "bank": bank_f},
                           outputs=["OUT"], basedir=sdir)
    ref1 = oneshot(art1, {"tok": toks1, "bank": bank_f}, work, "one1")
    ref2 = oneshot(art1, {"tok": toks2, "bank": bank_f}, work, "one2")

    art = compile_program(text, "cuda", sample={"tok": toks1, "bank": bank_f},
                          outputs=["OUT"], basedir=sdir, live=["tok"])
    check("serve-live-flag", art.get("live") == ["tok"], f"{art.get('live')}")
    exe = build_cu(art["source"], work, name="serve")
    argv, files, outs = dump_payload(work, art, {"tok": toks1, "bank": bank_f})

    sv = ServedExe(exe, argv, err_path=os.path.join(work, "serve.err"))
    try:
        t0 = time.perf_counter()
        sv.start()
        t_load = time.perf_counter() - t0
        print(f"serve-load-first-step: {t_load:.1f}s (one-shot did this per step)",
              flush=True)
        got1 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-step0-match", bool((got1 == ref1).all()),
              f"{int((got1 == ref1).sum())}/{len(ref1)} vs one-shot")

        toks2.astype(np.int64).tofile(files["tok"])
        t0 = time.perf_counter()
        sv.step()
        t_step = time.perf_counter() - t0
        got2 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-step1-match", bool((got2 == ref2).all()),
              f"{t_step:.2f}s/step ({int((got2 == ref2).sum())}/{len(ref2)})")

        # Frozen-proof: poison the bank file; outputs must NOT move.
        rng.standard_normal(bank_f.shape).astype(np.float32).tofile(files["bank"])
        toks1.astype(np.int64).tofile(files["tok"])
        sv.step()
        got3 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-frozen-proof", bool((got3 == ref1).all()),
              "bank poisoned mid-run, outputs unmoved (read-once proven)")

        # Determinism: same live files twice -> identical bytes.
        sv.step()
        got4 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-deterministic", bool((got4 == got3).all()), "replay-identical")
    except ServeError as e:
        check("serve-protocol", False, str(e)[:200])
    finally:
        sv.close()

    # -- matmul overwrite gate (beta=0): loop steps must not accumulate.
    # A beta=1 emitter passes one-shot (fresh zero-pages) and fails here.
    from chain.builder import Prog
    p = Prog("serve-mm")
    p.inp("x", "w")
    y = p.op("MATMUL", "x", "w", out="Y")
    rng2 = np.random.default_rng(3)
    mx = np.ascontiguousarray(rng2.normal(size=(4, 8)))
    mw = np.ascontiguousarray(rng2.normal(size=(8, 6)))
    artm = compile_program(p.text(), "cuda", sample={"x": mx, "w": mw},
                           outputs=[y], basedir=sdir)
    refm = oneshot(artm, {"x": mx, "w": mw}, work, "onemm", np.float32)
    artml = compile_program(p.text(), "cuda", sample={"x": mx, "w": mw},
                            outputs=[y], basedir=sdir, live=["x"])
    exem = build_cu(artml["source"], work, name="servemm")
    argv, files, outs = dump_payload(work, artml, {"x": mx, "w": mw})
    svm = ServedExe(exem, argv, err_path=os.path.join(work, "servemm.err"))
    try:
        svm.start()
        m0 = np.fromfile(outs[0], dtype=np.float32)
        check("serve-mm-step0", np.allclose(m0, refm, atol=1e-4),
              "loop step0 == one-shot")
        svm.step()  # same live files: must NOT accumulate (beta=0)
        m1 = np.fromfile(outs[0], dtype=np.float32)
        check("serve-mm-no-accum", np.array_equal(m0, m1),
              f"maxdiff {np.abs(m0 - m1).max():.2e} (beta=1 drifts here)")
        mx2 = np.ascontiguousarray(rng2.normal(size=(4, 8)))
        mx2.astype(np.float32).tofile(files["x"])
        svm.step()
        m2 = np.fromfile(outs[0], dtype=np.float32)
        refm2 = oneshot(artm, {"x": mx2, "w": mw}, work, "onemm2", np.float32)
        check("serve-mm-follows", np.allclose(m2, refm2, atol=1e-4),
              "loop tracks new live inputs")
        # sync_each=False must be bitwise-identical (same kernels, same
        # stream order; only host-side waits removed).
        artq = compile_program(p.text(), "cuda", sample={"x": mx, "w": mw},
                               outputs=[y], basedir=sdir, sync_each=False)
        refq = oneshot(artq, {"x": mx, "w": mw}, work, "onenosync", np.float32)
        check("serve-nosync-equiv", np.array_equal(refm, refq),
              "stream-sync == per-op-sync, bit-exact")
    except ServeError as e:
        check("serve-mm-protocol", False, str(e)[:200])
    finally:
        svm.close()
    # -- graph replay gate: one launch per step, same bytes ------------
    artg = compile_program(text, "cuda", sample={"tok": toks1, "bank": bank_f},
                           outputs=["OUT"], basedir=sdir,
                           live=["tok"], graph=True)
    check("serve-graph-flag", artg.get("graph") is True, "graph recorded")
    exeg = build_cu(artg["source"], work, name="serveg")
    argv, files, outs = dump_payload(work, artg, {"tok": toks1, "bank": bank_f})
    svg = ServedExe(exeg, argv, err_path=os.path.join(work, "serveg.err"))
    try:
        svg.start()
        g0 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-graph-step0", bool((g0 == ref1).all()),
              "captured step == one-shot")
        toks2.astype(np.int64).tofile(files["tok"])
        svg.step()
        g1 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-graph-step1", bool((g1 == ref2).all()),
              "replay tracks new live inputs")
        svg.step()
        g2 = np.fromfile(outs[0], dtype=np.int64)
        check("serve-graph-deterministic", bool((g2 == g1).all()),
              "replay-identical")
    except ServeError as e:
        check("serve-graph-protocol", False, str(e)[:200])
    finally:
        svg.close()
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
