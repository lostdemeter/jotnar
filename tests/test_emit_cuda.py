"""Agreement gate v0.4: CUDA backend (kernels + cuBLAS, FP32).

Bigram decisions must be bit-exact vs lattice (counts exact in FP32);
compute units gate vs numpy at FP32 eps. Proves the second backend on
the 3090 Ti. Skips cleanly where nvcc is absent.
Usage: python3 tests/test_emit_cuda.py (needs nvcc + GPU).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import compile_program
from chain.emit_cuda import build_cu

FAIL = []
EPS_F32 = 1e-4  # FP32 rounding guard vs float64 numpy


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def run_cu(art, payload_f32, work, name):
    from chain.emit_cuda import build_cu as _b
    exe = _b(art["source"], work, name=name)
    argv = [exe]
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        fn = os.path.join(work, f"in_{n}.bin")
        if k == "F":
            payload_f32[n].astype(np.float32).tofile(fn)
        else:
            payload_f32[n].astype(np.int64).tofile(fn)
        argv.append(fn)
    out_files = []
    for o in art["outputs"]:
        fn = os.path.join(work, f"out_{o}.bin")
        out_files.append(fn)
        argv.append(fn)
    r = subprocess.run(argv, capture_output=True, text=True)
    return r, art, out_files


def main():
    if shutil.which("nvcc") is None:
        print("SKIP (no nvcc)")
        sys.exit(0)
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    rng = np.random.default_rng(7)

    # -- bigram decisions, bit-exact vs lattice ---------------------
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(sdir, "bigram_lm.asm")).read()
    bank_f = np.ascontiguousarray(counts, dtype=np.float64)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks = np.array(sorted(rng.choice(nz, size=200, replace=False).tolist()),
                    np.int64)
    # lattice reference needs triples; CUDA path needs decoded floats.
    bs = np.zeros_like(counts, np.int8)
    be = np.zeros_like(counts, np.int32)
    bz = np.zeros_like(counts, np.uint8)
    for v in np.unique(counts):
        if v == 0:
            continue
        ws, we, wz = S.encode(np.array([float(v)]))
        m = counts == v
        bs[m], be[m], bz[m] = ws[0], we[0], wz[0]
    bz[counts == 0] = 1
    bank_t = (np.ascontiguousarray(bs), np.ascontiguousarray(be),
              np.ascontiguousarray(bz))
    feeds = ASM.run_text(text, REGISTRY, {"tok": toks, "bank": bank_t},
                         sigs=SIGS, basedir=sdir)
    ref = np.ascontiguousarray(feeds["OUT"]).reshape(-1)
    art = compile_program(text, "cuda", sample={"tok": toks, "bank": bank_f},
                          outputs=["OUT"], basedir=sdir)
    work = "/tmp/emit_cuda_bigram"
    os.makedirs(work, exist_ok=True)
    r, art, outs = run_cu(art, {"tok": toks, "bank": bank_f}, work, "bigram")
    check("cu-bigram-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]}")
    if r.returncode == 0:
        got = np.fromfile(outs[0], dtype=np.int64)
        check("cu-bigram-exact", got.shape == ref.shape and bool((got == ref).all()),
              f"{int((got == ref).sum())}/{len(ref)} identical")
    ldd = subprocess.run(["ldd", os.path.join(work, "bigram")],
                         capture_output=True, text=True).stdout
    check("cu-links", "libcublas" in ldd and "phi" not in ldd,
          "cublas-linked, no phi" if ldd else "?")

    # -- compute units vs numpy --------------------------------------
    units = [
        ("mm", "IN a\nIN b\nOUT = MATMUL(a, b)\n",
         {"a": rng.normal(size=(8, 16)), "b": rng.normal(size=(16, 32))},
         lambda p: p["a"] @ p["b"]),
        ("bmm", "IN a\nIN b\nOUT = BATCH_MATMUL(a, b)\n",
         {"a": rng.normal(size=(2, 4, 8)), "b": rng.normal(size=(2, 8, 16))},
         lambda p: p["a"] @ p["b"]),
        ("rms", "CONFIG eps_rms 1e-6\nIN x\nIN w\nOUT = RMSNORM(x, w)\n",
         {"x": rng.normal(size=(4, 32)), "w": rng.normal(size=(32,))},
         lambda p: p["x"] / np.sqrt((p["x"] ** 2).mean(-1, keepdims=True) + 1e-6) * p["w"]),
        ("sm", "IN x\nOUT = SOFTMAX_WIDE(x)\n",
         {"x": rng.normal(size=(4, 64)) * 3},
         lambda p: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(p["x"] - p["x"].max(-1, keepdims=True)))),
        ("elem", "IN a\nIN b\nC = ADD(a, b)\nOUT = MUL(C, a)\n",
         {"a": rng.normal(size=(4, 32)), "b": rng.normal(size=(4, 32))},
         lambda p: (p["a"] + p["b"]) * p["a"]),
    ]
    for tag, prog, sample, ref_fn in units:
        sample = {k: np.ascontiguousarray(v, dtype=np.float64)
                  for k, v in sample.items()}
        art = compile_program(prog, "cuda", sample=sample, outputs=["OUT"])
        work = f"/tmp/emit_cuda_{tag}"
        os.makedirs(work, exist_ok=True)
        r, art, outs = run_cu(art, sample, work, tag)
        if r.returncode != 0:
            check(f"cu-{tag}-run", False, f"rc={r.returncode} {r.stderr[:200]}")
            continue
        ref = np.ascontiguousarray(ref_fn(sample))
        got = np.fromfile(outs[0], dtype=np.float32).reshape(ref.shape)
        dmax = float(np.abs(got - ref).max())
        check(f"cu-{tag}", dmax < EPS_F32, f"maxabs={dmax:.3e}")
    # argmax with a deliberate tie row (first-index rule both sides)
    xa = rng.normal(size=(4, 64))
    xa[2, :] = 1.0
    xa[2, 0] = 1.0
    art = compile_program("IN x\nOUT = ARGMAX(x, 1)\n", "cuda",
                          sample={"x": np.ascontiguousarray(xa)},
                          outputs=["OUT"])
    work = "/tmp/emit_cuda_am"
    os.makedirs(work, exist_ok=True)
    r, art, outs = run_cu(art, {"x": xa}, work, "am")
    if r.returncode != 0:
        check("cu-argmax-run", False, f"rc={r.returncode} {r.stderr[:200]}")
    else:
        got = np.fromfile(outs[0], dtype=np.int64)
        ref = np.argmax(xa, axis=1)
        check("cu-argmax", bool((got == ref).all()),
              f"{int((got == ref).sum())}/{len(ref)} (tie row -> {got[2]})")

    headt_case()

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


def headt_case():
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    for f in ("lm_svd_IvoQ.npz", "bankhn.npz"):
        if not os.path.isfile(os.path.join(dd, f)):
            print(f"SKIP headt-cu (missing {f})")
            return
    CFG = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
           "CONFIG beta_b 0.25\n")
    text = CFG + open(os.path.join(sdir, "lm_headt.asm")).read()
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ids = [12, 471, 59]
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    trip = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
            "wv": enc(d["wv"]), "wo": enc(d["wo"]), "wup": enc(d["wup"]),
            "wgate": enc(d["wgate"]), "wdown": enc(d["wdown"]),
            "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
            "wlog": enc(d["wlog"]), "ukt": enc(b["ukt"]),
            "evb": enc(b["evb"])}
    payload = {"tok": toks, "pos": pos, "cmask": cm}
    payload.update(trip)
    feeds = ASM.run_text(text, REGISTRY, payload, sigs=SIGS, basedir=sdir)
    ref_logits = dec(feeds["LOGITS"])
    ref_out = np.ascontiguousarray(feeds["OUT"]).reshape(-1)
    fpayload = {"tok": toks, "pos": pos, "cmask": cm}
    fpayload.update({k: np.ascontiguousarray(dec(v))
                     for k, v in trip.items()})
    art = compile_program(text, "cuda", sample=fpayload,
                          outputs=["LOGITS", "OUT"], basedir=sdir)
    work = "/tmp/emit_cuda_headt"
    os.makedirs(work, exist_ok=True)
    r, art, outs = run_cu(art, fpayload, work, "headt")
    check("cu-headt-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:300]}")
    if r.returncode != 0:
        return
    got_logits = np.fromfile(outs[0], dtype=np.float32
                             ).reshape(ref_logits.shape)
    got_out = np.fromfile(outs[1], dtype=np.int64)
    dmax = float(np.abs(got_logits - ref_logits).max())
    print(f"cu-headt-logits: maxabs={dmax:.3e} (cal 6e-02, same as C)")
    check("cu-headt-logits-eps", dmax < 6e-2, f"maxabs={dmax:.3e}")
    check("cu-headt-out-exact", bool((got_out == ref_out).all()),
          f"{int((got_out == ref_out).sum())}/{len(ref_out)}")


if __name__ == "__main__":
    main()
