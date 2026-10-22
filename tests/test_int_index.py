"""Integer/index algebra gate: IADD/ISUB/IMUL/IMAD/IDIV/IMOD + GATHER3,
plus the explicit F2I/I2F boundaries.

Proves the n-gram path end to end:
  * I-kind arithmetic is exact and wraps as specified (C + non-FPU);
  * IDIV/IMOD truncate toward zero and refuse zero / INT64_MIN % -1 loud;
  * GATHER3 reads (V1,V2,C) rows by two ids and feeds ARGMAX;
  * F2I truncates and range-checks; I2F rounds to nearest-even; the
    non-FPU target implements both by decoding/constructing IEEE-754
    bits with integer operations only;
  * C, non-FPU, and CUDA agree where the backend admits the kinds
    (CUDA F is F32; CUDA F tables only, same data-prep discipline as
    GATHER);
  * non-FPU source stays float-free and links without libm;
  * bad table/ids kinds, bounds, divisors, and conversion ranges fail loud.
Usage: python3 tests/test_int_index.py (needs cc; CUDA build optional).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR",
                                  os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S  # noqa: E402
from chain import asm as ASM  # noqa: E402
from chain.asm_ops import REGISTRY, SIGS  # noqa: E402
from chain.emit_c import NoPattern, build, compile_program  # noqa: E402
from chain.emit_nonfpu import float_trap  # noqa: E402
from chain.emit_cuda import build_cu  # noqa: E402

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def _write_inputs(work, art, payload, f32=False, f16=False):
    """Write argv files in declaration order; T streams expand to s/e/z."""
    files = []
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        if k == "T":
            for comp, arr in zip("sez", payload[n]):
                fn = os.path.join(work, f"in_{n}_{comp}.bin")
                np.ascontiguousarray(arr).tofile(fn)
                files.append(fn)
        elif k == "F":
            fn = os.path.join(work, f"in_{n}.bin")
            dtype = np.float16 if f16 else (
                np.float32 if f32 else np.float64)
            np.ascontiguousarray(payload[n], dtype=dtype).tofile(fn)
            files.append(fn)
        else:
            fn = os.path.join(work, f"in_{n}.bin")
            np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
            files.append(fn)
    return files


def _read_outputs(work, art, f32=None):
    if f32 is None:
        f32 = getattr(art.get("backend"), "name", "") == "cuda"
    out = {}
    for o in art["outputs"]:
        k, _, sh = art["streams"][o]
        if k == "T":
            planes = []
            for comp, dt in (("s", np.int8), ("e", np.int32),
                             ("z", np.uint8)):
                fn = os.path.join(work, f"out_{o}_{comp}.bin")
                planes.append(np.fromfile(fn, dtype=dt).reshape(sh))
            out[o] = tuple(planes)
        elif k == "F":
            fn = os.path.join(work, f"out_{o}.bin")
            out[o] = np.fromfile(fn, dtype=np.float32 if f32
                                else np.float64).reshape(sh)
        else:
            fn = os.path.join(work, f"out_{o}.bin")
            out[o] = np.fromfile(fn, dtype=np.int64).reshape(sh)
    return out


def _run_art(work, art, payload, name, target):
    if target == "nonfpu":
        float_trap(art["source"])
        exe = build(art["source"], work, name=name, libs=[])
    else:
        exe = build(art["source"], work, name=name)
    files = _write_inputs(work, art, payload, f32=(target == "cuda"))
    outs = []
    for o in art["outputs"]:
        k, _, _ = art["streams"][o]
        if k == "T":
            outs.extend(os.path.join(work, f"out_{o}_{c}.bin") for c in "sez")
        else:
            outs.append(os.path.join(work, f"out_{o}.bin"))
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name}: rc={r.returncode} {r.stderr[:400]}")
    return _read_outputs(work, art, f32=(target == "cuda"))


def _run_rc(work, art, payload, name, target):
    """Build/run expecting a nonzero exit; returns the process return code."""
    if target == "nonfpu":
        float_trap(art["source"])
        exe = build(art["source"], work, name=name, libs=[])
    else:
        exe = build(art["source"], work, name=name)
    files = _write_inputs(work, art, payload, f32=(target == "cuda"))
    outs = []
    for o in art["outputs"]:
        k, _, _ = art["streams"][o]
        if k == "T":
            outs.extend(os.path.join(work, f"out_{o}_{c}.bin") for c in "sez")
        else:
            outs.append(os.path.join(work, f"out_{o}.bin"))
    r = subprocess.run([exe] + files + outs, capture_output=True, text=True)
    return r.returncode


def _same_triples(a, b):
    return all(bool((np.asarray(x) == np.asarray(y)).all())
               for x, y in zip(a, b))


def _enc(a):
    return tuple(np.ascontiguousarray(p) for p in S.encode(np.asarray(a)))


def main():
    work = "/tmp/jotnar_int_index"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(41)

    # -- 1. I arithmetic, exact and wrap-defined ------------------------
    arith = ("IN a\nIN b\n"
             "ADD = IADD(a, b)\n"
             "SUB = ISUB(a, b)\n"
             "MUL = IMUL(ADD, 3)\n"
             "MAD = IMAD(a, b, SUB)\n"
             "MADS = IMAD(a, 5, b)\n"
             "MADC = IMAD(a, b, 7)\n")
    a = np.array([2**63 - 1, -7, 0, 13, -5, 100], dtype=np.int64)
    b = np.array([1, 3, 9, -2, 4, -250], dtype=np.int64)
    pay = {"a": a, "b": b}
    ref = ASM.run_text(arith, REGISTRY, pay, sigs=SIGS)
    names = ["ADD", "SUB", "MUL", "MAD", "MADS", "MADC"]
    for target in ("c", "nonfpu"):
        art = compile_program(arith, target, sample=pay, outputs=names)
        got = _run_art(work, art, pay, f"arith_{target}", target)
        ok = all(bool((got[n] == ref[n]).all()) for n in names)
        check(f"int-arith-{target}-exact", ok,
              f"{len(names)} outputs, wrap case a[0]+1={int(got['ADD'][0])}")
    # non-FPU source has no float/libm even with integer arithmetic
    art = compile_program(arith, "nonfpu", sample=pay, outputs=names)
    try:
        float_trap(art["source"])
        check("int-arith-nonfpu-trap-clean", True, "int64/uint64 only")
    except Exception as e:  # noqa: BLE001
        check("int-arith-nonfpu-trap-clean", False, str(e)[:140])

    # -- 2. native trigram: GATHER3 + ARGMAX + IMAD, T path ------------
    V, C, N = 6, 4, 17
    tri = rng.normal(size=(V, V, C))
    bank_t = _enc(tri)
    prev = rng.integers(0, V, size=N, dtype=np.int64)
    prev2 = rng.integers(0, V, size=N, dtype=np.int64)
    tri_prog = ("IN prev\nIN prev2\nIN bank\n"
                "CTX = IMAD(prev, 6, prev2)\n"
                "ROW = GATHER3(bank, prev, prev2)\n"
                "IDX = ARGMAX(ROW, 1)\n"
                "OUT = IADD(CTX, IDX)\n")
    pay = {"prev": prev, "prev2": prev2, "bank": bank_t}
    ref = ASM.run_text(tri_prog, REGISTRY, pay, sigs=SIGS)
    for target in ("c", "nonfpu"):
        art = compile_program(tri_prog, target, sample=pay,
                              outputs=["ROW", "OUT"])
        got = _run_art(work, art, pay, f"trigram_{target}", target)
        check(f"gather3-{target}-rows-exact",
              _same_triples(got["ROW"], ref["ROW"]), "3 planes bit-identical")
        check(f"trigram-{target}-decisions-exact",
              bool((got["OUT"] == ref["OUT"]).all()),
              f"{int((got['OUT'] == ref['OUT']).sum())}/{N}")
    art = compile_program(tri_prog, "nonfpu", sample=pay,
                          outputs=["ROW", "OUT"])
    try:
        float_trap(art["source"])
        check("gather3-nonfpu-trap-clean", True, "pure integer moves/arithmetic")
    except Exception as e:  # noqa: BLE001
        check("gather3-nonfpu-trap-clean", False, str(e)[:140])

    # both gather forms in one program: row-view prefixes must not collide
    bank2_t = _enc(tri.reshape(V * V, C))
    both = ("IN prev\nIN prev2\nIN bank2\nIN bank3\n"
            "CTX = IMAD(prev, 6, prev2)\n"
            "R1 = GATHER(bank2, CTX)\n"
            "R2 = GATHER3(bank3, prev, prev2)\n"
            "I1 = ARGMAX(R1, 1)\n"
            "I2 = ARGMAX(R2, 1)\n"
            "OUT = IADD(I1, I2)\n")
    payb = {"prev": prev, "prev2": prev2, "bank2": bank2_t,
            "bank3": bank_t}
    refb = ASM.run_text(both, REGISTRY, payb, sigs=SIGS)
    for target in ("c", "nonfpu"):
        artb = compile_program(both, target, sample=payb, outputs=["OUT"])
        gotb = _run_art(work, artb, payb, f"both_{target}", target)
        check(f"two-gathers-{target}-exact",
              bool((gotb["OUT"] == refb["OUT"]).all()),
              f"{int((gotb['OUT'] == refb['OUT']).sum())}/{N}")

    # -- 3. F-table GATHER3 (C + CUDA data-prep path) ------------------
    bank_f64 = np.ascontiguousarray(tri, dtype=np.float64)
    f_prog = ("IN prev\nIN prev2\nIN bank\n"
              "ROW = GATHER3(bank, prev, prev2)\n"
              "IDX = ARGMAX(ROW, 1)\n"
              "OUT = IADD(prev, IDX)\n")
    payf = {"prev": prev, "prev2": prev2, "bank": bank_f64}
    ref = ASM.run_text(f_prog, REGISTRY, payf, sigs=SIGS)
    art = compile_program(f_prog, "c", sample=payf, outputs=["OUT"])
    got = _run_art(work, art, payf, "gather3_f_c", "c")
    check("gather3-c-f-table-exact",
          bool((got["OUT"] == ref["OUT"]).all()),
          f"{int((got['OUT'] == ref['OUT']).sum())}/{N}")

    # CUDA source must always be emittable; execution is gated when the
    # local toolkit/GPU can build (same policy as test_emit_cuda.py).
    try:
        artcu = compile_program(f_prog, "cuda", sample=payf, outputs=["OUT"])
        check("gather3-cuda-emits",
              all(s in artcu["source"] for s in
                  ("k_gather3_f32", "k_argmax_f32", "k_int_bin")),
              f"{len(artcu['source'])} bytes")
    except Exception as e:  # noqa: BLE001
        check("gather3-cuda-emits", False, str(e)[:160])
        artcu = None
    if artcu is not None and shutil.which("nvcc"):
        try:
            exe = build_cu(artcu["source"], work, name="trigram_cu",
                           libs=["-lcublas"])
            files = _write_inputs(work, artcu, payf, f32=True)
            fout = os.path.join(work, "out_cu.bin")
            r = subprocess.run([exe] + files + [fout],
                               capture_output=True, text=True)
            if r.returncode != 0:
                check("gather3-cuda-run", False,
                      f"rc={r.returncode} {r.stderr[:240]}")
            else:
                gotcu = np.fromfile(fout, dtype=np.int64).reshape(N)
                check("gather3-cuda-exact", bool((gotcu == ref["OUT"]).all()),
                      f"{int((gotcu == ref['OUT']).sum())}/{N}")
        except Exception as e:  # noqa: BLE001
            print(f"gather3-cuda-run: SKIP (build unavailable: "
                  f"{str(e)[:160]})")
    else:
        print("gather3-cuda-run: SKIP (no nvcc)")

    # -- 4. IDIV/IMOD: exact C-style truncation ------------------------
    div_prog = ("IN a\nIN b\n"
                "D = IDIV(a, b)\n"
                "M = IMOD(a, b)\n"
                "DS = IDIV(a, 3)\n"
                "MS = IMOD(a, -3)\n")
    da = np.array([7, -7, 7, -7, 0, 100, -100,
                   2**63 - 1, -2**63 + 1, -2**63], dtype=np.int64)
    db = np.array([3, 3, -3, -3, -7, 5, -5, 2, -2, 3], dtype=np.int64)
    payd = {"a": da, "b": db}
    refd = ASM.run_text(div_prog, REGISTRY, payd, sigs=SIGS)
    for target in ("c", "nonfpu"):
        artd = compile_program(div_prog, target, sample=payd,
                               outputs=["D", "M", "DS", "MS"])
        gotd = _run_art(work, artd, payd, f"div_{target}", target)
        ok = all(bool((gotd[n] == refd[n]).all())
                 for n in ("D", "M", "DS", "MS"))
        check(f"idiv-imod-{target}-exact", ok,
              f"min/3 q={int(gotd['D'][-1])} r={int(gotd['M'][-1])}")

    # random truncation fuzz (avoid the one refused overflow pair)
    rngd = np.random.default_rng(99)
    fa = rngd.integers(-(2**63), 2**63 - 1, size=4096, dtype=np.int64)
    fb = rngd.integers(-1000, 1001, size=4096, dtype=np.int64)
    fb[fb == 0] = 7
    fa[(fa == -(2**63)) & (fb == -1)] = 0
    payfz = {"a": fa, "b": fb}
    fzprog = "IN a\nIN b\nD = IDIV(a, b)\nM = IMOD(a, b)\n"
    reffz = ASM.run_text(fzprog, REGISTRY, payfz, sigs=SIGS)
    for target in ("c", "nonfpu"):
        artfz = compile_program(fzprog, target, sample=payfz,
                                outputs=["D", "M"])
        gotfz = _run_art(work, artfz, payfz, f"divfz_{target}", target)
        check(f"idiv-imod-{target}-fuzz",
              bool((gotfz["D"] == reffz["D"]).all())
              and bool((gotfz["M"] == reffz["M"]).all()),
              "4096 random pairs")

    # -- 5. F2I/I2F: explicit boundary, bit-exact C/non-FPU ------------
    conv_prog = ("IN xf\nIN n\n"
                 "I = F2I(xf)\n"
                 "F = I2F(n)\n")
    xf = np.array([3.9, -2.7, 0.0, -0.0, 42.0, -1.5,
                   1e15, -1e15, float(2**53), -float(2**53),
                   float(2**63 - 1024), -float(2**63)],
                  dtype=np.float64)
    n = np.array([0, 1, -1, 123456789, -123456789,
                  2**53, 2**53 + 1, 2**53 + 2, 2**53 + 3,
                  -(2**53), -(2**53) - 1, -(2**53) - 2, -(2**53) - 3,
                  2**62, -2**62, 2**63 - 1, 2**63 - 1024,
                  -(2**63)], dtype=np.int64)
    payc = {"xf": xf, "n": n}
    refc = ASM.run_text(conv_prog, REGISTRY, payc, sigs=SIGS)
    for target in ("c", "nonfpu"):
        artc = compile_program(conv_prog, target, sample=payc,
                               outputs=["I", "F"])
        gotc = _run_art(work, artc, payc, f"conv_{target}", target)
        check(f"f2i-{target}-exact",
              bool((gotc["I"] == refc["I"]).all()),
              f"{len(xf)} inputs incl. -2^63")
        check(f"i2f-{target}-round-even-bits",
              bool((gotc["F"].view(np.uint64)
                    == refc["F"].view(np.uint64)).all()),
              f"{len(n)} inputs incl. 2^53 ties, INT64_MIN/MAX")
    artnf = compile_program(conv_prog, "nonfpu", sample=payc,
                            outputs=["I", "F"])
    try:
        float_trap(artnf["source"])
        check("f2i-i2f-nonfpu-trap-clean", True,
              "IEEE bits decoded/constructed with int64/uint64 only")
    except Exception as e:  # noqa: BLE001
        check("f2i-i2f-nonfpu-trap-clean", False, str(e)[:140])

    # F is a real intermediate here (non-FPU stores it as raw uint64_t).
    rt_prog = "IN n\nF = I2F(n)\nI = F2I(F)\n"
    nrt = np.array([0, 1, -1, 2**53, -2**53, 123456789, -123456789],
                   dtype=np.int64)
    payrt = {"n": nrt}
    refrt = ASM.run_text(rt_prog, REGISTRY, payrt, sigs=SIGS)
    for target in ("c", "nonfpu"):
        artrt = compile_program(rt_prog, target, sample=payrt,
                                outputs=["F", "I"])
        gotrt = _run_art(work, artrt, payrt, f"roundtrip_{target}", target)
        check(f"i2f-f2i-{target}-roundtrip",
              bool((gotrt["I"] == refrt["I"]).all())
              and bool((gotrt["F"].view(np.uint64)
                        == refrt["F"].view(np.uint64)).all()),
              "F intermediate, bit-exact")

    # CUDA is F32: use values exact in F32 so the same listing is a gate.
    xf32 = np.array([3.9, -2.7, 0.0, -0.0, 42.0, -1.5],
                    dtype=np.float32)
    n32 = np.array([0, 1, -1, 2**20, -2**20, 123456,
                    2**24, -2**24], dtype=np.int64)
    paycu = {"xf": xf32, "n": n32}
    refcu = ASM.run_text(conv_prog, REGISTRY, paycu, sigs=SIGS)
    all_prog = ("IN a\nIN b\nIN xf\nIN n\n"
                "D = IDIV(a, b)\n"
                "M = IMOD(a, b)\n"
                "I = F2I(xf)\n"
                "F = I2F(n)\n")
    payall = {"a": da, "b": db, "xf": xf32, "n": n32}
    refall = ASM.run_text(all_prog, REGISTRY, payall, sigs=SIGS)
    try:
        artall = compile_program(all_prog, "cuda", sample=payall,
                                 outputs=["D", "M", "I", "F"])
        check("div-conv-cuda-emits",
              all(s in artall["source"] for s in
                  ("k_int_div", "k_f2i_f32", "k_i2f_f32", "ierr")),
              f"{len(artall['source'])} bytes")
    except Exception as e:  # noqa: BLE001
        check("div-conv-cuda-emits", False, str(e)[:160])
        artall = None
    artfp = None
    try:
        artfp = compile_program(conv_prog, "cuda", sample=paycu,
                                outputs=["I", "F"], use_fp16=True)
        check("f2i-cuda-fp16-emits",
              "k_f2i_f16" in artfp["source"]
              and "__half *xf" in artfp["source"],
              "half input upcast before the checked boundary")
    except Exception as e:  # noqa: BLE001
        check("f2i-cuda-fp16-emits", False, str(e)[:160])
    if artfp is not None and shutil.which("nvcc"):
        try:
            exe = build_cu(artfp["source"], work, name="conv_cu_fp16",
                           libs=["-lcublas"])
            xf16 = np.asarray(xf32, dtype=np.float16)
            payfp = {"xf": xf16, "n": n32}
            reffp = ASM.run_text(conv_prog, REGISTRY,
                                 {"xf": xf16.astype(np.float32), "n": n32},
                                 sigs=SIGS)
            files = _write_inputs(work, artfp, payfp, f16=True)
            outs = [os.path.join(work, f"out_{o}.bin") for o in ("I", "F")]
            r = subprocess.run([exe] + files + outs,
                               capture_output=True, text=True)
            if r.returncode != 0:
                check("f2i-cuda-fp16-run", False,
                      f"rc={r.returncode} {r.stderr[:240]}")
            else:
                gotI = np.fromfile(outs[0], dtype=np.int64).reshape(xf32.shape)
                gotF = np.fromfile(outs[1], dtype=np.float32).reshape(n32.shape)
                check("f2i-cuda-fp16-run",
                      bool((gotI == reffp["I"]).all())
                      and bool((gotF == np.asarray(
                          reffp["F"], dtype=np.float32)).all()),
                      "half input -> checked I; I2F output F32")
        except Exception as e:  # noqa: BLE001
            print(f"f2i-cuda-fp16-run: SKIP (build unavailable: "
                  f"{str(e)[:160]})")
    else:
        print("f2i-cuda-fp16-run: SKIP (no nvcc)")
    if artall is not None and shutil.which("nvcc"):
        try:
            exe = build_cu(artall["source"], work, name="conv_cu",
                           libs=["-lcublas"])
            files = _write_inputs(work, artall, payall, f32=True)
            outs = [os.path.join(work, f"out_{o}.bin")
                    for o in ("D", "M", "I", "F")]
            r = subprocess.run([exe] + files + outs,
                               capture_output=True, text=True)
            if r.returncode != 0:
                check("div-conv-cuda-run", False,
                      f"rc={r.returncode} {r.stderr[:240]}")
            else:
                gotD = np.fromfile(outs[0], dtype=np.int64).reshape(da.shape)
                gotM = np.fromfile(outs[1], dtype=np.int64).reshape(da.shape)
                gotI = np.fromfile(outs[2], dtype=np.int64).reshape(xf32.shape)
                gotF = np.fromfile(outs[3], dtype=np.float32).reshape(n32.shape)
                check("div-conv-cuda-exact",
                      bool((gotD == refall["D"]).all())
                      and bool((gotM == refall["M"]).all())
                      and bool((gotI == refall["I"]).all())
                      and bool((gotF == np.asarray(
                          refall["F"], dtype=np.float32)).all()),
                      "IDIV/IMOD/F2I exact; I2F F32")
        except Exception as e:  # noqa: BLE001
            print(f"div-conv-cuda-run: SKIP (build unavailable: "
                  f"{str(e)[:160]})")
    else:
        print("div-conv-cuda-run: SKIP (no nvcc)")

    # -- 6. checked failures (reference + C + non-FPU) ------------------
    errs = [
        ("idiv-zero", "IN a\nIN b\nD = IDIV(a, b)\n",
         {"a": np.array([1], dtype=np.int64),
          "b": np.array([0], dtype=np.int64)}, "D"),
        ("idiv-overflow", "IN a\nIN b\nD = IDIV(a, b)\n",
         {"a": np.array([-(2**63)], dtype=np.int64),
          "b": np.array([-1], dtype=np.int64)}, "D"),
        ("imod-overflow", "IN a\nIN b\nM = IMOD(a, b)\n",
         {"a": np.array([-(2**63)], dtype=np.int64),
          "b": np.array([-1], dtype=np.int64)}, "M"),
        ("f2i-nan", "IN xf\nI = F2I(xf)\n",
         {"xf": np.array([np.nan], dtype=np.float64)}, "I"),
        ("f2i-inf", "IN xf\nI = F2I(xf)\n",
         {"xf": np.array([np.inf], dtype=np.float64)}, "I"),
        ("f2i-range", "IN xf\nI = F2I(xf)\n",
         {"xf": np.array([float(2**63)], dtype=np.float64)}, "I"),
    ]
    for tag, prog, epay, out in errs:
        try:
            ASM.run_text(prog, REGISTRY, epay, sigs=SIGS)
            check(f"checked-ref-{tag}-loud", False, "reference accepted it")
        except Exception:  # noqa: BLE001
            check(f"checked-ref-{tag}-loud", True, "reference refused")
        for target in ("c", "nonfpu"):
            art = compile_program(prog, target, sample=epay, outputs=[out])
            rc = _run_rc(work, art, epay, f"{tag}_{target}", target)
            check(f"checked-{tag}-{target}-rc30", rc == 30, f"rc={rc}")

    # compile-time literal zero is refused before codegen
    for target in ("c", "nonfpu", "cuda"):
        try:
            compile_program("IN a\nD = IDIV(a, 0)\n", target,
                            sample={"a": np.array([1], dtype=np.int64)},
                            outputs=["D"])
            check(f"loud-literal-zero-{target}", False, "compiled")
        except (ASM.AsmError, NoPattern) as e:
            check(f"loud-literal-zero-{target}", True, str(e)[:100])

    # graph capture cannot hide the host-side error check
    for tag, prog, gpay, live in (
            ("idiv", "IN a\nIN b\nD = IDIV(a, b)\n", payd, ["a"]),
            ("f2i", "IN xf\nI = F2I(xf)\n", payc, ["xf"])):
        try:
            compile_program(prog, "cuda", sample=gpay, outputs=["D"]
                            if tag == "idiv" else ["I"],
                            live=live, graph=True)
            check(f"loud-graph-{tag}", False, "graph compiled")
        except NoPattern as e:
            check(f"loud-graph-{tag}", True, str(e)[:100])

    # -- 7. loud refusals ----------------------------------------------
    bad = {
        "2d-table": ("IN i\nIN j\nIN t\nOUT = GATHER3(t, i, j)\n",
                     {"i": prev, "j": prev, "t": np.zeros((V, C))}),
        "id-length": ("IN i\nIN j\nIN t\nOUT = GATHER3(t, i, j)\n",
                      {"i": prev, "j": prev2[:-1],
                       "t": np.zeros((V, V, C))}),
        "float-iadd": ("IN a\nIN b\nOUT = IADD(a, b)\n",
                       {"a": np.ones(4), "b": np.ones(4)}),
        "float-idiv": ("IN a\nIN b\nOUT = IDIV(a, b)\n",
                       {"a": np.ones(4), "b": np.ones(4)}),
        "f2i-int": ("IN a\nOUT = F2I(a)\n",
                    {"a": np.ones(4, dtype=np.int64)}),
        "i2f-float": ("IN a\nOUT = I2F(a)\n",
                      {"a": np.ones(4, dtype=np.float64)}),
    }
    for tag, (prog, sample) in bad.items():
        try:
            compile_program(prog, "c", sample=sample, outputs=["OUT"])
            check(f"loud-{tag}", False, "compiled")
        except (ASM.AsmError, NoPattern) as e:
            check(f"loud-{tag}", True, str(e)[:100])
    try:
        compile_program(f_prog, "nonfpu", sample=payf, outputs=["OUT"])
        check("loud-nonfpu-f-table", False, "compiled")
    except NoPattern as e:
        check("loud-nonfpu-f-table", True, str(e)[:100])
    try:
        compile_program(tri_prog, "cuda", sample=pay, outputs=["OUT"])
        check("loud-cuda-t-table", False, "compiled")
    except NoPattern as e:
        check("loud-cuda-t-table", True, str(e)[:100])

    # runtime bounds: C pattern returns 4 before reading a bad row
    oob = prev.copy()
    oob[0] = V
    pay_oob = {"prev": oob, "prev2": prev2, "bank": bank_t}
    art = compile_program(tri_prog, "c", sample=pay_oob,
                          outputs=["OUT"])
    exe = build(art["source"], work, name="oob")
    files = _write_inputs(work, art, pay_oob)
    fout = os.path.join(work, "out_oob.bin")
    r = subprocess.run([exe] + files + [fout], capture_output=True, text=True)
    check("gather3-bounds-runtime-loud", r.returncode == 4,
          f"rc={r.returncode}")

    # -- 8. shape-only T bank (no multi-GB allocation at compile) -------
    from chain.emit_c import IRef, TRef
    big = TRef((2048, 2048, 8))
    ids = IRef((512,))
    shape_prog = ("IN prev\nIN prev2\nIN bank\n"
                  "CTX = IMAD(prev, 2048, prev2)\n"
                  "ROW = GATHER3(bank, prev, prev2)\n"
                  "OUT = ARGMAX(ROW, 1)\n")
    shape_pay = {"prev": ids, "prev2": ids, "bank": big}
    for target in ("c", "nonfpu"):
        art = compile_program(shape_prog, target, sample=shape_pay,
                              outputs=["OUT"])
        ok = ("DIM_Dbank_0 2048L" in art["source"]
              and "DIM_Dbank_1 2048L" in art["source"]
              and art["streams"]["ROW"][2] == (512, 8))
        check(f"tref-shape-only-{target}", ok,
              "2048x2048x8 bank, no payload")
        if target == "nonfpu":
            try:
                float_trap(art["source"])
                check("tref-shape-only-nonfpu-trap-clean", True, "")
            except Exception as e:  # noqa: BLE001
                check("tref-shape-only-nonfpu-trap-clean", False,
                      str(e)[:140])

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
