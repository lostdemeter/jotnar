"""Integer/index algebra gate: IADD/ISUB/IMUL/IMAD + GATHER3.

Proves the n-gram path end to end:
  * I-kind arithmetic is exact and wraps as specified (C + non-FPU);
  * GATHER3 reads (V1,V2,C) rows by two ids and feeds ARGMAX;
  * C, non-FPU, and CUDA agree where the backend admits the kinds
    (CUDA F tables only, same data-prep discipline as GATHER);
  * non-FPU source stays float-free and links without libm;
  * bad table/ids kinds and bounds fail loud.
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


def _write_inputs(work, art, payload, f32=False):
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
            np.ascontiguousarray(payload[n],
                                 dtype=np.float32 if f32 else np.float64).tofile(fn)
            files.append(fn)
        else:
            fn = os.path.join(work, f"in_{n}.bin")
            np.ascontiguousarray(payload[n], dtype=np.int64).tofile(fn)
            files.append(fn)
    return files


def _read_outputs(work, art):
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
            out[o] = np.fromfile(fn, dtype=np.float32).reshape(sh)
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
    return _read_outputs(work, art)


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

    # -- 4. loud refusals ----------------------------------------------
    bad = {
        "2d-table": ("IN i\nIN j\nIN t\nOUT = GATHER3(t, i, j)\n",
                     {"i": prev, "j": prev, "t": np.zeros((V, C))}),
        "id-length": ("IN i\nIN j\nIN t\nOUT = GATHER3(t, i, j)\n",
                      {"i": prev, "j": prev2[:-1],
                       "t": np.zeros((V, V, C))}),
        "float-iadd": ("IN a\nIN b\nOUT = IADD(a, b)\n",
                       {"a": np.ones(4), "b": np.ones(4)}),
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

    # -- 5. shape-only T bank (no multi-GB allocation at compile) -------
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
