"""Non-FPU gate v0.1: integer-only emission, proven without an FPU.

  1. bigram via target "nonfpu": bit-exact vs lattice (200 toks);
  2. float_trap() over every emitted source (no float/double/libm);
  3. links with libs=[] (no libm) and runs;
  4. T-moves program (TRANSPOSE/SLICE/CONCAT/SELECT over triples):
     bit-exact vs lattice;
  5. float program refused loud (MATMUL on F -> NoPattern).
Usage: python3 tests/test_emit_nonfpu.py (needs cc; fast).
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import NoPattern, build, compile_program
from chain.emit_nonfpu import float_trap

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    for f in ("lm_bigrams.npz",):
        if not os.path.isfile(os.path.join(dd, f)):
            print(f"SKIP (missing {f})")
            sys.exit(0)
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(sdir, "bigram_lm.asm")).read()
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
    bank = (np.ascontiguousarray(bs), np.ascontiguousarray(be),
            np.ascontiguousarray(bz))
    rng = np.random.default_rng(0)
    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    toks = np.array(sorted(rng.choice(nz, size=200, replace=False).tolist()),
                    np.int64)
    payload = {"tok": toks, "bank": bank}
    feeds = ASM.run_text(text, REGISTRY, payload, sigs=SIGS, basedir=sdir)
    ref = np.ascontiguousarray(feeds["OUT"]).reshape(-1)

    art = compile_program(text, "nonfpu", sample=payload, outputs=["OUT"],
                          basedir=sdir)
    try:
        float_trap(art["source"])
        check("nonfpu-trap-clean", True, "no float/double/libm in source")
    except Exception as e:  # noqa: BLE001 -- gate must report, not crash
        check("nonfpu-trap-clean", False, str(e)[:120])
    work = "/tmp/emit_nonfpu_v01"
    os.makedirs(work, exist_ok=True)
    exe = build(art["source"], work, name="bigram_nf", libs=[])
    fbs, fbe, fbz = (os.path.join(work, f"bank_{c}.bin") for c in "sez")
    f_tok, f_out = os.path.join(work, "tok.bin"), os.path.join(work, "out.bin")
    bs.tofile(fbs)
    be.tofile(fbe)
    bz.tofile(fbz)
    toks.tofile(f_tok)
    # argv order = inputs order (tok, bank): tok file, then 3 bank files
    r = subprocess.run([exe, f_tok, fbs, fbe, fbz, f_out],
                       capture_output=True, text=True)
    check("nonfpu-build-run-nolm", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]}")
    got = np.fromfile(f_out, dtype=np.int64)
    check("nonfpu-bigram-exact", got.shape == ref.shape and bool((got == ref).all()),
          f"{int((got == ref).sum())}/{len(ref)} identical, no libm")

    # -- T-moves program, bit-exact -----------------------------------
    mov = ("IN t\nIN msk\n"
           "TR = TRANSPOSE(t)\n"
           "SL = SLICE(TR, 1, 1, 3)\n"
           "CT = CONCAT(SL, SL, 1)\n"
           "MS = SELECT(msk, CT, CT)\n")
    rng2 = np.random.default_rng(3)
    t = (rng2.integers(-5, 6, size=(4, 6)).astype(np.int8),
         rng2.integers(32000, 33500, size=(4, 6)).astype(np.int32),
         (rng2.random(size=(4, 6)) < 0.2).astype(np.uint8))
    # Shapes: t (4,6) -> TRANSPOSE (6,4) -> SLICE ax1 [1,3) (6,2) ->
    # CONCAT ax1 (6,4); mask must be (6,4) to match.
    msk = np.array([[1, 0, 1, 1]] * 6, np.int64)
    pay2 = {"t": t, "msk": msk}
    f2 = ASM.run_text(mov, REGISTRY, pay2, sigs=SIGS, basedir=sdir)
    art2 = compile_program(mov, "nonfpu", sample=pay2, outputs=["MS"],
                           basedir=sdir)
    try:
        float_trap(art2["source"])
        check("nonfpu-moves-trap-clean", True, "")
    except Exception as e:  # noqa: BLE001
        check("nonfpu-moves-trap-clean", False, str(e)[:120])
    exe2 = build(art2["source"], work, name="moves_nf", libs=[])
    files = []
    for comp, arr in zip("sez", t):
        fn = os.path.join(work, f"t_{comp}.bin")
        np.ascontiguousarray(arr).tofile(fn)
        files.append(fn)
    fm = os.path.join(work, "msk.bin")
    msk.tofile(fm)
    fo = [os.path.join(work, f"out_MS_{c}.bin") for c in "sez"]
    r = subprocess.run([exe2, files[0], files[1], files[2], fm] + fo,
                         capture_output=True, text=True)
    check("nonfpu-moves-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:200]}")
    if r.returncode == 0:
        ref2 = f2["MS"]
        ok = True
        for comp, arr, dt in (("s", ref2[0], np.int8), ("e", ref2[1], np.int32),
                              ("z", ref2[2], np.uint8)):
            gotp = np.fromfile(os.path.join(work, f"out_MS_{comp}.bin"),
                               dtype=dt).reshape(np.shape(arr))
            same = bool((gotp == np.ascontiguousarray(arr)).all())
            ok = ok and same
        check("nonfpu-moves-exact", ok, "3 planes bit-identical")

    # -- float refused loud --------------------------------------------
    try:
        compile_program("IN a\nIN b\nOUT = MATMUL(a, b)\n", "nonfpu",
                        sample={"a": np.ones((2, 2)), "b": np.ones((2, 2))},
                        outputs=["OUT"])
        check("nonfpu-float-refused", False, "float matmul compiled")
    except NoPattern as e:
        check("nonfpu-float-refused", True, str(e)[:100])

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
