"""Agreement gate v0.1: emitted C == lattice interpreter, bit-exact.

Compiles programs/bigram_lm.asm to standalone C (libc only), runs both
on the same 200-token batch, demands identical OUT. Also pins the
missing-op contract: unsupported mnemonics fail loud (NoPattern).
Usage: python3 tests/test_emit_c.py (needs cc; ~seconds: V=512 bank).
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.emit_c import NoPattern, build, compile_program

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    for f in ("lm_bigrams.npz",):
        if not os.path.isfile(os.path.join(dd, f)):
            print(f"SKIP (run scripts/freeze_lm.py first: missing {f})")
            sys.exit(0)
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    text = open(os.path.join(root, "programs", "bigram_lm.asm")).read()
    sdir = os.path.join(root, "programs")
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

    nz = [i for i in range(counts.shape[0]) if counts[i].sum() > 0]
    rng = np.random.default_rng(0)
    toks = np.array(sorted(rng.choice(nz, size=min(200, len(nz)),
                                      replace=False).tolist()), np.int64)
    payload = {"tok": toks, "bank": bank}

    # reference (lattice interpreter)
    feeds = ASM.run_text(text, REGISTRY, payload, sigs=SIGS, basedir=sdir)
    ref = np.ascontiguousarray(feeds["OUT"]).reshape(-1)

    # compile -> standalone C -> binary
    art = compile_program(text, "c", sample=payload, outputs=["OUT"],
                          basedir=sdir)
    work = "/tmp/emit_c_v01"
    exe = build(art["source"], work)
    f_bank_s, f_bank_e, f_bank_z = (os.path.join(work, f"bank_{c}.bin")
                                    for c in "sez")
    f_tok, f_out = os.path.join(work, "tok.bin"), os.path.join(work, "out.bin")
    bs.tofile(f_bank_s)
    be.tofile(f_bank_e)
    bz.tofile(f_bank_z)
    toks.tofile(f_tok)
    r = subprocess.run([exe, f_tok, f_bank_s, f_bank_e, f_bank_z, f_out],
                       capture_output=True, text=True)
    check("c-build-run", r.returncode == 0, f"rc={r.returncode} {r.stderr[:200]}")
    got = np.fromfile(f_out, dtype=np.int64)
    check("c-agrees-bit-exact", got.shape == ref.shape and bool((got == ref).all()),
          f"{int((got == ref).sum())}/{len(ref)} identical, "
          f"e.g. ref={ref[:5].tolist()} got={got[:5].tolist()}")
    # binary independence: pure C, links libc only
    ldd = subprocess.run(["ldd", exe], capture_output=True, text=True).stdout
    check("c-standalone", "libc" in ldd and "phi" not in ldd and "python" not in ldd.lower(),
          ldd.strip().split("\n")[0] if ldd else "static?")

    # missing-op contract: ADD has no C pattern yet -> loud, not wrong
    try:
        compile_program("IN a\nIN b\nOUT = ADD(a, b)\n", "c",
                        sample={"a": np.array([1], np.int64),
                                "b": np.array([2], np.int64)},
                        outputs=["OUT"])
        check("c-missing-op-loud", False, "ADD compiled but has no pattern")
    except NoPattern as e:
        check("c-missing-op-loud", True, str(e)[:80])
    # unknown target -> loud
    try:
        compile_program(text, "cuda", sample=payload, outputs=["OUT"])
        check("c-unknown-target-loud", False, "cuda compiled but no backend")
    except NoPattern as e:
        check("c-unknown-target-loud", True, str(e)[:80])

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
