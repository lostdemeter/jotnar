"""Diagonal DX gate: fuse() combinator legs + demo proof-run.

N-lane exactness (mixed rows, shared weights), default-refuse on the
full flagship, drop-exactness on the prefix AND on the full listing
outside the halo, exchange-refuses-loud with the seam-protocol
reason, unlisted-op refusal, per-lane error attribution (wrong shape
in lane 2 names lane 2), then the demo script itself green.
Usage: python3 tests/test_diagonal_dx.py (CPU only, seconds).
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

CHAIN = """CONFIG m_acc 31344
CONFIG m_cov 31344
IN a AS T:SEQ
IN b AS T:SEQ
IN c AS T:SEQ
MID = MATMUL(a, b)
SUM = ADD(MID, c)
OUT = SILU(SUM)
"""

FLAG_PREFIX = """CONFIG beta 0.5
IN rgb AS U8:HWC
LIN = SRGB_DECODE(rgb)
Y = LUMA(LIN)
A = SQRT(Y)
"""


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import asm as ASM, batch as B
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S
    from PIL import Image

    rng = np.random.default_rng(33)
    enc = lambda *sh: S.encode(np.ascontiguousarray(rng.normal(size=sh) * .5))

    # -- N-lane exactness (mixed rows, shared weights) ----------------------
    W = enc(4, 3)
    lanes = [{"a": enc(i + 1, 4), "b": W, "c": enc(i + 1, 3)}
             for i in range(4)]
    tr = []
    t0 = time.perf_counter()
    got = B.fuse(CHAIN, lanes, trace=tr)
    wf = time.perf_counter() - t0
    t0 = time.perf_counter()
    refs = [ASM.run_text(CHAIN, REGISTRY, l, sigs=SIGS) for l in lanes]
    ws = time.perf_counter() - t0
    check("dx-nlane-exact",
          bool(all(np.array_equal(np.stack(got[i][k]), np.stack(refs[i][k]))
                   for i in range(4) for k in ("MID", "SUM", "OUT"))),
          "4 lanes rows 1..4, every output byte-identical")
    check("dx-op-count", len(tr) == 3, f"one execution, {len(tr)} execs")
    print(f"dx-cost: fused {len(tr)} execs in {wf * 1000:.1f}ms vs "
          f"separate {4 * len(tr)} in {ws * 1000:.1f}ms (dispatch vs "
          f"flops disclaimer as Phase 1 -- direction not asserted)",
          flush=True)

    # -- flagship: default refuses, drop is exact ---------------------------
    im1 = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))
    H = im1.shape[0]
    imlanes = [{"rgb": im1}, {"rgb": im1[::-1].copy()}]
    flag = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    try:
        B.fuse(flag, imlanes)
        check("dx-default-refuse", False, "fused a spatial listing")
    except B.BatchError as e:
        check("dx-default-refuse", "radius 1" in str(e), str(e)[:80])
    got = B.fuse(FLAG_PREFIX, imlanes, halo="drop")
    refs = [ASM.run_text(FLAG_PREFIX, REGISTRY, l, sigs=SIGS) for l in imlanes]
    check("dx-prefix-drop",
          bool(all(np.array_equal(np.stack(got[i][k]), np.stack(refs[i][k]))
                   for i in range(2) for k in ("LIN", "Y", "A"))),
          "halo-0 prefix fully exact through fuse")
    got = B.fuse(flag, imlanes, halo="drop")
    refs = [ASM.run_text(flag, REGISTRY, l, sigs=SIGS) for l in imlanes]
    O = [np.asarray(got[i]["OUT"]) for i in range(2)]
    R = [np.asarray(refs[i]["OUT"]) for i in range(2)]
    check("dx-full-drop-halo",
          bool(np.array_equal(O[0][:H - 1], R[0][:H - 1])
               and np.array_equal(O[1], R[1][1:])),
          "full flagship under drop: exact outside the 1-row halo")

    # -- exchange + unlisted refuse loud ------------------------------------
    msgs = {}
    for tag, fn in [
            ("dx-exchange-refuses",
             lambda: B.fuse(CHAIN, lanes, halo="exchange")),
            ("dx-unlisted-refuses",
             lambda: B.fuse("IN x\nOUT = CONV(x, x)\n", lanes)),
            ("dx-bad-policy",
             lambda: B.fuse(CHAIN, lanes, halo="weave"))]:
        try:
            fn()
            check(tag, False, "accepted")
        except B.BatchError as e:
            msgs[tag] = str(e)
            check(tag, True, str(e)[:80])
    check("dx-exchange-reason", "seam protocol" in msgs.get("dx-exchange-refuses", ""),
          msgs.get("dx-exchange-refuses", "")[:80])

    # -- per-lane error attribution ------------------------------------------
    bad = [dict(l) for l in lanes]
    bad[2] = dict(bad[2], a=enc(9, 5))  # lane 2 wrong inner dim
    try:
        B.fuse(CHAIN, bad)
        check("dx-lane-attribution", False, "fused mismatched lanes")
    except B.BatchError as e:
        check("dx-lane-attribution", "lane 2" in str(e), str(e)[:90])

    # -- demo proof-run --------------------------------------------------------
    r = subprocess.run([sys.executable, os.path.join(ROOT, "demo_diagonals.py")],
                       capture_output=True, text=True, cwd=ROOT)
    markers = ("fork-join:", "shared-weights:", "no-locks observation:",
               "halo-drop:", "halo-refuse:", "diagonals-demo: ALL OK")
    check("dx-demo", r.returncode == 0 and all(m in r.stdout for m in markers),
          r.stderr[-200:] if r.returncode else "all three demos printed")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
