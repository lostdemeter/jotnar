"""Diagonals threading-pattern demos: fuse() as a usable combinator.

Three demos, one script (CPU only, seconds):
1. fork-join map: 4 chains fused through one run_text, split-join,
   byte-exact vs separates (mixed row counts 1..4, shared weights).
2. shared-weights readers: 6 inferences, one readonly weight set --
   byte-exact; prints the no-locks observation (shared inputs pass
   through unstacked; the fused run is single execution, so there is
   nothing to lock).
3. halo policy: flagship prefix fused under halo="drop" (exact) and
   the full flagship under halo="refuse" (loud, naming the radius).
Usage: python3 demo_diagonals.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

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


def enc(rng, *sh):
    import phi_core.lattice as S
    return S.encode(np.ascontiguousarray(rng.normal(size=sh) * .5))


def demo_fork_join():
    from chain import asm as ASM, batch as B
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(21)
    W = enc(rng, 4, 3)
    lanes = [{"a": enc(rng, i + 1, 4), "b": W,
              "c": enc(rng, i + 1, 3)} for i in range(4)]
    t0 = time.perf_counter()
    tr = []
    got = B.fuse(CHAIN, lanes, trace=tr)
    wf = time.perf_counter() - t0
    t0 = time.perf_counter()
    refs = [ASM.run_text(CHAIN, REGISTRY, l, sigs=SIGS) for l in lanes]
    ws = time.perf_counter() - t0
    exact = all(np.array_equal(np.stack(got[i][k]), np.stack(refs[i][k]))
                for i in range(4) for k in ("MID", "SUM", "OUT"))
    print(f"fork-join: 4 lanes rows 1..4, fused {len(tr)} execs in "
          f"{wf * 1000:.1f}ms vs separate {4 * len(tr)} execs in "
          f"{ws * 1000:.1f}ms, split-join byte-exact: {exact}", flush=True)
    return exact


def demo_shared_weights():
    from chain import asm as ASM, batch as B
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(22)
    W = enc(rng, 4, 3)  # ONE readonly weight set, shared object
    lanes = [{"a": enc(rng, 2, 4), "b": W, "c": enc(rng, 2, 3)}
             for _ in range(6)]
    assert all(l["b"] is W for l in lanes)
    got = B.fuse(CHAIN, lanes)
    refs = [ASM.run_text(CHAIN, REGISTRY, l, sigs=SIGS) for l in lanes]
    exact = all(np.array_equal(np.stack(got[i]["OUT"]), np.stack(refs[i]["OUT"]))
                for i in range(6))
    print(f"shared-weights: 6 readers, 1 weight object (shared is W: "
          f"{all(l['b'] is W for l in lanes)}), byte-exact: {exact}",
          flush=True)
    print("no-locks observation: shared inputs pass through unstacked and "
          "the fused run is single execution -- one reader, one read, "
          "nothing to lock", flush=True)
    return exact


def demo_halo_policy():
    from PIL import Image
    from chain import batch as B
    im1 = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))
    H = im1.shape[0]
    lanes = [{"rgb": im1}, {"rgb": im1[::-1].copy()}]
    got = B.fuse(FLAG_PREFIX, lanes, halo="drop")
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    refs = [ASM.run_text(FLAG_PREFIX, REGISTRY, l, sigs=SIGS) for l in lanes]
    exact = all(np.array_equal(np.stack(got[i][k]), np.stack(refs[i][k]))
                for i in range(2) for k in ("LIN", "Y", "A"))
    print(f"halo-drop: flagship prefix (halo-0) fused, LIN/Y/A exact "
          f"both lanes: {exact}", flush=True)
    flag = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    try:
        B.fuse(flag, lanes)  # default halo="refuse"
        refused = False
    except B.BatchError as e:
        refused = True
        print(f"halo-refuse: full flagship under default policy: {e}",
              flush=True)
    return exact and refused


def main():
    ok = demo_fork_join() and demo_shared_weights() and demo_halo_policy()
    print("diagonals-demo:", "ALL OK" if ok else "MISMATCH", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
