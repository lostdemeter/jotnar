"""Diagonal batch gate (Phase 1): two inferences, one fused execution.

Same listing, two input sets (shared weights = the batch-decoding
shape): stack activation rows, run once, split outputs, byte-compare
vs two separate run_texts. Stacking choice: batch rows (no listing
change, no weight change -- the hypotenuse is just a taller matrix).
N=12 randomized trials, row counts 1..4; exactness every output
every trial; fused rerun byte-identical; op-execution counts from
the trace (fused 3 vs separate 3+3) + wall accounting printed.
Usage: python3 tests/test_diagonal.py (CPU only, seconds).
"""
import os
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
OUTS = ["MID", "SUM", "OUT"]
TRIALS = 12


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def cat(t1, t2):
    return tuple(np.concatenate([np.asarray(a), np.asarray(b)])
                 for a, b in zip(t1, t2))


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S

    bound = ASM.assemble(CHAIN, REGISTRY, SIGS)[2]
    nops = sum(1 for _ in bound)
    check("diag-static-ops", nops == 3, f"{nops} bound ops (one listing)")

    rng = np.random.default_rng(20261018)
    wf, ws, nf, ns = 0.0, 0.0, 0, 0
    for t in range(TRIALS):
        r = (t % 4) + 1  # row counts 1..4 across trials
        mk = lambda *sh: S.encode(
            np.ascontiguousarray(rng.normal(size=sh) * .5))
        A1, B, C1, A2, C2 = mk(r, 4), mk(4, 3), mk(r, 3), mk(r, 4), mk(r, 3)

        t0 = time.perf_counter()
        tr1, tr2 = [], []
        r1 = ASM.run_text(CHAIN, REGISTRY, {"a": A1, "b": B, "c": C1},
                          sigs=SIGS, trace=tr1)
        r2 = ASM.run_text(CHAIN, REGISTRY, {"a": A2, "b": B, "c": C2},
                          sigs=SIGS, trace=tr2)
        ws += time.perf_counter() - t0
        ns += len(tr1) + len(tr2)

        t0 = time.perf_counter()
        trf = []
        rf = ASM.run_text(CHAIN, REGISTRY,
                          {"a": cat(A1, A2), "b": B, "c": cat(C1, C2)},
                          sigs=SIGS, trace=trf)
        wf += time.perf_counter() - t0
        nf += len(trf)

        # determinism: fused rerun byte-identical
        rf2 = ASM.run_text(CHAIN, REGISTRY,
                           {"a": cat(A1, A2), "b": B, "c": cat(C1, C2)},
                           sigs=SIGS)
        det = all(np.array_equal(np.stack(rf[k]), np.stack(rf2[k]))
                  for k in OUTS)

        exact = True
        for k in OUTS:
            f = np.stack(rf[k])
            if not (np.array_equal(f[:, 0:r, :], np.stack(r1[k]))
                    and np.array_equal(f[:, r:2 * r, :], np.stack(r2[k]))):
                exact = False
        check(f"diag-t{t:02d}-exact", exact, f"R={r} rows split == separate")
        if t == 0:
            check("diag-determinism", det, "fused rerun byte-identical")

    check("diag-op-count", nf == TRIALS * 3 and ns == TRIALS * 6,
          f"fused execs={nf} vs separate execs={ns}")
    print(f"cost: fused {nf} op-execs in {wf * 1000:.1f}ms vs "
          f"separate {ns} op-execs in {ws * 1000:.1f}ms "
          f"(same {nops} static ops; placement honesty: dispatch halved, "
          f"flops identical -- no latency claim beyond this row)",
          flush=True)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
