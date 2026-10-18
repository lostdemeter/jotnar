"""Fused-elementwise codegen gate: advisor chain runs fused on lattice.

The flagship 6-op sink chain (BETA_V5,MUL,ADD,SQUARE,GAIN,SRGB_ENCODE)
executes via chain/fuse_exec.run_fused with the advisor's lines wired
in mechanically (the gate asserts the consumed chain IS the advisor's
current report -- hand-wiring without the advisor link is refused by
construction here: lines come from advise(), never literals).
Legs: materialized outputs byte-identical; interiors absent from
feeds (non-materialization proven) with values exact via env;
fused count == 5 (6th = chain-final, materialized); determinism
rerun; advisor report identical (consumed, not moved); wall
accounting printed, direction never asserted.
CUDA leg: documents the block -- T-kind chain has zero CUDA coverage
(decode-at-data-prep doctrine + no BETA_V5/GAIN/SRGB patterns); the
compile attempt must refuse loud with NoPattern naming T. The day it
compiles, this leg fails demanding the real parity gate. Stamp keys
for fused binaries are specified in docs/FUSION.md, unbuilt (no fused
binaries exist to stamp -- dead contract refused).
Usage: python3 tests/test_fusion_codegen.py (CPU lattice always runs).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

WANT_OPS = ["BETA_V5", "MUL", "ADD", "SQUARE", "GAIN", "SRGB_ENCODE"]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def N(v):
    return np.stack(v) if isinstance(v, tuple) else np.asarray(v)


def main():
    from PIL import Image
    from chain import asm as ASM, fusion as F, fuse_exec as FE
    from chain.asm_ops import REGISTRY, SIGS

    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    rep = F.advise(text, REGISTRY, sigs=SIGS)
    check("fx-advisor-link", len(rep["chains"]) == 1
          and rep["chains"][0]["ops"] == WANT_OPS,
          "consuming the advisor's 6-op chain, nothing hand-wired")
    ch = rep["chains"][0]
    rgb = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))

    t0 = time.perf_counter()
    ref = ASM.run_text(text, REGISTRY, {"rgb": rgb}, sigs=SIGS)
    tu = time.perf_counter() - t0
    t0 = time.perf_counter()
    got, n, env = FE.run_fused(text, ch["lines"], {"rgb": rgb})
    tf = time.perf_counter() - t0

    check("fx-materialized",
          bool(all(np.array_equal(N(got[k]), N(ref[k]))
                   for k in got if k != "CONFIG")),
          "every materialized output byte-identical")
    interior = list(ch["streams"])
    check("fx-nonmaterialized",
          bool(all(s not in got for s in interior)), interior)
    check("fx-interior-exact",
          bool(all(np.array_equal(N(env[s]), N(ref[s])) for s in interior)),
          "fused values exact via env")
    check("fx-count", n == 5, f"{n} interior ops fused (6th materialized)")
    _, _, env2 = FE.run_fused(text, ch["lines"], {"rgb": rgb})
    check("fx-determinism",
          bool(all(np.array_equal(N(env[s]), N(env2[s])) for s in interior)),
          "rerun byte-identical")
    rep2 = F.advise(text, REGISTRY, sigs=SIGS)
    check("fx-advisor-identical", rep == rep2, "consumed, not moved")
    print(f"fx-wall: unfused {tu * 1000:.1f}ms vs fused {tf * 1000:.1f}ms "
          f"(feeds-dict traffic only; numpy temporaries dominate both -- "
          f"direction reported, never asserted)", flush=True)

    # -- CUDA leg: the block, documented as a tripwire -------------------------
    try:
        from chain.emit_c import compile_program
        sample = {"rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
        compile_program(text, "cuda", sample=sample, outputs=["OUT"],
                        basedir=os.path.join(ROOT, "programs"))
        check("fx-cuda-block", False, "T chain compiled for CUDA?!")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # block surface, any order: T-kind refusal, or the ops with zero
        # emitted patterns anywhere (BETA_V5/GAIN/SRGB_*). The day none
        # of these fire (it compiled), this leg fails demanding parity.
        covered = ("T input" in str(e) or "no pattern for SRGB" in str(e)
                   or "no pattern for BETA_V5" in str(e)
                   or "no pattern for GAIN" in str(e))
        check("fx-cuda-block", type(e).__name__ == "NoPattern" and covered,
              msg[:140])

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
