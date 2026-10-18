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

DCHAIN = """CONFIG m_acc 31344
CONFIG m_cov 31344
IN a AS T:SEQ
IN b AS T:SEQ
IN c AS T:SEQ
MID = MATMUL(a, b)
SUM = ADD(MID, c)
OUT = SILU(SUM)
"""


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

    # -- profile legs: the dispatch-hunt table, deterministic form ------------
    # Walls live in docs/FUSION.md (environmental, never asserted); the
    # gate asserts the structural counts the table rests on, plus the
    # graph tripwire (pure-Python compile, no GPU) proving per-op launch
    # cost is zero on the served path.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from demo_img import build_img_prog
    from serve_decode import build_decode
    drep = F.advise(build_img_prog(12, 16).text(), REGISTRY, sigs=SIGS)
    check("fx-profile-flagship",
          rep["summary"]["n_ops"] == 11 and len(rep["chains"]) == 1
          and rep["summary"]["passes_saved"] == 10,
          "11 ops / 1 chain / 10 passes")
    check("fx-profile-demo",
          drep["summary"]["n_ops"] == 43 and len(drep["chains"]) == 4
          and drep["summary"]["passes_saved"] == 28,
          "43 ops / 4 chains / 28 passes")
    p, _, _ = build_decode(16)
    qrep = F.advise(p.text(), REGISTRY, sigs=SIGS)
    check("fx-profile-decode",
          qrep["summary"]["n_ops"] == 619 and qrep["chains"] == [],
          "619 ops / 0 chains (nothing dispatch-fusible there)")
    import phi_core.lattice as S
    rng = np.random.default_rng(5)
    mk = lambda *sh: S.encode(np.ascontiguousarray(rng.normal(size=sh) * .5))
    WB = mk(4, 3)
    lanes = [{"a": mk(2, 4), "b": WB, "c": mk(2, 3)} for _ in range(4)]
    from chain import batch as B
    t0 = time.perf_counter()
    B.fuse(DCHAIN, lanes)
    wf = time.perf_counter() - t0
    t0 = time.perf_counter()
    nsep = 0
    for l in lanes:
        tr = []
        ASM.run_text(DCHAIN, REGISTRY, l, sigs=SIGS, trace=tr)
        nsep += len(tr)
    ws = time.perf_counter() - t0
    check("fx-toy-dispatch", nsep == 12,
          f"4 lanes x 3 execs separate vs 3 fused (dispatch-dominated toy)")
    print(f"fx-toy-wall: separate {ws * 1000:.1f}ms vs fused {wf * 1000:.1f}ms "
          f"(reported, never asserted)", flush=True)
    from chain.emit_c import compile_program
    tiny = ("CONFIG m_acc 41200\nIN x\nIN k\nY = MUL(x, k)\n"
            "OUT = ADD(Y, k)\n")
    samp = {"x": np.arange(32, dtype=np.float64).reshape(4, 8) + 1.0,
            "k": np.full((4, 8), 2.0)}
    import re as _re
    counts = {}
    for g in (False, True):
        art = compile_program(tiny, "cuda", sample=dict(samp),
                              outputs=["OUT"], live=["x"], graph=g,
                              sync_each=not g)
        loop = art["source"].split("while (fgets")[1]
        counts[g] = (len(_re.findall(r"<<<", loop)),
                     len(_re.findall(r"cudaGraphLaunch", loop)))
    check("fx-graph-amortizes", counts[False] == (2, 0)
          and counts[True] == (0, 1),
          f"per-op launches without graphs {counts[False]} vs one replay "
          f"launch with graphs {counts[True]} (emit_cuda graph capture)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
