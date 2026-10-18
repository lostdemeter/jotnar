"""Fusion advisor gate: reported chains match hand-read listings.

Expected sets determined by reading the listings (how, per listing):
- flagship (11 ops): fanout map (census) gives A:3 D:2 LIN:2 Y:2 as
  blockers; the only fanout-1 pointwise path is
  AS->SUB(dead-end: D fanout2) ... BEFF->MUL->BD->ADD->AE->SQUARE->
  YENH->GAIN->G->SRGB_ENCODE, i.e. exactly [BETA_V5,MUL,ADD,SQUARE,
  GAIN,SRGB_ENCODE], 6 ops, 10 passes, sink boundary. Singletons
  ([SRGB_DECODE],[SQRT],[SUB]) correctly skipped (min length 2).
- demo_img (12x16, 43 ops): longest-path greedy (program-order
  tiebreak) over fanout-1 pointwise edges gives [MUL,ADD,ADD]
  (t0-line; t1/t3 overlap-skipped), [SUB,MUL,ADD] (t4-line),
  [ADDx8,MUL] (s-line), [SUB,MUL,ADD] (detail-line): 18 ops,
  28 passes; blocked gray:13 blur:2 crop:3.
- decode program (619 ops, serve_decode.build_decode(16)): the fused
  opcodes ate everything (SWIGLU/ATTN/ROPE absorb; MATMUL/RMSNORM/
  SCATTER/GATHER/ARGMAX bound) -- zero fusible->fusible fanout-1
  edges exist (cross-checked in-gate, not trusted blind), so zero
  chains is correct, not a bug. No MATMUL in any reported chain.
Escape leg: flagship with outputs={OUT,AE} reports zero chains (the
AE-interior chain blocked) vs one chain normally -- the delta proves
the no-escape rule fired. Determinism: advise twice, identical.
Usage: python3 tests/test_fusion_advisor.py (CPU only, seconds).
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    from chain import census as C
    from chain import fusion as F
    from chain.asm_ops import REGISTRY, SIGS
    from demo_img import build_img_prog
    from serve_decode import build_decode

    # -- flagship ------------------------------------------------------------
    ftext = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    rep = F.advise(ftext, REGISTRY, sigs=SIGS)
    check("fus-flagship-n", len(rep["chains"]) == 1,
          f"{len(rep['chains'])} chains")
    c = rep["chains"][0] if rep["chains"] else {}
    check("fus-flagship-ops",
          c.get("ops") == ["BETA_V5", "MUL", "ADD", "SQUARE", "GAIN",
                           "SRGB_ENCODE"], c.get("ops"))
    check("fus-flagship-profit",
          c.get("passes_saved") == 10 and c.get("boundary") == "sink"
          and c.get("out") == "OUT",
          f"{c.get('passes_saved')} passes, {c.get('boundary')}")
    check("fus-flagship-blocked",
          {k: v["fanout"] for k, v in rep["blocked"].items()}
          == {"LIN": 2, "A": 3, "Y": 2, "D": 2}, rep["blocked"])
    check("fus-flagship-clean", rep["unanalyzable"] == [],
          rep["unanalyzable"])
    check("fus-flagship-no-reduce",
          all(op not in ("MATMUL", "BATCH_MATMUL") for op in c.get("ops", [])),
          "no reduction inside")
    rep2 = F.advise(ftext, REGISTRY, sigs=SIGS,
                    outputs={"OUT", "AE"})
    check("fus-flagship-escape", rep2["chains"] == [],
          "AE-interior chain blocked by explicit outputs (delta proves it)")

    # -- demo_img -------------------------------------------------------------
    dtext = build_img_prog(12, 16).text()
    rep = F.advise(dtext, REGISTRY, sigs=SIGS)
    seqs = sorted([tuple(c["ops"]) for c in rep["chains"]])
    check("fus-demo-seqs", seqs == sorted([
        ("MUL", "ADD", "ADD"), ("SUB", "MUL", "ADD"),
        ("SUB", "MUL", "ADD"),
        ("ADD", "ADD", "ADD", "ADD", "ADD", "ADD", "ADD", "ADD", "MUL")]),
        seqs)
    check("fus-demo-profit", rep["summary"]["passes_saved"] == 28
          and rep["summary"]["n_fused_ops"] == 18, rep["summary"])
    check("fus-demo-blocked",
          {k: v["fanout"] for k, v in rep["blocked"].items()}
          == {"gray": 13, "blur": 2, "crop": 3}, rep["blocked"])
    check("fus-demo-clean", rep["unanalyzable"] == [], rep["unanalyzable"])

    # -- decode program: zero chains, cross-checked -----------------------------
    p, _, _ = build_decode(16)
    rep = F.advise(p.text(), REGISTRY, sigs=SIGS)
    check("fus-decode-empty", rep["chains"] == []
          and rep["summary"]["n_fused_ops"] == 0,
          f"{len(rep['chains'])} chains")
    crep = C.census_text(p.text(), REGISTRY, sigs=SIGS)
    fus = F.fusible_set()
    nedges = 0
    for s, st in crep["streams"].items():
        if st["producer"] is None or st["fanout"] != 1:
            continue
        pop = st["producer"][0]
        cmn = st["consumers"][0][0]
        if pop in fus and cmn in fus:
            nedges += 1
    check("fus-decode-crosscheck", nedges == 0,
          f"{nedges} fusible->fusible fanout-1 edges (zero => empty correct)")
    check("fus-decode-clean", rep["unanalyzable"] == [], rep["unanalyzable"][:2])

    # -- determinism ---------------------------------------------------------------
    a = F.advise(ftext, REGISTRY, sigs=SIGS)
    b = F.advise(ftext, REGISTRY, sigs=SIGS)
    check("fus-determinism", a == b, "identical reports across runs")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
