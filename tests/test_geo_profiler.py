"""Geometric profiler gate (BACKLOG item 1): one tool, static + dynamic.

Legs (CPU-only, seconds):
- self-consistency on two listings (flagship 24x32 synthetic rgb +
  demo_img 12x16 triples, sizes chosen for seconds; demo size matches the
  fusion gate's 12x16 so the static halves cross-check exactly):
  static passes/bytes prediction vs measured within band 0.40.
- static cross-checks vs the fusion gate's hand-read numbers (flagship
  29 passes / 10 saved / 1 chain; demo 109 passes / 28 saved) -- the two
  halves unified, not just co-located.
- --budget legs: near-zero budget -> every op over; huge budget -> every
  op under; burn-down sorted worst-first; over sums to total-wall-budget.
- determinism: same listing twice -> identical canonical report (absolute
  wall leaves excluded, documented) AND identical verdict lists (5/5
  stable on-box, pasted in BACKLOG).
- synthetic verdict edges: dispatch (tiny+slow mover), traffic (fast
  mover), compute (slow+heavy) incl. the roof-op-is-traffic rule.
Usage: python3 tests/test_geo_profiler.py
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def flagship():
    from chain import geo_profiler as G
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    pay = G.flagship_payload(24, 32)
    return text, pay, "flagship:24x32"


def demo():
    from chain import geo_profiler as G
    return (*G.demo_payload(12, 16), "demo:12x16")


def main():
    from chain import geo_profiler as G
    from chain.asm_ops import REGISTRY, SIGS

    # -- flagship self-consistency ---------------------------------------
    text, pay, name = flagship()
    rep = G.profile_text(text, REGISTRY, pay, sigs=SIGS,
                         basedir=os.path.join(ROOT, "programs"),
                         budget_ms=33.0)
    c = rep["consistency"]
    check("geo-flagship-band", c["ok"] and c["wmape"] <= G.BAND,
          f"WMAPE={c['wmape']:.3f} band={G.BAND}")
    t = rep["totals"]
    check("geo-flagship-static", t["n_ops"] == 11 and t["total_passes"] == 29
          and t["passes_saved"] == 10 and t["n_chains"] == 1, str(
              {k: t[k] for k in ("n_ops", "total_passes", "passes_saved",
                                 "n_chains")}))
    check("geo-flagship-fused",
          [o["op"] for o in rep["ops"] if o["fused"]] ==
          ["BETA_V5", "MUL", "ADD", "SQUARE", "GAIN", "SRGB_ENCODE"],
          "advisor chain visible per-op")
    check("geo-flagship-verdicts",
          sorted((o["op"], o["verdict"]) for o in rep["ops"]) ==
          sorted([("SRGB_DECODE", "traffic"), ("LUMA", "traffic"),
                  ("SQRT", "traffic"), ("SPLAT_BLUR", "compute"),
                  ("SUB", "compute"), ("BETA_V5", "compute"),
                  ("MUL", "traffic"), ("ADD", "compute"),
                  ("SQUARE", "compute"), ("GAIN", "compute"),
                  ("SRGB_ENCODE", "traffic")]),
          "moves/traffic vs lattice-arithmetic/compute split")

    # -- demo self-consistency --------------------------------------------
    dtext, dpay, dname = demo()
    drep = G.profile_text(dtext, REGISTRY, dpay, sigs=SIGS,
                          basedir="programs", budget_ms=33.0)
    dc = drep["consistency"]
    check("geo-demo-band", dc["ok"] and dc["wmape"] <= G.BAND,
          f"WMAPE={dc['wmape']:.3f} band={G.BAND}")
    dt = drep["totals"]
    check("geo-demo-static", dt["n_ops"] == 43 and dt["total_passes"] == 109
          and dt["passes_saved"] == 28, str(
              {k: dt[k] for k in ("n_ops", "total_passes", "passes_saved")}))
    from collections import Counter
    check("geo-demo-verdicts",
          Counter(o["verdict"] for o in drep["ops"]) ==
          Counter({"traffic": 27, "compute": 16}),
          "memcpy movers traffic, lattice ADD/SUB compute, zero dispatch")

    # -- --budget legs (known over/under) ----------------------------------
    tiny = G.profile_text(text, REGISTRY, pay, sigs=SIGS,
                          basedir=os.path.join(ROOT, "programs"),
                          budget_ms=1e-6)
    check("geo-budget-all-over",
          all(r["over_ms"] > 0 for r in tiny["budget"]["rows"]),
          "1ns budget: every op over")
    huge = G.profile_text(text, REGISTRY, pay, sigs=SIGS,
                          basedir=os.path.join(ROOT, "programs"),
                          budget_ms=1e9)
    check("geo-budget-all-under",
          all(r["over_ms"] < 0 for r in huge["budget"]["rows"]),
          "1e9ms budget: every op under")
    b = rep["budget"]
    overs = [r["over_ms"] for r in b["rows"]]
    check("geo-budget-sorted", overs == sorted(overs, reverse=True),
          "burn-down worst-first")
    check("geo-budget-sums",
          abs(sum(overs) - b["total_over_ms"]) < 1e-6
          and abs(b["total_over_ms"] -
                  (t["total_wall_s"] * 1000.0 - 33.0)) < 1e-6,
          f"sum(over)={sum(overs):+.3f} total={b['total_over_ms']:+.3f}")
    check("geo-budget-head",
          b["rows"][0]["op"] == "SPLAT_BLUR" and b["rows"][0]["over_ms"] > 0,
          "33ms frame: SPLAT_BLUR is the burn-down head")

    # -- determinism: same listing -> identical report ----------------------
    rep2 = G.profile_text(text, REGISTRY, pay, sigs=SIGS,
                          basedir=os.path.join(ROOT, "programs"),
                          budget_ms=33.0)
    check("geo-determinism-canonical",
          G.canonical(rep) == G.canonical(rep2),
          "static + bytes + verdicts bit-identical across runs")
    check("geo-determinism-bytes",
          [o["bytes"] for o in rep["ops"]] ==
          [o["bytes"] for o in rep2["ops"]] and
          [o["bytes"] for o in drep["ops"]] ==
          [o["bytes"] for o in
           G.profile_text(dtext, REGISTRY, dpay, sigs=SIGS)["ops"]],
          "measured bytes exactly reproducible (shapes, not clock)")
    check("geo-determinism-verdicts",
          [o["verdict"] for o in rep["ops"]] ==
          [o["verdict"] for o in rep2["ops"]],
          "verdicts stable run-to-run")

    # -- synthetic verdict edges (pure function, no clock) -------------------
    v, _ = G._verdict(1e-6, 100, 1e-6, 1e9)
    check("geo-verdict-dispatch", v == "dispatch",
          "fastest op + slow mover -> dispatch-bound")
    v, _ = G._verdict(1e-6, 10**9, 1e-6, 1e9)
    check("geo-verdict-roof", v == "traffic",
          "roof op itself is traffic, never dispatch")
    v, _ = G._verdict(1.0, 10**9, 1e-6, 1e9)
    check("geo-verdict-traffic", v == "traffic", "fast mover -> traffic")
    v, _ = G._verdict(1.0, 100, 1e-6, 1e9)
    check("geo-verdict-compute", v == "compute", "slow + heavy -> compute")

    # -- fail-loud legs -------------------------------------------------------
    try:
        G.stream_bytes({"kind": "nope", "shape": (4,)})
        check("geo-loud-kind", False, "unknown kind must raise")
    except G.ProfilerError:
        check("geo-loud-kind", True, "unknown kind raises ProfilerError")
    try:
        G.profile_text("IN a\nOUT = ADD(a, b)\n", REGISTRY, {"a": 1.0},
                       sigs=SIGS)
        check("geo-loud-unbound", False, "unbound stream must raise")
    except Exception as e:
        check("geo-loud-unbound", True, f"raised {type(e).__name__}")

    # -- CLI smoke (--budget required) ----------------------------------------
    r = subprocess.run(
        [sys.executable, "-m", "chain.geo_profiler", "--listing",
         os.path.join(ROOT, "programs", "holo_flagship.asm"),
         "--payload", "flagship:24x32", "--budget", "33"],
        capture_output=True, text=True, cwd=ROOT)
    check("geo-cli-rc", r.returncode == 0, f"rc={r.returncode} "
          f"{r.stderr[-200:] if r.returncode else ''}")
    check("geo-cli-table", "burn-down" in r.stdout and "traffic" in r.stdout
          and "WMAPE=" in r.stdout, "table + verdicts + consistency printed")
    r = subprocess.run(
        [sys.executable, "-m", "chain.geo_profiler", "--listing",
         os.path.join(ROOT, "programs", "holo_flagship.asm"),
         "--payload", "flagship:24x32"],
        capture_output=True, text=True, cwd=ROOT)
    check("geo-cli-budget-required", r.returncode != 0,
          "--budget omitted -> usage error (flag is required, not optional)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
