"""Resonance-depth gate (docs/RESONANCE.md constitution): CPU-only, seconds.

Legs:
- phase determinism: same key -> same phase, in-process repeat AND a
  fresh subprocess (cross-run, seedless by construction).
- sibling pristine: import wrote nothing into their tree (no
  __pycache__ — dont_write_bytecode discipline).
- null sanity (pure math, ms): shuffled-keys resonance on a tiny
  synthetic set matches the any-head chance rate within tolerance.
- banked correlation legs: schema integrity (3 frozen variants,
  4032 tokens, 8.1M pairs) + lift vs the frozen 0.10 bar, reported
  against it, never asserted beyond it (verdict must read
  dead-for-depth with all lifts < bar).
Usage: python3 tests/test_resonance_depth.py
"""
import json
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []
SIBLING = "/home/balrog/Documents/OpenCode/resonant_array"
BANK = os.path.join(ROOT, "samples", "marigold", "resonance_depth.json")
BAR = 0.10


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    sys.dont_write_bytecode = True
    sys.path.insert(0, SIBLING)
    import resonant_array as RA

    # -- phase determinism (in-process + fresh subprocess) -----------------
    p1 = [RA.phase(123456, g) for g in RA.GAMMAS[:4]]
    p2 = [RA.phase(123456, g) for g in RA.GAMMAS[:4]]
    check("resonance-phase-repeat", p1 == p2, "same key, same phases")
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.dont_write_bytecode=True; "
         f"sys.path.insert(0, {SIBLING!r}); "
         "import resonant_array as R; "
         "print([R.phase(123456, g) for g in R.GAMMAS[:4]])"],
        capture_output=True, text=True)
    check("resonance-phase-xrun",
          r.returncode == 0 and eval(r.stdout.strip()) == p1,
          "fresh process, identical phases")

    # -- sibling pristine (their tree, their rules) -------------------------
    import glob as _g
    strays = (_g.glob(os.path.join(SIBLING, "__pycache__", "*"))
              + _g.glob(os.path.join(SIBLING, "*.pyc")))
    check("resonance-sibling-clean", strays == [],
          "no bytecode written into sibling tree")

    # -- banked correlation legs (schema + frozen bar) ----------------------
    assert os.path.isfile(BANK), "banked analysis missing — run the script"
    rep = json.load(open(BANK))
    res = rep.get("results", {})
    check("resonance-schema",
          set(res) == {"V1_luma", "V2_lumapos", "V3_rgbpack"}
          and all(set(v) >= {"rate_same", "rate_null", "lift",
                             "spearman", "pass"} for v in res.values())
          and rep.get("bar") == BAR,
          "3 frozen variants + bar intact")

    # -- null cross-check: two independent null constructions agree -------
    # Synthetic uniform keys resonate at ~0.34, NOT the naive 0.59:
    # heads share the key (phase_h = γ_h·key), so per-head phases are
    # deterministically linked, never independent — the 1-(1-2τ)^H
    # formula assumed independence and is wrong (spec correction
    # recorded). The honest baseline is empirical unpaired keys;
    # uniform-synthetic must match the seeded-shuffle neighborhood.
    import random
    rng = random.Random(0)
    keys = [rng.randrange(10 ** 6) for _ in range(400)]
    m = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if any(RA.circ_dist(RA.phase(keys[i], g),
                                RA.phase(keys[j], g)) < 0.1
                   for g in RA.GAMMAS[:4]):
                m += 1
    syn = m / (len(keys) * (len(keys) - 1) / 2)
    nulls = [v["rate_null"] for v in res.values()]
    check("resonance-null-xcheck",
          abs(syn - sum(nulls) / len(nulls)) < 0.02,
          f"synthetic-uniform {syn:.3f} vs seeded-shuffle "
          f"{sum(nulls) / len(nulls):.3f} (independence formula void)")
    lifts = {k: v["lift"] for k, v in res.items()}
    check("resonance-lifts",
          all(v < BAR for v in lifts.values()),
          f"lifts { {k: round(v, 4) for k, v in lifts.items()} } "
          f"all < bar {BAR}")
    check("resonance-verdict", rep.get("verdict") == "dead-for-depth",
          "banked verdict matches frozen rule")
    check("resonance-null-documented",
          all(0.30 < v["rate_null"] < 0.40 for v in res.values()),
          "empirical nulls ~0.35 (independence formula void — heads "
          "share the key; seeded null rules)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
