"""Hopfield-capacity gate (docs/HOPFIELD.md constitution): CPU-only, seconds.

Legs:
- Lambert W: hand-rolled Newton reproduces W(e)=1, W(1)=Omega,
  W(0)=0 to 1e-12 (exact mathematical constants — the formulas
  are checked, not trusted).
- classical leg: 0.138N definitional (stated as such).
- shape legs: N_exp strictly increasing in d; c>1 every dim;
  crossover pinned (modern < classical at 64, > at 128).
- banked-table integrity: table() recomputed in-gate equals the
  banked JSON exactly (deterministic pure functions).
- verdict legs: recomputed verdicts + robustness flags match
  banked (64-insufficient robust; 512+ sufficient robust;
  128/256 rule-sufficient but fragile — stated, not hidden).
Usage: python3 tests/test_hopfield_capacity.py
"""
import json
import math
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []
BANK = os.path.join(ROOT, "samples", "marigold", "hopfield_capacity.json")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import hopfield_capacity as H

    check("hopfield-lambert",
          abs(H.lambert_w0(math.e) - 1.0) < 1e-12
          and abs(H.lambert_w0(1.0) - 0.5671432904097838) < 1e-12
          and H.lambert_w0(0.0) == 0.0,
          "W(e)=1, W(1)=Omega, W(0)=0")
    check("hopfield-classical",
          all(abs(H.capacity_classical(d) / d - 0.138) < 1e-15
              for d in H.DIMS),
          "0.138N definitional")
    exps = [H.capacity_exp(d)[0] for d in H.DIMS]
    check("hopfield-monotone",
          all(b > a for a, b in zip(exps, exps[1:])),
          "exponential shape (strictly increasing)")
    check("hopfield-c",
          all(H.capacity_exp(d)[1] > 1.0 for d in H.DIMS),
          f"c in {[round(H.capacity_exp(d)[1], 3) for d in H.DIMS]}")
    check("hopfield-crossover",
          exps[0] < H.capacity_classical(64)
          and exps[1] > H.capacity_classical(128),
          f"modern {exps[0]:.2f}<{H.capacity_classical(64):.2f} at 64, "
          f"{exps[1]:.0f}>{H.capacity_classical(128):.1f} at 128")
    rep = json.load(open(BANK))
    rows = H.table()
    check("hopfield-banked",
          {str(d): v for d, v in rows.items()} == rep["rows"]
          and rep.get("K") == 20 and rep.get("beta") == 1.0
          and rep.get("p") == 0.001,
          "table recomputed == banked (K=20, beta=1, p=0.001)")
    got = {str(d): (v["verdict"], v["robust_minus20"])
           for d, v in rows.items()}
    want = {str(d): (rep["rows"][str(d)]["verdict"],
                     rep["rows"][str(d)]["robust_minus20"])
            for d in H.DIMS}
    check("hopfield-verdicts", got == want
          and got == {"64": ("insufficient", "insufficient"),
                      "128": ("sufficient", "insufficient"),
                      "256": ("sufficient", "insufficient"),
                      "512": ("sufficient", "sufficient"),
                      "1024": ("sufficient", "sufficient")},
          f"{got}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
