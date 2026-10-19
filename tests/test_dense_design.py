"""Dense-predictor design integrity (design round): CPU-only seconds.

Every design component C1..C11 carries op-citations or a named gap;
every gap carries price + gate-shape + falsifier (A3-card reference
accepted where stated); verdict CLOSES/BLOCKED present with
justification; judge names strata bars + AbsRel/delta + alignment.
Tripwire: no implementation exists (named component files/patterns
absent; worktree diff vs main is exactly this doc + this gate).
Usage: python3 tests/test_dense_design.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FAIL = []
COMPS = [f"C{i}" for i in range(1, 12)]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    doc = open(os.path.join(ROOT, "docs", "DENSE_PREDICTOR.md")).read()
    bodies = {}
    for m in re.finditer(r"^- (C\d+) ", doc, re.M):
        start = m.start()
        nxt = doc.find("\n- C", start + 1)
        bodies[m.group(1)] = doc[start:nxt if nxt > 0 else len(doc)]
    check("design-components", set(bodies) == set(COMPS),
          f"{sorted(bodies)}")
    cited_ok = True
    for cid in COMPS:
        b = bodies.get(cid, "")
        if ("Ops:" not in b) and ("Gap:" not in b):
            cited_ok = False
            print(f"  uncited component {cid}", flush=True)
    check("design-cited-or-gap", bool(cited_ok),
          "every component cited or gap-named")
    gap_ok = True
    for cid, b in bodies.items():
        if "Gap:" not in b:
            continue
        bl = b.lower()
        has_own = ("price" in bl and "gate" in bl and "falsifier" in bl)
        has_ref = "per a3 card" in bl
        if not (has_own or has_ref):
            gap_ok = False
            print(f"  unpriced gap in {cid}", flush=True)
    check("design-gaps-priced", bool(gap_ok),
          "price + gate-shape + falsifier (A3-card refs accepted)")
    check("design-verdict",
          bool(re.search(r"Design verdict: (CLOSES|BLOCKED)", doc)),
          "CLOSES/BLOCKED with justification")
    jl = doc.lower()
    check("design-judge",
          "strata bars" in jl and "absrel" in jl
          and ("ransac" in jl or "least-squares" in jl),
          "bars + AbsRel/delta + alignment protocol")

    # -- tripwire: no implementation --------------------------------------
    r = subprocess.run(["git", "diff", "--name-only", "main...HEAD"],
                       capture_output=True, text=True, cwd=ROOT)
    changed = sorted(x for x in r.stdout.split() if x.strip()) if r.returncode == 0 else None
    r2 = subprocess.run(["git", "status", "--short"],
                        capture_output=True, text=True, cwd=ROOT)
    dirty = sorted(ln[3:] for ln in r2.stdout.splitlines()
                   if ln.strip() and not ln.startswith("??")) if r2.returncode == 0 else None
    untracked = sorted(ln[3:] for ln in r2.stdout.splitlines()
                       if ln.startswith("??")) if r2.returncode == 0 else None
    allowed = {"docs/DENSE_PREDICTOR.md", "tests/test_dense_design.py"}
    check("design-no-impl-files",
          changed is not None and dirty is not None and untracked is not None
          and set(changed) <= allowed and set(dirty) <= allowed
          and set(untracked) <= allowed,
          f"diff={changed} dirty={dirty} untracked={untracked}")
    cu = open(os.path.join(ROOT, "chain/emit_cuda.py")).read()
    cc = open(os.path.join(ROOT, "chain/emit_c.py")).read()
    check("design-no-impl-patterns",
          "k_conv" not in cu and "k_interp" not in cu
          and 'mn == "CONV"' not in cc
          and not os.path.exists(os.path.join(ROOT, "scripts/train_dpt_student.py")),
          "no CONV/INTERP emission, no DPT trainer")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
