"""Pattern-gap proposal integrity (scoping round): CPU-only, seconds.

Every Track-A/B item named in docs/PATTERN_GAPS.md carries all four
price fields (Unlocks / Parity-gate shape / Size / Dependencies)
plus a Falsifier; the stated dependency order is acyclic (checked
programmatically from Depends-lines); no implementation exists
(scoping tripwire: files changed vs main are exactly this proposal
+ this gate, and the README parked rows still show no patterns —
so code can't sneak in under a docs round).
Usage: python3 tests/test_pattern_gaps.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FAIL = []
ITEMS = ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3"]
FIELDS = ["Unlocks:", "Parity-gate shape:", "Size:", "Dependencies:",
          "Falsifier:"]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    doc = open(os.path.join(ROOT, "docs", "PATTERN_GAPS.md")).read()
    bodies = {}
    for m in re.finditer(r"^### ([AB][1-6]) \u2014", doc, re.M):
        start = m.start()
        nxt = doc.find("\n### ", start + 1)
        bodies[m.group(1)] = doc[start:nxt if nxt > 0 else len(doc)]
    check("gaps-items-present", set(bodies) == set(ITEMS),
          f"{sorted(bodies)}")
    fields_ok = True
    for iid in ITEMS:
        b = bodies.get(iid, "")
        for f in FIELDS:
            if f not in b:
                fields_ok = False
                print(f"  missing {f} in {iid}", flush=True)
    check("gaps-price-fields", bool(fields_ok),
          "Unlocks/Gate/Size/Depends/Falsifier x9")

    # -- dependency acyclicity from Depends-lines -------------------------
    edges = {}
    for iid, b in bodies.items():
        dep_line = next((ln for ln in b.splitlines()
                         if ln.strip().startswith("Dependencies:")), "")
        refs = sorted(set(re.findall(r"\b[AB][1-6]\b", dep_line)) - {iid})
        edges[iid] = refs
    order, state, cyclic = [], {}, [False]

    def visit(n, stack):
        if state.get(n) == "done":
            return
        if n in stack:
            cyclic[0] = True
            return
        stack.append(n)
        for d in edges.get(n, []):
            visit(d, stack)
        stack.pop()
        state[n] = "done"
        order.append(n)

    for iid in ITEMS:
        visit(iid, [])
    check("gaps-acyclic", not cyclic[0],
          f"edges={ {k: v for k, v in edges.items() if v} }")

    # -- scoping tripwire: no implementation -------------------------------
    # Committed diff vs main AND worktree status: only the proposal +
    # this gate may appear (untracked worktree edits are invisible to
    # diff, so both are checked).
    r = subprocess.run(["git", "diff", "--name-only", "main...HEAD"],
                       capture_output=True, text=True, cwd=ROOT)
    changed = sorted(x for x in r.stdout.split() if x.strip()) if r.returncode == 0 else None
    r2 = subprocess.run(["git", "status", "--short"],
                        capture_output=True, text=True, cwd=ROOT)
    dirty = sorted(ln[3:] for ln in r2.stdout.splitlines()
                   if ln.strip() and not ln.startswith("??")) if r2.returncode == 0 else None
    untracked = sorted(ln[3:] for ln in r2.stdout.splitlines()
                       if ln.startswith("??")) if r2.returncode == 0 else None
    allowed = {"docs/PATTERN_GAPS.md", "tests/test_pattern_gaps.py"}
    check("gaps-no-impl-files",
          changed is not None and dirty is not None and untracked is not None
          and set(changed) <= allowed and set(dirty) <= allowed
          and set(untracked) <= allowed,
          f"diff={changed} dirty={dirty} untracked={untracked}")
    parked_ok = True
    for mn in ("LAYERNORM", "GELU"):
        line = next((ln for ln in open(os.path.join(ROOT, "README.md"))
                     if re.match(rf"^\| {mn} \|", ln)), "")
        cells = [c.strip() for c in line.split("|")[3:6]]
        parked_ok &= (cells == ["\u2014", "\u2014", "\u2014"])
    conv = next((ln for ln in open(os.path.join(ROOT, "README.md"))
                 if re.match(r"^\| CONV \|", ln)), "")
    ccells = [c.strip() for c in conv.split("|")[3:6]]
    parked_ok &= (ccells == ["\u2014", "\u2014", "\u2014"])
    check("gaps-parked-rows", bool(parked_ok),
          "LAYERNORM/GELU/CONV still patternless in README table")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
