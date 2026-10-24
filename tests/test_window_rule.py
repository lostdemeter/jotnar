#!/usr/bin/env python3
"""Windowing-rule gate (Q6).

Ruling: windowing is split by rank -- rank-1 I streams use IWINDOW (the
non-FPU byte path), rank>=2 F/T streams use SLICE.  The compiler errors
point across, and the byte-index family is registered consistently
(REGISTRY / SIGS / census / INTEGER_MN) with a gate test each.
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR",
                                  os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain import asm as ASM  # noqa: E402
from chain.asm_ops import REGISTRY, SIGS  # noqa: E402
from chain.census import ACCESS  # noqa: E402
from chain.emit_c import NoPattern, compile_program  # noqa: E402
from chain.emit_nonfpu import INTEGER_MN  # noqa: E402

FAIL = []

FAMILY = {"IWINDOW": "moves", "IGATHER1": "gather", "IASSEMBLE": "reduce",
          "ISUM": "reduce", "ISCATTER1": "gather"}
GATES = {"IWINDOW": "test_iwindow.py", "IGATHER1": "test_igather1.py",
         "IASSEMBLE": "test_ibyte.py", "ISUM": "test_ibyte.py",
         "ISCATTER1": "test_ibyte.py"}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    e1 = np.arange(8, dtype=np.int64)
    e2 = np.zeros((2, 4), dtype=np.int64)

    # SLICE on 1D I refuses and points to IWINDOW
    try:
        compile_program("IN E\nW = SLICE(E, 0, 0, 2)\n", "c",
                        sample={"E": e1}, outputs=["W"])
        check("window-rule-slice-1d", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as exc:
        check("window-rule-slice-1d", "use IWINDOW for 1D I" in str(exc),
              str(exc)[:100])

    # IWINDOW on rank >= 2 refuses and points to SLICE
    try:
        compile_program("IN E\nW = IWINDOW(E, 0, 2)\n", "c",
                        sample={"E": e2}, outputs=["W"])
        check("window-rule-iwindow-2d", False, "accepted")
    except (NoPattern, ASM.AsmError, ValueError) as exc:
        check("window-rule-iwindow-2d", "use SLICE for 2D+ F/T" in str(exc),
              str(exc)[:100])

    # the family is registered consistently and each op has a gate file
    for name, cls in FAMILY.items():
        check(f"family-{name}-registry", name in REGISTRY)
        check(f"family-{name}-sigs", name in SIGS)
        check(f"family-{name}-census", ACCESS.get(name) == cls,
              str(ACCESS.get(name)))
        check(f"family-{name}-nonfpu", name in INTEGER_MN)
        gate = os.path.join(ROOT, "tests", GATES[name])
        check(f"family-{name}-gate", os.path.isfile(gate), gate)

    # the split is by rank: IWINDOW's first operand is 1D I, SLICE's is any
    check("window-rule-iwindow-1d-i",
          SIGS["IWINDOW"][0][0] == "I:*", str(SIGS["IWINDOW"]))
    check("window-rule-slice-general",
          SIGS["SLICE"][0][0] == "$A", str(SIGS["SLICE"]))

    print("FAILURES:", FAIL if FAIL else "none")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
