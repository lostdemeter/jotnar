"""Stranger test (v1.0 Gate 6): docs-only build of a small program.

Exercise record: with ONLY docs/LANGUAGE.md open, wrote a 5-line
fizzbuzz-equivalent (input, transform, branch, output), ran it following
the §3 tutorial pattern, green on the FIRST try (OUT [0.35 0.35 0.25 0.25]:
SELECT picks S where the mask is true, a elsewhere). No code was read;
every construct used traces to a LANGUAGE.md section (cited below).

The listing (IN/ADD/SELECT are §§2-4 surface only):

    IN a
    IN b
    IN m
    S = ADD(a, b)
    OUT = SELECT(m, S, a)

Gates below pin the exercise: the green run, plus the two self-rescue
probes (a stranger's two most likely mistakes produce actionable errors).
Frictions found became docs/V1_1_BACKLOG.md -- if the docs alone had not
sufficed, v1.0 would wait; they did, with the frictions queued.
Usage: python3 tests/test_stranger.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []

FIZZ = """IN a
IN b
IN m
S = ADD(a, b)
OUT = SELECT(m, S, a)
"""


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def main():
    # the stranger run (§3 tutorial pattern: encode, dict payload, decode)
    a = S.encode(np.full(4, 0.25))
    b = S.encode(np.full(4, 0.10))
    m = np.array([True, True, False, False])  # §2 streams + §4 SELECT mask
    feeds = ASM.run_text(FIZZ, REGISTRY, {"a": a, "b": b, "m": m}, sigs=SIGS)
    got = dec(feeds["OUT"])
    check("stranger-green", bool((np.abs(got - [0.35, 0.35, 0.25, 0.25]) < 0.01).all()),
          f"docs-only 5-line listing runs green ({np.round(got, 3)})")
    # self-rescue 1: bare payload for a multi-IN listing (§2 convention)
    try:
        ASM.run_text(FIZZ, REGISTRY, a, sigs=SIGS)
        check("stranger-error-payload", False, "ran without error")
    except ASM.AsmError as e:
        check("stranger-error-payload", "dict payload" in str(e),
              f"actionable ({str(e)[:60]})")
    # self-rescue 2: typo'd mnemonic (fail-loud doctrine)
    try:
        ASM.run_text("IN a\nOUT = ADDD(a, a)\n", REGISTRY, a, sigs=SIGS)
        check("stranger-error-mnemonic", False, "ran without error")
    except ASM.AsmError as e:
        check("stranger-error-mnemonic", "ADDD" in str(e),
              f"names the culprit ({str(e)[:60]})")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
