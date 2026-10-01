"""Demo 3 (v1.5): modify between executions, preview twice in a row.

The assoc_mem store (test_demo2.py) edited across runs with predicted
recall, confirmed consecutively: run base (16 patterns) -> EDIT A (drop
to 8: fewer competitors, margins widen, predict 100% @<=8) -> EDIT B
(add 8 fresh patterns back to a DIFFERENT 16: capacity reasoning as
demo 2, predict 100% @<=8). Two consecutive correct previews = edits
work repeatedly, not once by luck. Same listing text throughout; only
the store data changes between runs (flexibility = data, not code).
Usage: python3 tests/test_demo3.py (fast)
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
DIM = 64


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    text = open(os.path.join(root, "programs", "assoc_mem.asm")).read()
    sdir = os.path.join(root, "programs")

    def make_store(seed, n):
        r = np.random.default_rng(seed)
        return r.choice([-1.0, 1.0], size=(n, DIM))

    def rate(pats, nflip, trials=4):
        keys, values = pats.T.copy(), pats.copy()

        def recall(cue):
            feeds = ASM.run_text(text, REGISTRY,
                                 {"cue": enc(cue.reshape(1, -1)),
                                  "keys": enc(keys), "values": enc(values)},
                                 sigs=SIGS, basedir=sdir)
            return np.asarray(feeds["OUT"][0]).reshape(-1) > 0

        ok = tot = 0
        r = np.random.default_rng(1000 + nflip)
        for i in range(pats.shape[0]):
            for _ in range(trials):
                cue = pats[i].copy()
                cue[r.choice(DIM, size=nflip, replace=False)] *= -1
                got = np.where(recall(cue), 1.0, -1.0)
                ok += bool((got == pats[i]).all())
                tot += 1
        return ok / tot

    # run 1: base store (16 patterns, seed 0 -- test_demo2.py's freeze)
    base = make_store(0, 16)
    r = rate(base, 8)
    check("demo3-base", r == 1.0, f"base recall {r:.2f} @8 flips")
    # EDIT A: drop to 8 (same listing, halved store). Predict 100%:
    # fewer competitors can only widen margins (stated mechanism).
    r = rate(base[:8], 8)
    check("demo3-editA", r == 1.0,
          f"halved store recall {r:.2f} @8 (preview: stays 100%)")
    # EDIT B: fresh 16 (seed 1, disjoint content). Predict 100% @<=8 by
    # demo-2 capacity reasoning (correct leads 2*(32-8)=48 vs ~21 spread).
    fresh = make_store(1, 16)
    r = rate(fresh, 8)
    check("demo3-editB", r == 1.0,
          f"fresh store recall {r:.2f} @8 (preview: 100% by capacity)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
