"""Demo 2 (v1.5): content-addressable memory from scratch (test).

16 bipolar patterns (dim 64, seeded -- the freeze), stored as keys+values;
noisy cues retrieve by similarity-argmax-gather. NO trained weights: the
only numbers are the patterns themselves + noise. Bands paper-first from
capacity reasoning (correct leads by 2*(32-f) vs competitor spread ~24):
exact 100% @0 flips, 100% @<=8 flips, >=80% @<=16 flips. 64 trials/level.
Usage: python3 test_demo2.py (fast, pure assembly + numpy)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
N_PAT, DIM, SEED = 16, 64, 0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(root, "programs", "assoc_mem.asm")).read()
    sdir = os.path.join(root, "programs")
    rng = np.random.default_rng(SEED)
    pats = rng.choice([-1.0, 1.0], size=(N_PAT, DIM))
    keys, values = pats.T.copy(), pats.copy()

    def recall(cue):
        feeds = ASM.run_text(text, REGISTRY,
                             {"cue": enc(cue.reshape(1, -1)),
                              "keys": enc(keys), "values": enc(values)},
                             sigs=SIGS, basedir=sdir)
        return np.asarray(feeds["OUT"][0]).reshape(-1) > 0

    def rate(nflip, trials=4):
        ok = tot = 0
        r = np.random.default_rng(1000 + nflip)
        for i in range(N_PAT):
            for _ in range(trials):
                cue = pats[i].copy()
                cue[r.choice(DIM, size=nflip, replace=False)] *= -1
                got = np.where(recall(cue), 1.0, -1.0)
                ok += bool((got == pats[i]).all())
                tot += 1
        return ok / tot

    r0 = rate(0)
    check("demo2-exact", r0 == 1.0, f"{r0:.2f} @0 flips (must be exact)")
    r8 = rate(8)
    check("demo2-robust8", r8 == 1.0, f"{r8:.2f} @<=8 flips")
    r16 = rate(16)
    check("demo2-robust16", r16 >= 0.80, f"{r16:.2f} @<=16 flips")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
