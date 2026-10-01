"""T-transform gates (v1.5/v1.4 follow-through): full-range attention ops.

TSHIFT (row-max shift) + SOFTMAX_WIDE (wide bridge + existing fixed path):
parity on WIDE-range scores (+-500, where legacy saturates), TSHIFT row-max
property, vendor 0-diff vs the phi-core branch (SKIPPED with a note when
the checkout sits on main -- the branch is a proposal, not a dependency),
in-listing use, legacy SOFTMAX untouched (its pinned saturating behavior
still gated in test_asm.py -- the contract split, both sides green).
Usage: python3 tests/test_tshift.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain import wide as W

FAIL = []
BAR_DB = 40.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def main():
    rng = np.random.default_rng(0)
    t = S.encode(rng.uniform(-500, 500, (4, 32)))
    # vendor 0-diff vs the branch implementation (proposal, not dependency)
    try:
        from phi_core import numpy_ops as N
        N.to_fixed_wide
        a = W.to_fixed_wide(t[0], t[1], t[2], S.BIAS)
        b = N.to_fixed_wide(t[0], t[1], t[2], S.BIAS)
        check("tshift-vendor-0diff", bool((a == b).all()),
              "vendored copy bit-exact vs branch (delete on merge)")
    except (ImportError, AttributeError) as e:
        print(f"tshift-vendor-0diff: SKIP (checkout on main: {e})")
    # TSHIFT row-max property: every row peaks at ~0 (shifted domain)
    ts = REGISTRY["TSHIFT"][0]([t], {}, {})
    v = dec(ts)
    check("tshift-rowmax", bool((np.abs(v.max(-1)) < 0.05).all()),
          f"row maxima {np.round(v.max(-1), 4)} (shifted to ~0)")
    # SOFTMAX_WIDE parity vs float on wide-range scores (peak=1.0 basis)
    sw = REGISTRY["SOFTMAX_WIDE"][0]([t], {}, {})
    pv = dec(sw)
    x = dec(t)
    e = np.exp(x - x.max(-1, keepdims=True))
    ref = e / e.sum(-1, keepdims=True)
    mse = float(np.mean((pv - ref) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("softmax-wide-parity", d >= BAR_DB,
          f"{d:.1f}dB vs float on +-500 scores (legacy saturates here)")
    # legacy untouched: in-contract inputs agree bit-exactly (contract split)
    tc = S.encode(rng.uniform(-0.9, 0.9, (4, 32)))
    g_old = REGISTRY["SOFTMAX"][0]([tc], {}, {})
    g_new = REGISTRY["SOFTMAX_WIDE"][0]([tc], {}, {})
    check("softmax-split-clean", all(
        bool((g_old[k] == g_new[k]).all()) for k in (0, 1, 2)),
        "in-contract: WIDE == legacy bit-exact (no behavior change)")
    # in-listing use
    try:
        gl = ASM.run_text("IN x AS T:SEQ\nS = TSHIFT(x)\nOUT = SOFTMAX_WIDE(S)\n",
                          REGISTRY, {"x": t}, sigs=SIGS)
        gd = ASM.run_text("IN x AS T:SEQ\nOUT = SOFTMAX_WIDE(x)\n",
                          REGISTRY, {"x": t}, sigs=SIGS)
        check("softmax-wide-listing", all(
            bool((gl["OUT"][k] == gd["OUT"][k]).all()) for k in (0, 1, 2)),
            "TSHIFT+WIDE == WIDE-direct (shift is softmax-invariant)")
    except ASM.AsmError as e:
        check("softmax-wide-listing", False, str(e)[:70])
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
