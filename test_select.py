"""Selection-side decomposition (v1.4 gate 2): S08's first instance.

Attention output is LINEAR in V at fixed P (P depends on QK only): per
(value-row) contributions via complement-ablation, exact by construction.
Gates on a minimal in-contract listing (programs/attn_mini.asm, S=4 D=8):
P-invariance under V masks (precondition, bit-exact), recomposition sum
== full output, top-attention ablation beats bottom- (ordering). S08 goes
PREDICTED-ish -> SINGLE here.
Usage: python3 test_select.py (fast, toy magnitudes)
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


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(root, "programs", "attn_mini.asm")).read()
    sdir = os.path.join(root, "programs")
    rng = np.random.default_rng(0)
    Sq, D = 4, 8
    Qf = (rng.random((Sq, D)) - 0.5) * 0.6
    Kf = (rng.random((Sq, D)) - 0.5) * 0.6
    Vf = (rng.random((Sq, D)) - 0.5) * 0.6
    pay = {"Q": enc(Qf), "K": enc(Kf), "V": enc(Vf)}
    full = ASM.run_text(text, REGISTRY, pay, sigs=SIGS, basedir=sdir)
    P0 = full["P"]
    scmax = float(np.abs(dec(full["SCORES"])).max())
    check("select-contract", scmax <= 1.0,
          f"scoremax={scmax:.3f} (softmax contract; proves row non-vacuous)")
    # 1. precondition: P invariant under V masks (bit-exact)
    masks = []
    for j in range(Sq):
        M = np.zeros((Sq, D))
        M[j] = Vf[j]
        f = ASM.run_text(text, REGISTRY,
                         {"Q": enc(Qf), "K": enc(Kf), "V": enc(M)},
                         sigs=SIGS, basedir=sdir)
        masks.append(f)
    same = all(bool((m["P"][k] == P0[k]).all()) for m in masks
               for k in (0, 1, 2))
    check("select-p-invariance", same,
          "P bit-exact under all V masks (precondition holds)")
    # 2. recomposition: sum of single-row outputs == full
    acc = sum(dec(m["OUT"]) for m in masks)
    ref = dec(full["OUT"])
    mse = float(np.mean((acc - ref) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("select-recompose", d >= 60.0,
          f"{d:.1f}dB sum-of-rows vs full (linearity)")
    # 3. ordering: killing the top-attention row moves output more than
    # killing the bottom one (gains mean something, stated weakly)
    Pv = dec(P0)
    top = [(int(np.argmax(Pv[i])), i) for i in range(Sq)]
    dtop, dbot = [], []
    for j, i in top:
        M = Vf.copy()
        M[j] = 0
        g = dec(ASM.run_text(text, REGISTRY,
                             {"Q": enc(Qf), "K": enc(Kf), "V": enc(M)},
                             sigs=SIGS, basedir=sdir)["OUT"])
        mse = float(np.mean((g - ref) ** 2))
        dtop.append(10 * np.log10(1.0 / mse) if mse > 0 else float("inf"))
    for i in range(Sq):
        j = int(np.argmin(Pv[i]))
        M = Vf.copy()
        M[j] = 0
        g = dec(ASM.run_text(text, REGISTRY,
                             {"Q": enc(Qf), "K": enc(Kf), "V": enc(M)},
                             sigs=SIGS, basedir=sdir)["OUT"])
        mse = float(np.mean((g - ref) ** 2))
        dbot.append(10 * np.log10(1.0 / mse) if mse > 0 else float("inf"))
    check("select-top-beats-bottom",
          float(np.mean(dtop)) < float(np.mean(dbot)),
          f"top-ablation {np.mean(dtop):.1f}dB < bottom {np.mean(dbot):.1f}dB "
          f"(thin margin by mechanism: flat attention ~= flat contributions)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
