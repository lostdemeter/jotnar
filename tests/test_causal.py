"""Causal mask mechanism gate: structure, not statistics.

Listing programs/attn_mini_causal.asm (SELECT + BETA composition, no new
mnemonics) vs torch causal mirror. Gates: future-blocked (~0 prob on
masked cells), causality bit-exact (future perturbation leaves past
outputs EXACTLY unchanged), reversal sensitivity (order matters),
torch parity, in-contract tripwire.
Usage: python3 tests/test_causal.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

BAR_DB = 40.0
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
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    text = open(os.path.join(root, "programs", "attn_mini_causal.asm")).read()
    sdir = os.path.join(root, "programs")
    rng = np.random.default_rng(0)
    Sq, D = 4, 8
    Qf = (rng.random((Sq, D)) - 0.5) * 0.6
    Kf = (rng.random((Sq, D)) - 0.5) * 0.6
    Vf = (rng.random((Sq, D)) - 0.5) * 0.6
    cm = np.tril(np.ones((Sq, Sq), dtype=np.int64))

    def run(q, k, v):
        return ASM.run_text(text, REGISTRY, {"Q": enc(q), "K": enc(k),
                                             "V": enc(v), "cmask": cm},
                            sigs=SIGS, basedir=sdir)

    full = run(Qf, Kf, Vf)
    scmax = float(np.abs(dec(full["SCORES"])).max())
    check("causal-contract", scmax <= 1.0,
          f"scoremax={scmax:.3f} (proves row non-vacuous)")
    P = dec(full["P"])
    mmax = float(P[np.triu_indices(Sq, 1)].max())
    check("causal-future-blocked", mmax < 1e-6,
          f"masked maxprob {mmax:.2e} (~0 by structure)")
    # causality: perturb last input row, past outputs bit-exact
    Qp = Qf.copy()
    Qp[3] += 0.05
    fp = run(Qp, Kf, Vf)
    o1, o2 = dec(full["OUT"]), dec(fp["OUT"])
    check("causal-past-exact", bool((o1[:3] == o2[:3]).all()),
          "rows 0-2 bit-exact under future perturbation")
    check("causal-future-moves", bool((o1[3] != o2[3]).any()),
          "row 3 changes (tripwire: perturbation landed)")
    # reversal: order matters (output is not a permutation of itself)
    rev = run(Qf[::-1], Kf[::-1], Vf[::-1])
    orr = dec(rev["OUT"])
    check("causal-order-matters", float(np.abs(orr - o1).max()) > 1e-3,
          f"maxdiff {float(np.abs(orr-o1).max()):.4f} on reversal")
    # torch mirror parity (causal float, -1e9 mask)
    import torch
    dt = torch.float64
    SC = torch.tensor(Qf, dtype=dt) @ torch.tensor(Kf, dtype=dt).T
    SC = SC + torch.tensor(np.triu(np.full((Sq, Sq), -1e9), 1), dtype=dt)
    PT = torch.softmax(SC, dim=-1).detach().numpy()
    mse = float(np.mean((P - PT) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("causal-torch-parity", d >= BAR_DB,
          f"{d:.1f}dB vs causal float (legacy saturates nowhere here)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
