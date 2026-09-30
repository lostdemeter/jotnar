"""Weight-edit routes (v1.3 research): which interventions carry signal.

Exploration (2026-09-30, xf_block toy fixture, seed 0) tried four routes:
A. whole-matrix zero: V/O 36dB > MLP 43dB > Q/K 50dB (mechanism: scores
   small -> softmax near-uniform, V carries the signal).
B. channel zero: 9dB spread across wv channels (43-52dB); ch3 dead across
   3 held inputs with fixed weights (60-65dB) -> partly weight-property.
C. scale x2 == zero in dB (36.4 vs 36.43): linearity, algebra predicts it
   (|2O-O| == |O-0|); consistency check, not new info.
D. rank truncation: flat spectrum on random weights (0.37..0.03) -> no
   low-rank structure TO exploit; meaningful only on trained weights.
Gates pin the METHOD invariants (deterministic on the fixed fixture):
scale-linearity, channel-spread-exists, rank-monotonicity. The exploration
table prints as measured rows. Standard process lives in docs/LIBRARY_NOTES
#LIB-045: whole-matrix ordering -> channel sweep (fixed weights x N inputs)
-> rank spectrum -> candidate edit with predicted band + held-out confirm.
Usage: python3 test_edits.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
import test_xf_block as XB

FAIL = []
NAMES = ['wq', 'wk', 'wv', 'wo', 'wup', 'wgate', 'wdown']


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((np.ascontiguousarray(a, dtype=np.float64)
                         - np.ascontiguousarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "programs", "xf_block.asm")).read()
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "programs")
    xf, posf, wf, r1f, r2f = XB.fixture(seed=0)
    W0 = dict(zip(NAMES, wf))

    def pay_with(mods):
        W = dict(W0)
        W.update(mods)
        return {"x": XB.enc(xf), "pos": posf,
                "wq": XB.enc(W['wq']), "wk": XB.enc(W['wk']),
                "wv": XB.enc(W['wv']), "wo": XB.enc(W['wo']),
                "wup": XB.enc(W['wup']), "wgate": XB.enc(W['wgate']),
                "wdown": XB.enc(W['wdown']),
                "rms_w1": XB.enc(r1f), "rms_w2": XB.enc(r2f)}

    base = XB.dec(ASM.run_text(text, REGISTRY, pay_with({}), sigs=SIGS,
                               basedir=sdir)["OUT"])
    run = lambda mods: XB.dec(ASM.run_text(
        text, REGISTRY, pay_with(mods), sigs=SIGS, basedir=sdir)["OUT"])
    # A. whole-matrix ordering (measured rows)
    print("--- route A: whole-matrix zero (seed 0) ---")
    da = {}
    for n in NAMES:
        d = psnr(run({n: np.zeros_like(W0[n])}), base)
        da[n] = d
        print(f"    zero-{n}: {d:.1f}dB")
    check("edits-ordering", da['wv'] < da['wup'] < da['wq'],
          f"V {da['wv']:.0f} < MLP {da['wup']:.0f} < Q {da['wq']:.0f}dB")
    # B. channel spread on wv (fixed weights AND fixed input)
    print("--- route B: wv channel zero (seed 0) ---")
    db = {}
    for col in range(W0['wv'].shape[1]):
        m = W0['wv'].copy()
        m[:, col] = 0
        db[col] = psnr(run({'wv': m}), base)
    spread = max(db.values()) - min(db.values())
    print("   ", {k: round(v, 1) for k, v in db.items()})
    check("edits-channel-spread", spread > 3.0,
          f"spread {spread:.1f}dB (channels unequal -> editable signal)")
    # C. scale-linearity: double == kill (algebra: |2O-O| == |O-0|)
    d_kill = da['wo']
    d_dbl = psnr(run({'wo': W0['wo'] * 2.0}), base)
    check("edits-scale-linear", abs(d_kill - d_dbl) < 1.0,
          f"zero {d_kill:.2f} vs x2 {d_dbl:.2f}dB (linearity holds)")
    # D. rank monotonicity (flat spectrum -> weak route on random weights)
    U, s, Vt = np.linalg.svd(W0['wv'], full_matrices=False)
    dr = {}
    for k in (8, 4, 2):
        approx = (U[:, :k] * s[:k]) @ Vt[:k]
        dr[k] = psnr(run({'wv': approx}), base)
    print("   ", {f"rank{k}": round(v, 1) for k, v in dr.items()},
          f"svals[{s[0]:.2f}..{s[-1]:.2f}]")
    check("edits-rank-monotone", dr[8] >= dr[4] >= dr[2],
          "less rank never helps (method sanity)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
