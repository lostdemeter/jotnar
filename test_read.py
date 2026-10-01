"""Read-instrument gates (v1.3): table shape + query plumbing + content.

Runs a FULL 16-direction readout on the toy xf wdown (fast, no cache):
shape pinned, movers() matches brute-force argsort (plumbing), dead
shelves all above threshold and sorted (plumbing), and one content claim:
some token's effects spread >3dB (unequal token effects exist -- the
readout that justifies searching). The REAL readout (Qwen down_proj,
112 dirs) is the demonstration in docs/MODEL_READ.md, not duplicated here.
Usage: python3 test_read.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain import read as RD
from chain.asm_ops import REGISTRY, SIGS
import test_xf_block as XB

FAIL = []
NAMES = ['wq', 'wk', 'wv', 'wo', 'wup', 'wgate', 'wdown']


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(root, "programs")
    text = open(os.path.join(sdir, "xf_block.asm")).read()
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

    def run_fn(WdT):
        return XB.dec(ASM.run_text(text, REGISTRY, pay_with({"wdown": WdT}),
                                   sigs=SIGS, basedir=sdir)["OUT"])

    ro = RD.direction_readout(base, run_fn, W0["wdown"])
    check("read-shape", ro["tokdb"].shape == (16, 8) and len(ro["idx"]) == 16,
          f"16 dirs x 8 tokens {ro['tokdb'].shape}")
    t = 3
    man = np.argsort(ro["tokdb"][:, t])[:5]
    got = RD.movers(ro, t, k=5)
    check("read-movers", [i for i, _ in got] == [int(x) for x in man],
          f"query matches brute force {got}")
    dead = RD.dead_shelves(ro, 55.0)
    check("read-dead-plumbing",
          all(d > 55.0 for _, d, _ in dead)
          and [d for _, d, _ in dead] == sorted(
              [d for _, d, _ in dead], reverse=True),
          f"{len(dead)} shelves, all above thresh, sorted")
    spread = float(ro["tokdb"].max(-1).max() - ro["tokdb"].min(-1).min())
    check("read-content-spread", spread > 3.0,
          f"{spread:.1f}dB token-effect range (searching is justified)")
    sel = RD.selectivity(ro)
    check("read-selectivity", len(sel) == 16 and all(len(r) == 3 for r in sel),
          "per-direction (spread, argmin-token) triples")
    # shelf-map logic (synthetic readouts, no runs): intersection is dead-
    # everywhere, union dead-somewhere, grid mismatch fails loud.
    def _fake(dead, n=8):
        idx = np.arange(n)
        return {"idx": idx, "sval": np.ones(n),
                "gdb": np.array([60.0 if i in dead else 30.0 for i in idx]),
                "tokdb": np.full((n, 4), 40.0)}
    sm = RD.shelf_map([_fake({1, 3}), _fake({3, 5}), _fake({3})])
    check("read-shelf-map", sm["intersection"] == [3]
          and sm["union"] == [1, 3, 5]
          and sm["per_context"] == [[1, 3], [3, 5], [3]],
          f"inter {sm['intersection']} union {sm['union']}")
    try:
        RD.shelf_map([_fake({1}), _fake({1}, n=6)])
        check("read-shelf-grid", False, "accepted mismatched grids")
    except ValueError:
        check("read-shelf-grid", True, "grid mismatch fails loud")
    # factorization form (exact reals identity the B1 measurement relies on:
    # ablating s.u.vT moves output by s.(X.u)(+)v -- gated at 1e-12, NOT
    # bit-exact: BLAS reorders float sums (measured 9e-16, pure rounding).
    # Form pinned in-suite; B1's 0.998 lattice survival is measured).
    rng = np.random.default_rng(0)
    Xf = rng.normal(size=(5, 7))
    Wf = rng.normal(size=(7, 6))
    Uf, sf, Vtf = np.linalg.svd(Wf, full_matrices=False)
    i = 3
    d_direct = Xf @ (Wf - sf[i] * np.outer(Uf[:, i], Vtf[i]))
    d_factored = (Xf @ Wf) - sf[i] * np.outer(Xf @ Uf[:, i], Vtf[i])
    gap = float(np.abs(d_direct - d_factored).max())
    check("read-factor-form", gap < 1e-12,
          f"maxabsdiff {gap:.1e} (float rounding, not formula)")
    # predictor logic (synthetic, instant): calibrate on half the dirs,
    # predict the other half -- exact same code path as the real 0.2dB
    # demonstration (chain/read.py calibrate_C/predict_db).
    rng2 = np.random.default_rng(1)
    Xs = rng2.normal(size=(6, 9))
    Ws = rng2.normal(size=(9, 10))
    Us, ss, Vts = np.linalg.svd(Ws, full_matrices=False)
    Ys = Xs @ Ws
    cal, held = [0, 2, 4, 6, 8], [1, 3, 5, 7]
    gcal = []
    for i in cal:
        d = Ys - (Xs @ (Ws - ss[i] * np.outer(Us[:, i], Vts[i])))
        mse = float((d ** 2).mean())
        gcal.append(10 * np.log10(1.0 / mse))
    al = np.sqrt(((Xs @ Us[:, cal]) ** 2).mean(0))
    C = RD.calibrate_C(ss[cal], al, np.array(gcal))
    errs = []
    for i in held:
        a = float(np.sqrt(((Xs @ Us[:, i]) ** 2).mean()))
        d = Ys - (Xs @ (Ws - ss[i] * np.outer(Us[:, i], Vts[i])))
        mse = float((d ** 2).mean())
        errs.append(abs(RD.predict_db(ss[i], a, C) - 10 * np.log10(1.0 / mse)))
    check("read-predictor", max(errs) < 0.5,
          f"worst held-out err {max(errs):.2f}dB (logic exact, lattice adds ~0.2)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
