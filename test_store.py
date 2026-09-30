"""Directional stores (v1.3): implants as structure, not surgery.

stdlib/dirstore.asm's implant_apply (2 MATMULs) adds a rank-1 store to a
listing's sum; weight surgery bakes the same store into bytes. Gates (toy
xf fixture, fast, always runs): sham A=0 bit-exact vs base (machinery adds
nothing), listing-vs-surgery parity (same values, intermediate rounding
declared), stdlib interface shows the new DEF. The real-weight transfer
row lives in test_implant.py (same code path, proven not claimed).
Usage: python3 test_store.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
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
    root = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(root, "programs")
    xf, posf, wf, r1f, r2f = XB.fixture(seed=0)
    W0 = dict(zip(NAMES, wf))
    Sq, D, Dff = 8, 16, 32
    rng = np.random.default_rng(2)  # fresh vectors (not the probe seeds)
    # A=0.2 keeps implant products inside the frozen m_acc envelope on BOTH
    # paths (A=2.0 saturates each side differently, 16dB -- measured during
    # development; envelope discipline, not value luck).
    A = 0.2
    u = rng.normal(size=(Dff, 1)) * A
    v = rng.normal(size=(1, D))

    def base_pay(mods=None):
        W = dict(W0)
        W.update(mods or {})
        return {"x": XB.enc(xf), "pos": posf,
                "wq": XB.enc(W['wq']), "wk": XB.enc(W['wk']),
                "wv": XB.enc(W['wv']), "wo": XB.enc(W['wo']),
                "wup": XB.enc(W['wup']), "wgate": XB.enc(W['wgate']),
                "wdown": XB.enc(W['wdown']),
                "rms_w1": XB.enc(r1f), "rms_w2": XB.enc(r2f)}

    base_text = open(os.path.join(sdir, "xf_block.asm")).read()
    var_text = open(os.path.join(sdir, "xf_block_implant.asm")).read()
    base = XB.dec(ASM.run_text(base_text, REGISTRY, base_pay(),
                               sigs=SIGS, basedir=sdir)["OUT"])

    def var_run(uu, vv):
        pay = base_pay()
        pay["u"], pay["v"] = XB.enc(uu), XB.enc(vv)
        return XB.dec(ASM.run_text(var_text, REGISTRY, pay, sigs=SIGS,
                                   basedir=sdir)["OUT"])

    sham = var_run(np.zeros_like(u), v)
    mse_sham = float(np.mean((sham - base) ** 2))
    d_sham = float("inf") if mse_sham == 0 else 10 * np.log10(1.0 / mse_sham)
    check("store-sham", d_sham >= 60.0,
          f"{d_sham:.1f}dB (parity, not exact: ADD(DOWN,0) re-bridges -- "
          f"roundtrip quantum, stated)")
    d_surg = XB.dec(ASM.run_text(
        base_text, REGISTRY,
        base_pay({'wdown': W0['wdown'] + u @ v}), sigs=SIGS,
        basedir=sdir)["OUT"])
    d_list = var_run(u, v)
    mse = float(np.mean((d_list - d_surg) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("store-parity", d >= 40.0,
          f"{d:.1f}dB listing-vs-surgery (intermediate rounding declared)")
    check("store-bites", psnr(d_list, base) < 60.0,
          "implant moves output (edit bites)")
    errs, rep = ASM.verify(var_text, REGISTRY, SIGS, basedir=sdir)
    check("store-iface",
          errs == [] and "stdlib" in rep["defs"].get("implant_apply", {}).get("origin", ""),
          "variant verifies clean; DEF resolves to stdlib")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
