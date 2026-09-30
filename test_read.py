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
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
