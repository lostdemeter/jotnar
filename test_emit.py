"""Substrate matrix (v1.2 gate 3): listings x substrates, every cell gated.

flagship row (full): numpy vs C vs CUDA bit-exact YENH/u8 (the C driver is
new this round; CUDA landed earlier; numpy is the reference). xf_block row:
numpy torch-parity (standing gate, referenced not duplicated) + C/CUDA
loud-refusal cells (no lowering yet -- the refusal IS the gate: it proves
no silent fallback and flips to a number when lowerings land). Bogus
substrate/program names refuse loud. Dispatch is proven by run_log (the
binary path + returncode per cell -- agreement alone would also match a
silent fallback).
Usage: python3 test_emit.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import substrate as SUB

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def synth_y():
    gx, gy = np.meshgrid(np.linspace(0.05, 1.0, 16), np.linspace(0.05, 1.0, 16))
    y = (((gx + gy) / 2)).astype(np.float64)
    y[8, :] = 0.95
    y[0, 0] = 1.0
    return np.ascontiguousarray(y)


def triples_equal(a, b):
    return bool((a[0] == b[0]).all() and (a[1] == b[1]).all()
                and (a[2] == b[2]).all())


def main():
    from chain.holo_phi import _load_scales
    ma, mc = _load_scales()
    y = synth_y()
    SUB.run_log.clear()
    ref = SUB.run("flagship", y, "numpy", ma, mc)
    got_c = SUB.run("flagship", y, "c", ma, mc)
    check("emit-flagship-c", triples_equal(ref, got_c),
          "numpy vs C bit-exact YENH")
    n_c = [e for e in SUB.run_log
           if e["program"] == "flagship" and e["substrate"] == "c"]
    check("emit-dispatched-c", len(n_c) == 1 and n_c[0]["returncode"] == 0
          and n_c[0]["binary"] is not None and "flagship_c_exchange" in n_c[0]["binary"],
          f"dispatch proven ({n_c[0]['binary'] if n_c else 'MISSING'})")
    try:
        got_cuda = SUB.run("flagship", y, "cuda", ma, mc)
        check("emit-flagship-cuda", triples_equal(ref, got_cuda),
              "numpy vs CUDA bit-exact YENH")
    except SUB.EmissionError as e:
        if "no cuda binary" in str(e):
            print(f"emit-flagship-cuda: SKIP ({e})")
        else:
            check("emit-flagship-cuda", False, str(e)[:100])
    # xf_block row: numpy standing cell (torch parity, small and fast here
    # on the Gate-1 fixture) + loud refusals where no lowering exists.
    import test_xf_block as XB
    xf, posf, wf, r1f, r2f = XB.fixture()
    tref, _ = XB.torch_block(xf, posf, wf, r1f, r2f)
    feeds = XB.run_listing(xf, posf, wf, r1f, r2f)
    gm = XB.dec(feeds["OUT"])
    dm = float(np.mean((gm - tref) ** 2))
    check("emit-xf-numpy", 10 * np.log10(1.0 / dm) >= 40.0,
          f"{10 * np.log10(1.0 / dm):.1f}dB standing cell")
    for sub in ("c", "cuda"):
        try:
            SUB.run("xf_block", None, sub)
            check(f"emit-xf-{sub}-refusal", False, "ran without lowering?!")
        except SUB.EmissionError as e:
            check(f"emit-xf-{sub}-refusal", "no %s lowering" % sub in str(e),
                  f"loud, names cell ({str(e)[:60]})")
    for prog, sub in (("nope", "numpy"), ("flagship", "tpu")):
        try:
            SUB.run(prog, y, sub)
            check(f"emit-bad-{prog}-{sub}", False, "accepted?!")
        except SUB.EmissionError as e:
            check(f"emit-bad-{prog}-{sub}", True,
                  f"refuses loud ({str(e)[:60]})")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
