"""Modification demo (v1.3): listings are EDITABLE, not just runnable.

Variant programs/flagship_iso.asm replaces SPLAT_BLUR + BETA_V5 with
ISO_BLUR + scalar BETA (two-line structural edit). Gates: variant output
bit-exact vs the hand-written iso chain (the edit preserves meaning),
variant DIFFERS from the v5 flagship (45dB / 0.8 LSB -- the edit moves
values), variant census shows the new structure (AS produced by ISO_BLUR,
no COH stream). Modification with before/after numbers against the probe
baseline -- the first editable-structure proof.
Usage: python3 test_modify.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain import census as CS
from chain.asm_ops import REGISTRY, SIGS
from chain.holo_phi import enhance_image_int

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def load_rgb():
    root = os.path.dirname(os.path.abspath(__file__))
    ext = os.path.join(root, "..", "rife_reverse", "samples", "f_012.png")
    cand = ext if os.path.isfile(ext) else os.path.join(root, "samples", "input_example.png")
    return np.asarray(Image.open(cand).convert("RGB"))


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    rgb = load_rgb()
    var_text = open(os.path.join(root, "programs", "flagship_iso.asm")).read()
    got = ASM.run_text(var_text, REGISTRY, rgb, sigs=SIGS)["OUT"]
    lin = np.power(rgb.astype(np.float64) / 255.0, 2.2).astype(np.float32)
    ref, _, _ = enhance_image_int(lin, beta=0.5, blur="iso", ctrl=False)
    ref8 = np.clip(np.power(np.clip(ref, 0, 1), 1.0 / 2.2) * 255.0,
                   0, 255).astype(np.uint8)
    check("modify-iso-exact", bool((got == ref8).all()),
          "variant bit-exact vs hand-written iso chain")
    base = ASM.run_text(open(os.path.join(root, "programs",
                                          "holo_flagship.asm")).read(),
                        REGISTRY, rgb, sigs=SIGS)["OUT"]
    mse = float(np.mean((got.astype(np.float64) - base.astype(np.float64)) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    m = float(np.abs(got.astype(np.float64) - base.astype(np.float64)).mean())
    check("modify-differs", mse > 0, f"iso vs v5 {d:.2f}dB / {m:.2f} LSB")
    print(f"modify-measured: iso vs v5 {d:.2f}dB (edit moves values, reported)")
    rep = CS.census_text(var_text, REGISTRY, SIGS,
                         basedir=os.path.join(root, "programs"))
    check("modify-census",
          rep["streams"]["AS"]["producer"][0] == "ISO_BLUR"
          and "COH" not in rep["streams"],
          "census shows the new structure (AS from ISO_BLUR, no COH)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
