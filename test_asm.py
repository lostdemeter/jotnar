"""Assembly fidelity gates (stage 1): the listing IS the program.

test_asm proves the assembly level FAITHFUL: running programs/holo_flagship.asm
produces bit-exact output vs the hand-written chain path (same functions, but
the TEXT fully specifies the computation -- no hidden Python). Plus assembler
discipline gates (unknown mnemonic / arity / use-before-def fail at assemble
time, never mid-run). Stage 2 (generative proof) gets its own gate file.
Usage: python3 test_asm.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain.asm_ops import REGISTRY
from chain.holo_phi import enhance_image_int

FAIL = []
CAND = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "programs", "holo_flagship.asm")) as fh:
        text = fh.read()
    rgb = np.asarray(Image.open(CAND).convert("RGB"))
    feeds = ASM.run_text(text, REGISTRY, rgb)
    got = feeds["OUT"]

    # hand-written flagship path (demo.py --blur splat_soft --ctrl v5)
    lin = np.power(rgb.astype(np.float64) / 255.0, 2.2).astype(np.float32)
    ref, _, _ = enhance_image_int(lin, beta=0.5, blur="splat_soft", ctrl="v5")
    ref8 = np.clip(np.power(np.clip(ref, 0, 1), 1.0 / 2.2) * 255.0,
                   0, 255).astype(np.uint8)
    check("asm-faithful", bool((got == ref8).all()),
          "listing output bit-exact vs hand-written chain")

    # assembler discipline: all failures at assemble time
    for bad, tag in [
        ("IN x\nOUT = NOPE(x)\n", "unknown-mnemonic"),
        ("IN x\nY = LUMA(x, y)\n", "arity"),
        ("IN x\nY = LUMA(nope)\n", "use-before-def"),
        ("OUT = LUMA(x)\n", "missing-in"),
    ]:
        try:
            ASM.run_text(bad, REGISTRY, np.zeros((2, 2, 3), np.uint8))
            check(f"asm-{tag}", False, "assembled without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", True, f"fails loud ({str(e)[:60]})")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
