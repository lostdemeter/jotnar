"""Far-control probe: does the Europe-steerer move far content?

Same L26/27 contrast implants as clean_screen, but controls are far:
Japan (Tokyo), Water (212-degree region), fibonacci (newline). If far
holds while Europe moves -> region addressing (coarse but real content
addressing). If far moves -> blunt instrument (steers everything).
Usage: python3 research/far_control.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH
from chain.qwen7b import load7b

FAR = {
    "Japan": "The capital of Japan is",
    "Water": "Water boils at",
    "Code": "def fibonacci(n):",
}


def main():
    _, tok = load7b()
    base = {}
    for name, pr in {**{"Germany": "The capital of Germany is",
                         "Italy": "The capital of Italy is"}, **FAR}.items():
        lg, _ = fwd(pr)
        base[name] = (int(lg.argmax()), tok.decode([int(lg.argmax())]))
    print("unsteered tops:", {k: v[1] for k, v in base.items()}, flush=True)
    tfr = [fwdH("The capital of France is"), fwdH("The capital of Germany is")]
    for L in (26, 27):
        d = tfr[0][0][L] - tfr[1][0][L]
        d /= np.linalg.norm(d)
        for a in (0.5, 1.0):
            row = []
            for name, pr in {"Germany": "The capital of Germany is",
                             "Italy": "The capital of Italy is",
                             **FAR}.items():
                lg, _ = fwd(pr, steer=(L, d, a))
                top = int(lg.argmax())
                moved = "MOVED" if top != base[name][0] else "hold"
                row.append(f"{name}={tok.decode([top])[:12]!r}:{moved}")
            print(f"L={L} a={a}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
