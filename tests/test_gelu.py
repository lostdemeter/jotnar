"""GELU audit gate (pattern A2): variant audit killed the build.

The teacher's DiT MLPs use diffusers ApproximateGELU =
x*sigmoid(1.702*x) — NEITHER erf NOR tanh. Legs (CPU, torch,
seconds): erf-vs-sig1702 and tanh-vs-sig1702 measure BELOW 40dB
in the operating range (card falsifier fires — no erf/tanh
patterns built); the sigmoid-1.702 form composes EXACTLY from
tabled ops (BETA const + MUL + SILU); the README GELU row stays
parked and no GELU emission exists (tripwire).
Usage: python3 tests/test_gelu.py
"""
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import torch
    x = torch.linspace(-4, 4, 200001, dtype=torch.float64)
    erf_g = 0.5 * x * (1 + torch.special.erf(x / 2 ** 0.5))
    tanh_g = 0.5 * x * (1 + torch.tanh((2 / 3.141592653589793) ** 0.5
                                        * (x + 0.044715 * x ** 3)))
    sig1702 = x * torch.sigmoid(1.702 * x)

    def db(a, b):
        import math
        mse = float(((a - b) ** 2).mean())
        return 10 * math.log10(1.0 / max(mse, 1e-300))

    d_e, d_t = db(erf_g, sig1702), db(tanh_g, sig1702)
    check("gelu-erf-killed", d_e < 40,
          f"erf-vs-sig1702 {d_e:.1f}dB in [-4,4] (falsifier fires)")
    check("gelu-tanh-killed", d_t < 40,
          f"tanh-vs-sig1702 {d_t:.1f}dB in [-4,4] (falsifier fires)")

    # -- teacher form composes from tabled ops (310dB, ulp order) --------
    import math
    c1, c2 = 1.702, 1.0 / 1.702
    composed = torch.nn.functional.silu(x * c1) * c2
    mse = float(((composed - sig1702) ** 2).mean())
    cdb = 10 * math.log10(1.0 / max(mse, 1e-300))
    check("gelu-composed-exact", cdb >= 40,
          f"SILU(MUL(x,1.702))/1.702 vs x*sigmoid(1.702x): {cdb:.1f}dB")

    # -- row stays parked, no emission exists (tripwire) -------------------
    line = next((ln for ln in open(os.path.join(ROOT, "README.md"))
                 if re.match(r"^\| GELU \|", ln)), "")
    cells = [c.strip() for c in line.split("|")[3:6]]
    check("gelu-row-parked", cells == ["\u2014", "\u2014", "\u2014"],
          "drilled, no workload (still true)")
    cu = open(os.path.join(ROOT, "chain/emit_cuda.py")).read()
    cc = open(os.path.join(ROOT, "chain/emit_c.py")).read()
    check("gelu-no-emission",
          "k_gelu" not in cu and 'mn == "GELU"' not in cc,
          "no GELU kernel/case on either backend")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
