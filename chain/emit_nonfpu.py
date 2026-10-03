"""Non-FPU backend v0.1: integer-only emission (no float, no libm).

Subclasses the C backend but admits ONLY the exact-integer subgraph:
T-kind GATHER/ARGMAX/TRANSPOSE/SLICE/CONCAT/SELECT (+ I streams).
Any float (F streams, float patterns) is refused loud -- the FPU does
not exist on this target, so silence would be wrong code, not slow code.

Proof of "no FPU" (gated in tests, not claimed):
  1. float_trap(): regex over emitted source -- no float/double types,
     no libm calls (borrowed from phi-core's c_core `trap` doctrine);
  2. links without -lm;
  3. bit-exact agreement vs the lattice interpreter (integers only).

Arithmetic (MATMUL/RMSNORM/SOFTMAX/...) needs the fixed-point core
port (bridge tables + tmul + binop + matmul_int in integer C) -- the
next milestone, explicitly not this cut. Moves/selects/gather/argmax
are exact today and already useful (routing, reindexing, retrieval
paths run with no FPU at all).
"""
import re

from chain.asm import AsmError
from chain.emit_c import Backend, CBackend, NoPattern

INTEGER_MN = {"GATHER", "ARGMAX", "TRANSPOSE", "SLICE", "CONCAT", "SELECT"}

TRAP_RES = (
    re.compile(r"\b(float|double)\b"),
    re.compile(r"\b(sqrt|exp|sin|cos|tan|pow|fmax|fmin|fabs|round)\s*\("),
    re.compile(r"#include\s*<math\.h>"),
)


def float_trap(source):
    """Raise if the emitted source could touch an FPU. Comments are
    stripped first so the gate checks code, not prose."""
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"//.*", "", code)
    for rx in TRAP_RES:
        m = rx.search(code)
        if m:
            raise AsmError(f"nonfpu float-trap: {m.group(0)!r} -- "
                           f"FPU use in integer-only emission")


class NonFPUBackend(CBackend):
    """Integer-only C. Same scaffolding as CBackend; restricted set."""

    name = "nonfpu"
    emits = True
    DEFAULT_CFLAGS = ("-O2", "-std=c99", "-Wall")
    DEFAULT_LIBS: tuple = ()

    def prologue(self, ctx):
        L = ["#include <stdint.h>", "#include <stdio.h>",
             "#include <stdlib.h>", "#include <string.h>", ""]
        for k, v in ctx["dims"].items():
            L.append(f"#define DIM_{k} {v}L")
        return "\n".join(L) + "\n"

    def pattern(self, mn, outs, args, sig, ctx):
        if mn not in INTEGER_MN:
            raise NoPattern(
                f"nonfpu: no integer pattern for {mn} (fixed-point core "
                f"port is the next milestone, not this cut)")
        S = ctx["streams"]
        for a in args:
            if a in S and S[a][0] == "F":
                raise NoPattern(
                    f"nonfpu: float stream '{a}' refused (no FPU on this "
                    f"target -- decode nothing, quantize first)")
        return super().pattern(mn, outs, args, sig, ctx)
