"""Backend registry + target stubs (v0.2).

Targets: c (real, chain.emit_c.CBackend), lattice-ref (the interpreter
itself -- chain.asm.run, the agreement baseline), cuda + nonfpu (stubs:
registered, loud, with the extension recipe in the error).

Extension recipe (adding a pattern to a backend):
  1. add the shape rule in infer_shapes (or reuse: kinds/shapes are
     backend-neutral);
  2. add one pattern() branch emitting the target snippet;
  3. extend the agreement test with the new op's gate.
Adding a whole backend: subclass chain.emit_c.Backend, implement the
five methods, register here. Nothing else changes -- the driver,
shapes, and gates are shared.
"""
from chain.asm import AsmError
from chain.emit_c import Backend, CBackend, NoPattern


class CUDABackend(Backend):
    """CUDA target (stub). Intended patterns: one kernel (or cublas call)
    per opcode, prologue owns module/stream/kernel-preamble, epilogue owns
    device<->host marshalling. c_core/ in phi-core holds candidate kernels
    (ops.cu/conv.cu) -- inventory vs ISA mnemonics is the first real step.
    """

    name = "cuda"
    stub_note = ("cuda backend is a stub -- see chain/backends.py extension "
                 "recipe (c_core inventory first)")

    def pattern(self, mn, outs, args, sig, ctx):
        raise NoPattern(
            f"cuda: no pattern for {mn} yet -- see chain/backends.py "
            f"extension recipe (c_core inventory first)")


class NonFPUBackend(Backend):
    """Non-FPU target (stub). Intended patterns: fixed-point C (no libm,
    no float) replicating lattice fixed-point semantics bit-for-bit --
    the path that runs where no FPU exists. The T-triple patterns in
    CBackend (GATHER/ARGMAX-T) are its first residents; they move here
    (specialized to integer-only emission) when this backend goes real.
    """

    name = "nonfpu"
    stub_note = ("nonfpu backend is a stub -- T-triple residents "
                 "(GATHER/ARGMAX-T) migrate from CBackend first")

    def pattern(self, mn, outs, args, sig, ctx):
        raise NoPattern(
            f"nonfpu: no pattern for {mn} yet -- T-triple residents "
            f"(GATHER/ARGMAX-T) migrate from CBackend first")


REGISTRY = {
    "c": CBackend,
    "cuda": CUDABackend,
    "nonfpu": NonFPUBackend,
}


def get_backend(name):
    try:
        return REGISTRY[name]()
    except KeyError:
        raise NoPattern(
            f"no '{name}' backend (have: {', '.join(sorted(REGISTRY))}, "
            f"lattice-ref)") from None
