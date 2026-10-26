"""Ordered compositing as an explicit Jotnar builder primitive.

The distinction this module makes is semantic:

  ordered_composite(...)  # front-to-back alpha/visibility
  sum_composite(...)      # weighted average / visibility marginalized

This is the missing primitive from the splat/sum study. It is a builder
primitive (the assembler still sees ordinary MUL/ADD/SUB/DIV ops), not a new
backend opcode. The point is that callers cannot silently choose a sum when
they mean order/visibility.

Expected inputs are QValues from chain.qtypes (same m_cov). The caller must
supply explicit zero and one streams at that scale; no implicit float
identity is inserted.
"""
from __future__ import annotations

from dataclasses import dataclass

from chain.qtypes import QProg, QValue


@dataclass(frozen=True)
class Layer:
    value: QValue
    alpha: QValue
    depth: float


def _check(qp: QProg, layers, zero: QValue, one: QValue, background: QValue):
    names = ["zero", "one", "background"]
    vals = [zero, one, background]
    for n, v in zip(names, vals):
        if not isinstance(v, QValue):
            raise TypeError(f"{n} must be QValue, got {type(v).__name__}")
        if v.scale is None or v.scale.m != qp.m_cov:
            raise ValueError(f"{n} must carry program m_cov={qp.m_cov}")
    for i, layer in enumerate(layers):
        if not isinstance(layer, Layer):
            raise TypeError(f"layers[{i}] must be Layer")
        for n, v in (("value", layer.value), ("alpha", layer.alpha)):
            if not isinstance(v, QValue):
                raise TypeError(f"layers[{i}].{n} must be QValue")
            if v.scale is None or v.scale.m != qp.m_cov:
                raise ValueError(f"layers[{i}].{n} must carry "
                                 f"program m_cov={qp.m_cov}")


def ordered_composite(qp: QProg, layers: list[Layer], zero: QValue,
                      one: QValue, background: QValue,
                      out: str = "SBS_COMPOSITE") -> QValue:
    """Explicit front-to-back alpha compositing.

    D = sum_i T_i * alpha_i * value_i + T_N * background
    T_i = prod_{j<i} (1 - alpha_j)

    Layers are sorted by (depth, insertion order); ties are stable and
    deterministic. alpha=0 and alpha=1 are exact lattice values, so the
    endpoints do not depend on a soft kernel.
    """
    _check(qp, layers, zero, one, background)
    ordered = sorted(enumerate(layers), key=lambda t: (t[1].depth, t[0]))
    T = one
    D = zero
    for _, layer in ordered:
        a = qp.op("MUL", layer.alpha, layer.value)
        D = qp.op("ADD", D, qp.op("MUL", T, a))
        T = qp.op("MUL", T, qp.op("SUB", one, layer.alpha))
    D = qp.op("ADD", D, qp.op("MUL", T, background), out=out)
    return D


def sum_composite(qp: QProg, layers: list[Layer], zero: QValue,
                  one: QValue, background: QValue,
                  out: str = "SBS_SUM") -> QValue:
    """Visibility-marginalized weighted average.

    D = (sum_i alpha_i * value_i + background)
        / (sum_i alpha_i + 1)

    The background is included with explicit unit weight. This is a
    separate function so a graph cannot accidentally use it where order is
    required. It is a sum, not a hidden fallback for ordered_composite.
    """
    _check(qp, layers, zero, one, background)
    num = qp.op("MUL", one, background)
    den = one
    for layer in layers:
        num = qp.op("ADD", num, qp.op("MUL", layer.alpha, layer.value))
        den = qp.op("ADD", den, layer.alpha)
    return qp.op("DIV", num, den, out=out)
