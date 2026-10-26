"""Scale-typed authoring layer for Jotnar programs (research prototype).

The assembler executes stream names and validates layouts; it does not
carry scale/covariance through the Python builder. This layer adds that
context *without changing the assembler*: every triple value is wrapped in a
QValue carrying its scale, covariance context, and provenance. QProg keeps
the assembler as the authority (it still emits ordinary text) but refuses to
construct a graph when scale-bearing values are mixed illegally or when a
triple output has no scale context.

This is intentionally an authoring layer, not a new backend. Existing Prog
callers remain unchanged.

Usage:
  qp = QProg("demo", m_acc=36118, m_cov=35048)
  a = qp.inp("A", provenance="rgb.codec")
  b = qp.inp("B", provenance="constant.one")
  d = qp.op("ADD", a, b, out="D")
  qp.run({...})
"""
from __future__ import annotations

from dataclasses import dataclass

from chain.asm import AsmError
from chain.builder import Prog


class ScaleError(AsmError):
    """A loud scale/type failure with construction context."""


@dataclass(frozen=True)
class QScale:
    """Scale context for a lattice value.

    m: value/covariance scale (m_cov)
    m_acc: optional accumulator scale for products
    context: "value", "acc", "dimensionless", or "mask"
    provenance: where this scale came from (encoder, constant, op, file)
    """
    m: int
    m_acc: int | None = None
    context: str = "value"
    provenance: str = ""

    def __post_init__(self):
        if not (0 <= int(self.m) < 65536):
            raise ScaleError(f"scale m out of range: {self.m}")
        if self.m_acc is not None and not (0 <= int(self.m_acc) < 65536):
            raise ScaleError(f"scale m_acc out of range: {self.m_acc}")

    def label(self):
        p = f"m={self.m}"
        if self.m_acc is not None:
            p += f",m_acc={self.m_acc}"
        if self.context:
            p += f",{self.context}"
        return p


@dataclass(frozen=True)
class QValue:
    name: str
    scale: QScale | None
    kind: str = "T"
    provenance: str = ""

    def label(self):
        return f"{self.name}[{self.kind}:{self.scale.label() if self.scale else 'UNSCALED'}]"


def decode_triple(t):
    """Decode a triples tuple exactly as the assembler summarizes it."""
    import numpy as np
    import phi_core.lattice as S
    return S.decode(np.ascontiguousarray(t[0]),
                    np.ascontiguousarray(t[1])) * (
        1 - np.ascontiguousarray(t[2]).astype(np.float64))


# Operators where a lattice value at two different scales cannot be combined
# silently. This is deliberately conservative: MUL on triples is scale-free
# in the current encoder, but requiring equal context catches the common
# graph mistake before the assembler sees it.
_SAME_SCALE = {
    "ADD", "SUB", "DIV", "MUL", "SELECT", "GT", "SQUARE",
    "CLIP", "SIGMOID", "GELU", "SILU",
}

_T_OUTPUT = {
    "ADD", "SUB", "MUL", "DIV", "SQUARE", "BETA", "TBETA",
    "BETA_V5", "SPLAT_BLUR", "ISO_BLUR", "GAUSS", "CONV", "INTERP",
    "WARP", "MIXDYAD", "RESCALE", "CLIP", "SIGMOID", "SILU",
    "SOFTMAX", "SOFTMAX_WIDE", "TSHIFT", "RMSNORM", "LAYERNORM",
    "GELU", "MATMUL", "BATCH_MATMUL", "GATHER", "GATHER3",
}


class QProg:
    """Strict wrapper around Prog with scale/covariance provenance."""

    def __init__(self, name="qprog", m_acc: int | None = None,
                 m_cov: int | None = None):
        self.p = Prog(name)
        self.m_acc = m_acc
        self.m_cov = m_cov
        self.values: dict[str, QValue] = {}
        if m_acc is not None:
            self.p.config("m_acc", int(m_acc))
        if m_cov is not None:
            self.p.config("m_cov", int(m_cov))

    @property
    def default_scale(self):
        if self.m_cov is None:
            return None
        return QScale(self.m_cov, self.m_acc, "value", "QProg.default")

    def config(self, k, v):
        self.p.config(k, v)
        return self

    def inp(self, name: str, scale: QScale | None = None, kind: str = "T",
            provenance: str = "") -> QValue:
        if scale is None and kind == "T":
            scale = self.default_scale
        if kind == "T" and scale is None:
            raise ScaleError(
                f"QProg.inp('{name}'): triple input has no scale context; "
                f"pass scale=QScale(...) or construct QProg(m_cov=...)")
        if scale is not None:
            if self.m_cov is None:
                self.m_cov = scale.m
                self.p.config("m_cov", int(scale.m))
            elif int(scale.m) != int(self.m_cov):
                raise ScaleError(
                    f"QProg.inp('{name}'): scale {scale.label()} conflicts "
                    f"with program m_cov={self.m_cov}")
        self.p.inp(name)
        q = QValue(name, scale, kind, provenance or f"IN({name})")
        self.values[name] = q
        return q

    def _resolve(self, a):
        if isinstance(a, QValue):
            return a.name, a
        return a, None

    def op(self, mn: str, *args, out: str | None = None,
           scale: QScale | None = None, kind: str | None = None,
           provenance: str = "") -> QValue:
        mn = str(mn).upper()
        names, qargs = [], []
        for a in args:
            n, q = self._resolve(a)
            names.append(n)
            if q is not None:
                qargs.append(q)

        if mn in _SAME_SCALE:
            scaled = [q for q in qargs if q.scale is not None]
            if scaled:
                base = scaled[0]
                for q in scaled[1:]:
                    if int(q.scale.m) != int(base.scale.m):
                        raise ScaleError(
                            f"line {len(self.p._lines) + 1} ({mn}): "
                            f"scale mismatch: '{base.name}' is "
                            f"{base.scale.label()} ({base.provenance}) but "
                            f"'{q.name}' is {q.scale.label()} "
                            f"({q.provenance})")

        # Output scale: explicit > first scale-bearing operand > default for T.
        if scale is None and qargs:
            scale = next((q.scale for q in qargs if q.scale is not None), None)
        if scale is None and (kind == "T" or (kind is None and mn in _T_OUTPUT)):
            scale = self.default_scale
        if scale is None and (kind == "T" or (kind is None and mn in _T_OUTPUT)):
            raise ScaleError(
                f"line {len(self.p._lines) + 1} ({mn}): triple output has no "
                f"scale context; pass scale=QScale(...) or set QProg(m_cov=...)")

        if out is None:
            out = self.p._fresh()
        if out in self.values:
            raise ScaleError(f"line {len(self.p._lines) + 1}: duplicate output "
                             f"'{out}' (would shadow typed value)")
        self.p.op(mn, *names, out=out)
        q = QValue(out, scale, kind or "T",
                   provenance or f"OP {mn} at line {len(self.p._lines)}")
        self.values[out] = q
        return q

    def rescale(self, value: QValue, m2: int, out: str | None = None) -> QValue:
        if value.scale is None:
            raise ScaleError(f"RESCALE('{value.name}'): source has no scale")
        return self.op("RESCALE", value, int(m2), out=out,
                       scale=QScale(int(m2), None, value.scale.context,
                                    f"rescale from {value.scale.label()}"))

    def run_checked(self, feeds, expect=None, atol=1e-3,
                    forbidden_phi=()):
        """Execute and fail loud on an invariant mismatch.

        expect: {stream: expected decoded value}
        forbidden_phi: stream names that must not decode to PHI (the
        historical 2+0/phi failure). The diagnostic carries the op, line,
        input summaries, scale/provenance, expected, and actual.
        """
        import numpy as np
        import phi_core.lattice as S
        trace = []
        out = self.run(feeds, trace=trace)
        records = {ln: rec for rec in trace for ln in rec.get("out", {})}
        for name, exp in (expect or {}).items():
            if name not in out:
                raise ScaleError(f"run_checked: output '{name}' missing")
            got = decode_triple(out[name])
            if not np.allclose(got, exp, atol=atol, rtol=0):
                rec = records.get(name, {})
                q = self.values.get(name)
                ins = rec.get("in", {})
                raise ScaleError(
                    f"line {rec.get('line', '?')} ({rec.get('op', '?')}): "
                    f"stream '{name}' decoded {np.asarray(got).tolist()}, "
                    f"expected {exp}; inputs={ins}; "
                    f"output_scale={q.scale.label() if q and q.scale else '?'}")
        for name in forbidden_phi:
            if name not in out:
                raise ScaleError(f"run_checked: output '{name}' missing")
            got = decode_triple(out[name])
            if np.any(np.isclose(got, S.PHI, atol=1e-3, rtol=0)):
                rec = records.get(name, {})
                q = self.values.get(name)
                raise ScaleError(
                    f"line {rec.get('line', '?')} ({rec.get('op', '?')}): "
                    f"stream '{name}' unexpectedly decoded as PHI "
                    f"({np.asarray(got).tolist()}); inputs={rec.get('in', {})}; "
                    f"output_scale={q.scale.label() if q and q.scale else '?'}")
        return out

    def text(self):
        return self.p.text()

    def verify(self):
        from chain.asm import verify as _verify
        from chain.asm_ops import REGISTRY, SIGS
        return _verify(self.text(), REGISTRY, SIGS)

    def run(self, feeds, trace=None):
        from chain.asm import run_text
        from chain.asm_ops import REGISTRY, SIGS
        return run_text(self.text(), REGISTRY, feeds, sigs=SIGS,
                        trace=trace)
