"""First-class ordered fields and local refinement for Jotnar (prototype).

This is the field-level companion to chain.composite:

  Field        value + alpha + order key + provenance
  LayerField   one field in an ordered stack
  local_taps   bounded GATHER3 stencil abstraction
  refine       value += gate * residual, preserving order/alpha
  composite    explicit front-to-back composition

The backend still sees ordinary GATHER3/RESHAPE3/MUL/ADD/SUB ops.  The point
is to make the common field/refinement program a typed, validated builder
call rather than a hand-written wall of gather indices.

Design rules:

  0. A field carries its scale and provenance. A scale mismatch fails at
     construction time.
  1. alpha=0 and alpha=1 are exact. Unknown/hole is explicit alpha=0, never
     a hidden inpaint.
  2. Order is explicit. A field stack is never silently reduced with SUM.
  3. Local refinement is bounded by a declared stencil and an explicit gate.
     It cannot move values across an alpha=0 boundary unless the caller asks.
  4. Resolution is part of the field type. A 518x742 field is not a 742x518
     field.
"""
from __future__ import annotations

from dataclasses import dataclass

from chain.composite import Layer, ordered_composite
from chain.qtypes import QProg, QScale, QValue


@dataclass(frozen=True)
class Field:
    """A padded, single-scale, ordered-capable image/vector field.

    value/alpha are 3D QValues (H,W,C); row/col are 1D I arrays of length
    H*W.  The field is padded by the caller so every local tap is in-bounds.
    alpha is independent of value: alpha=0 is a hole/unknown, not a zero
    color.
    """
    value: QValue
    alpha: QValue
    row: QValue
    col: QValue
    shape: tuple[int, int, int]
    provenance: str = ""
    depth: float | None = None
    unknown: QValue | None = None
    background: QValue | None = None
    label: QValue | None = None
    label_namespace: str | None = None
    valid: QValue | None = None
    boundary_distance: QValue | None = None

    def __post_init__(self):
        if not isinstance(self.value, QValue):
            raise TypeError("Field.value must be a QValue")
        if not isinstance(self.alpha, QValue):
            raise TypeError("Field.alpha must be a QValue")
        for n, v in (("row", self.row), ("col", self.col)):
            if not isinstance(v, QValue):
                raise TypeError(f"Field.{n} must be a QValue")
            if v.kind != "I":
                raise TypeError(f"Field.{n} must be kind 'I'")
        if self.value.scale is None or self.alpha.scale is None:
            raise ValueError("Field value/alpha must carry a scale")
        if int(self.value.scale.m) != int(self.alpha.scale.m):
            raise ValueError(
                f"Field value/alpha scale mismatch: {self.value.scale.label()} "
                f"vs {self.alpha.scale.label()}")
        h, w, c = self.shape
        if min(h, w, c) <= 0:
            raise ValueError(f"Field shape must be positive, got {self.shape}")
        for n, q in (("unknown", self.unknown),
                     ("background", self.background),
                     ("valid", self.valid),
                     ("boundary_distance", self.boundary_distance)):
            if q is not None:
                if not isinstance(q, QValue):
                    raise TypeError(f"Field.{n} must be a QValue")
                if q.scale is None or int(q.scale.m) != int(self.value.scale.m):
                    raise ValueError(
                        f"Field.{n} scale must match value "
                        f"{self.value.scale.label()}")
        if self.valid is not None and self.valid.kind not in ("T", "I"):
            raise TypeError("Field.valid must be kind 'T' or 'I'")
        if self.boundary_distance is not None and self.boundary_distance.kind not in ("T", "F"):
            raise TypeError("Field.boundary_distance must be numeric (T/F)")
        if self.label is not None:
            if not isinstance(self.label, QValue):
                raise TypeError("Field.label must be a QValue")
            if self.label.kind != "I":
                raise TypeError("Field.label must be kind 'I'")
            if not self.label_namespace:
                raise ValueError(
                    "Field.label requires label_namespace (canonical id "
                    "namespace, e.g. 'hypersim.semantic' or 'scene.instance')")
        if self.provenance == "":
            object.__setattr__(self, "provenance",
                               self.value.provenance)


def field_shape(field: Field):
    return field.shape


def liminality(field: Field) -> QValue | None:
    """Distance-to-boundary field, if supplied by the producer.

    Liminality is metadata, not a color threshold.  A sampling policy can
    compare it with the source footprint and mark a target sample unknown.
    """
    return field.boundary_distance


def _gather3(qp: QProg, q: QValue, row: QValue, col: QValue,
             shape: tuple[int, int, int], name: str) -> QValue:
    h, w, c = shape
    g = qp.op("GATHER3", q, row, col, out=name, scale=q.scale)
    return qp.op("RESHAPE3", g, h, w, c, out=name + "3", scale=q.scale)


def sample_if(qp: QProg, source: Field, row: QValue, col: QValue,
              gate: QValue, shape: tuple[int, int, int],
              depth: float | None = None,
              out: str = "FIELD_SAMPLE") -> Field:
    """Gated source sampling with explicit validity and unknown propagation.

    ``gate`` is a 0/1 validity predicate supplied by the caller (for example
    source label == target label, or checker parity equality).  A gathered
    sample is valid only where both the source validity (if present) and the
    gate are true.  The returned field's alpha is zero where the gate fails;
    this is explicit unknown, never a silent blend.

    The source must be padded so ``row``/``col`` address every tap.  The
    returned field addresses the target grid with the same ``row``/``col``.
    """
    if not isinstance(source, Field):
        raise TypeError("sample_if: source must be Field")
    if row.kind != "I" or col.kind != "I":
        raise TypeError("sample_if: row/col must be kind 'I'")
    h, w, c = shape
    if min(h, w, c) <= 0:
        raise ValueError("sample_if: shape must be positive")
    if gate.scale is None or int(gate.scale.m) != int(source.value.scale.m):
        raise ValueError("sample_if: gate scale must match source value")
    value = _gather3(qp, source.value, row, col, shape, out + "_V")
    alpha = _gather3(qp, source.alpha, row, col, shape, out + "_A")
    if source.valid is not None:
        source_valid = _gather3(qp, source.valid, row, col, shape,
                                out + "_SV")
        valid = qp.op("MUL", source_valid, gate, out=out + "_VALID",
                      scale=value.scale)
    else:
        source_valid = None
        valid = qp.op("MUL", gate, gate, out=out + "_VALID",
                      scale=value.scale)
    sampled_alpha = qp.op("MUL", alpha, gate, out=out + "_ALPHA",
                          scale=alpha.scale)
    if source_valid is not None:
        sampled_alpha = qp.op("MUL", sampled_alpha, source_valid,
                              out=out + "_ALPHA_V", scale=alpha.scale)
    boundary = None
    if source.boundary_distance is not None:
        boundary = _gather3(qp, source.boundary_distance, row, col, shape,
                            out + "_LIM")
    return Field(value, sampled_alpha, row, col, shape,
                 provenance=f"{source.provenance}:sample_if",
                 depth=source.depth if depth is None else depth,
                 unknown=source.unknown, background=source.background,
                 label=source.label, label_namespace=source.label_namespace,
                 valid=valid, boundary_distance=boundary)


def local_stencil(qp: QProg, field: Field,
                  taps: list[tuple[int, int, float]],
                  out: str = "FIELD_RESIDUAL",
                  weight: QValue | None = None) -> QValue:
    """Bounded local residual: sum(w * field(p + tap)) - field(p).

    The stencil is declared as (dy, dx, weight) triples.  It lowers to
    GATHER3 + RESHAPE3 + MUL/ADD/SUB.  The field must be padded and row/col
    must address every output pixel.  No implicit full-grid operation is
    introduced.
    """
    h, w, c = field.shape
    if not taps:
        raise ValueError("local_stencil: taps must not be empty")
    acc = None
    for dy, dx, wt in taps:
        r = qp.op("IADD", field.row, int(dy), kind="I")
        col = qp.op("IADD", field.col, int(dx), kind="I")
        g = qp.op("GATHER3", field.value, r, col,
                  scale=field.value.scale)
        g3 = qp.op("RESHAPE3", g, h, w, c,
                   scale=field.value.scale)
        ww = float(wt)
        if ww == 1.0:
            pass
        elif ww == -1.0:
            g3 = qp.op("BETA", g3, scale=field.value.scale)
        elif ww != 0.0:
            raise ValueError(
                "local_stencil supports uniform +/-1 taps, or pass a "
                "same-shape weight field for arbitrary weights")
        if weight is not None:
            g3 = qp.op("MUL", g3, weight, scale=field.value.scale)
        acc = g3 if acc is None else qp.op(
            "ADD", acc, g3, scale=field.value.scale)
    center = qp.op("GATHER3", field.value, field.row, field.col,
                   scale=field.value.scale)
    center3 = qp.op("RESHAPE3", center, h, w, c,
                    scale=field.value.scale)
    return qp.op("SUB", acc, center3, out=out,
                 scale=field.value.scale)


def refine(qp: QProg, field: Field, residual: QValue, gate: QValue,
           out: str = "FIELD_REFINED", provenance: str = "") -> Field:
    """field.value += gate * residual, with alpha/order unchanged.

    gate=0 is an exact identity; gate=1 applies the full residual.  The
    caller is responsible for deriving gate from confidence/visibility, not
    from a target image.  alpha is intentionally preserved: refinement does
    not silently turn an unknown pixel into a known one.
    """
    if not isinstance(field, Field):
        raise TypeError("refine: field must be Field")
    if not isinstance(residual, QValue) or not isinstance(gate, QValue):
        raise TypeError("refine: residual/gate must be QValue")
    for n, q in (("residual", residual), ("gate", gate)):
        if q.scale is None or int(q.scale.m) != int(field.value.scale.m):
            raise ValueError(
                f"refine.{n} scale must match field "
                f"{field.value.scale.label()}")
    h, w, c = field.shape
    # The stored field may be padded; the refined field is the unpadded
    # center block.  alpha is cropped with the same address stream so a
    # refinement can never change visibility.
    vc = qp.op("GATHER3", field.value, field.row, field.col,
               scale=field.value.scale)
    vc3 = qp.op("RESHAPE3", vc, h, w, c, scale=field.value.scale)
    ac = qp.op("GATHER3", field.alpha, field.row, field.col,
               scale=field.alpha.scale)
    ac3 = qp.op("RESHAPE3", ac, h, w, c, scale=field.alpha.scale)
    gated = qp.op("MUL", gate, residual, scale=field.value.scale)
    value = qp.op("ADD", vc3, gated, out=out,
                  scale=field.value.scale)
    return Field(value, ac3, field.row, field.col, field.shape,
                 provenance=provenance or f"{field.provenance}:refine",
                 depth=field.depth, unknown=field.unknown,
                 background=field.background, label=field.label,
                 label_namespace=field.label_namespace,
                 valid=field.valid, boundary_distance=field.boundary_distance)


def local_refine(qp: QProg, field: Field,
                 taps: list[tuple[int, int, float]], gate: QValue,
                 out: str = "FIELD_REFINED",
                 weight: QValue | None = None) -> Field:
    """One bounded local residual refinement."""
    residual = local_stencil(qp, field, taps,
                             out=out + "_RESIDUAL", weight=weight)
    return refine(qp, field, residual, gate, out=out)


@dataclass(frozen=True)
class LayerField:
    """One field in a front-to-back stack."""
    field: Field
    depth: float


def composite_fields(qp: QProg, layers: list[LayerField],
                     zero: QValue, one: QValue, background: QValue,
                     out: str = "FIELD_COMPOSITE") -> QValue:
    """Explicit ordered composition; never a sum."""
    if not isinstance(layers, list) or not layers:
        raise ValueError("composite_fields: layers must be a non-empty list")
    vals = []
    for i, lf in enumerate(layers):
        if not isinstance(lf, LayerField):
            raise TypeError(f"layers[{i}] must be LayerField")
        vals.append(Layer(lf.field.value, lf.field.alpha, float(lf.depth)))
    return ordered_composite(qp, vals, zero, one, background, out=out)


def layer_stack(fields: list[tuple[Field, float]]) -> list[LayerField]:
    """Small convenience constructor with explicit depth keys."""
    return [LayerField(f, float(d)) for f, d in fields]
