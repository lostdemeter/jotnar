# Fields and local refinement

**Status:** builder prototype on `research/typed-scale-order`  
**Code:** `chain/fields.py`  
**Tests:** `tests/test_fields.py` (`5/5`)

This is the field-level companion to the scale-typed authoring layer and
`chain.composite`. It is deliberately a builder API, not a new backend opcode.

## Why this layer exists

The depth work kept rewriting the same shapes by hand:

```text
land an ordered stack of value/alpha/depth layers
make a bounded local residual
gate it with confidence/visibility
compose front-to-back without a SUM
```

Doing that directly in `Prog` means hand-built `GATHER3`/`RESHAPE3`/`MUL`/
`ADD` walls, and one missing scale or one `SUM` can silently destroy a
visibility boundary. `chain/fields.py` makes those operations explicit:

```python
Field(value, alpha, row, col, shape, provenance, depth)
LayerField(field, depth)
local_stencil(qp, field, taps)
refine(qp, field, residual, gate)
local_refine(qp, field, taps, gate)
composite_fields(qp, layers, zero, one, background)
```

## Contract

- `value`, `alpha` are 3D same-scale `QValue`s; `row`, `col` are `I` streams.
- `alpha=0` is an explicit hole/unknown. It is not a zero color and it is
  not a silent inpaint.
- `Field.unknown`, `Field.background`, and `Field.label` are optional
  explicit metadata; a refinement preserves them. `label` is an integer
  surface/entity stream.
- `Field.valid` and `Field.boundary_distance` are optional explicit
  sampling metadata. `valid` is a validity mask; `boundary_distance` is the
  distance-to-texture/silhouette boundary. `liminality(field)` is the
  read-only accessor for that distance.
- `sample_if` gathers a source field through a caller-supplied validity gate.
  Invalid samples have explicit `alpha=0`; they are never silently blended.
  Source validity and the gate are both enforced.
- `refine` changes value but **preserves alpha/order, valid, and
  boundary_distance**. A refinement cannot turn an unknown pixel into known
  without a deliberate alpha edit.
- `composite_fields` always uses ordered front-to-back composition. There is
  no fallback to `sum_composite`.
- Resolution is part of `Field.shape`; resolution is not inferred from a
  stream.
- The local stencil must be bounded by a declared `(dy, dx, weight)` list and
  the field must be padded so every tap is in-bounds.

## Current limits

- `local_stencil` supports uniform `+1` taps and an optional same-shape
  weight field for arbitrary weights. Scalar `-1` taps fail loud: the current
  `BETA` lowering is a constant fill, not a negation. Scalar broadcast is not
  part of the ISA.
- The builder emits ordinary `GATHER3`/`RESHAPE3`/`MUL`/`ADD`/`SUB` (and
  `BETA` only where a caller explicitly requests it);
  there is no compiled `LOCAL_STENCIL` or `COMPOSITE` opcode yet.
- A field operation is still a dense grid program. Local support is expressed
  by the stencil, not by a sparse kernel.
- `refine` is one pass. The stored field may be padded; the refined field is
  the unpadded center block. Re-padding/cropping between repeated passes is
  the caller's responsibility in this prototype.

## Tests

```bash
cd /tmp/opencode/jotnar
PYTHONPATH=/home/balrog/Documents/OpenCode/phi-core \
  python3 -m unittest tests.test_fields -v
```

Result: `10/10`.

Covered:

- scale mismatch fails at `Field` construction;
- constant-field residual is zero;
- `gate=0` is an exact identity;
- `gate=1` applies the full residual;
- ordered field composition ignores insertion order and uses depth;
- validity/liminality metadata is retained through refinement;
- `sample_if` zeroes alpha and validity for a rejected sample;
- `sample_if` preserves a valid sample and gathered boundary distance.

## Jotnar improvements this points to

### P0 — first-class field and local-stencil authoring

Add the `Field`/`LayerField` objects to the public stdlib. A user should not
write `GATHER3`/`RESHAPE3` by hand for a local patch.

### P0 — compiled `LOCAL_STENCIL` / `FIELD_REFINE`

The builder currently unrolls every tap into ordinary ops. A compiled op with a
declared radius, border mode, and gate would make local refinement cheap and
keep the IR small.

```text
LOCAL_STENCIL(field, taps, border) -> residual
FIELD_REFINE(value, alpha, residual, gate) -> value
```

Requirements:

- exact `gate=0/1`;
- alpha unchanged;
- explicit border policy (`zero`, `replicate`, `reflect`);
- deterministic tap order;
- scale checked at construction.

### P0 — compiled `COMPOSITE`

`ordered_composite` is still a builder primitive. A real `COMPOSITE` opcode
would avoid unrolling a layer stack into a long `MUL/ADD/SUB` chain and would
make the order/visibility contract visible to the assembler.

### P1 — field metadata in the IR

Fields need canonical metadata:

```text
shape
scale / covariance
alpha semantics
order key
provenance
validity
border mode
```

Without that, a saved program is not reproducible and a field cannot be
validated at the assembler boundary.

### P1 — padding and validity

A local refinement should not require the caller to pre-pad every field. Add:

- `PAD`/`BORDER` op or builder helper;
- `valid` mask distinct from alpha;
- explicit unknown propagation.

### P1 — sparse/local support

The dense grid program is acceptable for a prototype. A production local
refinement pass should support local support bounds so cost scales with the
region being refined, not the full image.

## What not to do

- Do not add a `SUM` fallback for ordered visibility.
- Do not infer scale from a default.
- Do not let `refine` change alpha or order implicitly.
- Do not hide border behavior in a kernel.
- Do not call a block gate a full graph.

## Next in the SBS project

The ordered field should be built as:

```text
source RGB-D
  -> per-view field layers (value, alpha, depth)
  -> target warp
  -> local residual refinement, gated by confidence/visibility
  -> ordered_composite
```

This is the representation the point-cloud calibration was trying to reach:
a stable ordered surface, not a cloud of unsupported samples.


## Validity and liminality contract

`Field.valid` is distinct from `alpha`:

```text
alpha = 0   -> no output sample exists here
valid = 0   -> a source sample exists but is not valid for this target
```

`sample_if` is the standard gated resampling operation:

```python
out = sample_if(qp, source, target_row, target_col, gate, target_shape)
```

Contract:

- `gate` is the caller's validity predicate (for example source label equals
  target label);
- source `valid` and `gate` are both enforced;
- `out.alpha = source.alpha * gate * source.valid`;
- `out.valid` carries the same conjunction;
- `out.boundary_distance` is gathered from the source;
- confirmation that a sample is invalid is reported through alpha/valid, not
  through a color threshold.

`boundary_distance` is intentionally a field, not a hard-coded color spread.
A policy can compare it with the source footprint and mark liminal samples
unknown.

The checker-parity predicate used in the SBS lab is an application-level gate,
not a Jotnar-specific operation.
