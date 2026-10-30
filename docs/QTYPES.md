# Scale-typed authoring and ordered compositing

**Status:** merged on `main`; `chain.builder` authoring layer.

The assembler remains the authority: `QProg` emits ordinary Jotnar text and
never replaces `Prog`, `assemble`, or the backend gates. This layer adds
scale/type provenance during graph construction and makes a class of silent
mistakes loud before execution.

## Why

The depth work hit a real failure mode:

```text
2 + 0   (default m_cov)
→ decoded as phi
```

The default scale was plausible enough to survive until alpha/visibility
compositing, where a zero/one gate stopped being an identity. The problem was
not a missing opcode; it was an implicit scale contract.

The same work also showed that a sum and an ordered composite are different
semantic operations: a sum can erase visibility at a depth boundary.

## API

```python
from chain.qtypes import QProg, QScale
from chain.composite import Layer, ordered_composite

qp = QProg("demo", m_acc=36118, m_cov=35048)
a = qp.inp("A", provenance="encoder.rgb")
b = qp.inp("B", provenance="constant.zero")
d = qp.op("ADD", a, b, out="D")
qp.run(feeds)
```

`QScale` carries:

- `m` — value/covariance scale (`m_cov`);
- `m_acc` — optional accumulator scale;
- `context` — value / acc / dimensionless / mask;
- `provenance` — encoder, constant, or op line.

`QValue` carries the stream name, scale, kind, and provenance.

### Construction-time failures

`QProg.inp` rejects a triple without scale context. `QProg.op` rejects
same-scale operations whose scale-bearing operands disagree, and rejects a
triple output with no scale context. The error includes the op, line, stream,
both scale labels, and provenance.

### Checked execution

```python
qp.run_checked(
    feeds,
    expect={"D": 2.0},
    forbidden_phi=("D",),
)
```

A mismatch reports the line, op, input summaries, output scale, expected, and
actual decoded value. `forbidden_phi` is for streams where a `phi` decode is
never valid.

## Ordered compositing

```python
Layer(value, alpha, depth)
ordered_composite(qp, layers, zero, one, background)
sum_composite(qp, layers, zero, one, background)
```

The ordered primitive:

```text
D = sum_i T_i * alpha_i * value_i + T_N * background
T_i = prod_{j<i} (1 - alpha_j)
```

Layers sort by `(depth, insertion order)`; ties are deterministic.
`alpha=0` and `alpha=1` are exact lattice endpoints; no soft kernel is
introduced.

The sum is a separate function, so a graph cannot silently choose a weighted
average where order/visibility is required.

## Tests

```bash
PHI_CORE_DIR=/home/balrog/Documents/OpenCode/phi-core \
  python3 -m unittest tests.test_qtypes -v
```

The test file covers:

- `2 + 0` at explicit scale;
- scale mismatch diagnostics;
- missing scale;
- ordered binary alpha with deliberately unsorted layers;
- sum as a separate operation;
- checked-run failure context.

## Scope and limitations

- This is an authoring layer, not an assembler-level type system.
- The text IR still carries ordinary stream names; QProg checks at Python
  construction time.
- `ordered_composite` is a builder primitive, not a compiled `COMPOSITE`
  opcode or C/CUDA lowering. The primitive is deliberately small and
  compiles through existing `MUL`/`ADD`/`SUB`/`DIV` ops.
- It validates scale *context*, not shape or every op's numeric semantics;
  those remain with the assembler and backend gates.
- Existing `Prog` users are unaffected.

## Follow-up

If accepted, the next steps are:

1. add a typed `Prog` option or a small type-checking pass that can use the
   same metadata at assembly time;
2. add a C/CUDA `COMPOSITE` lowering with exact alpha endpoints;
3. add a runtime trace formatter that prints decoded values and scale labels
   together.
