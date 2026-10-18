# Diagonals: same-program batch fusion

Two inferences through the same listing should fuse into one
execution (hypotenuse, not width+height). Phase 1 proves it on the
lattice MATMUL+ADD+SILU chain; Phase 2 carries it to the flagship
seam and cross-program alignment.

## Phase 1 — batch rows (proven, `tests/test_diagonal.py`)

Two input sets, shared weights (the batch-decoding shape):
activations `(R,4)` + biases `(R,3)` stacked along rows to `(2R,·)`,
weights `(4,3)` untouched. The listing runs UNCHANGED — the fused
program is the same text over taller matrices; outputs split by row
halves. No op refused the stacked shape (that refusal would have
been the finding scoping Phase 2; there was none on this chain).

## Exactness contract

Row-independence is structural (MATMUL rows, elementwise ADD/SILU),
so the bar is byte-identity: every output (MID/SUM/OUT), every
trial, split halves == separate runs; fused rerun byte-identical.
Graded so in the gate, 12/12 randomized trials, R=1..4.

## Measured (CPU lattice, 12 trials)

```
fused:    36 op-execs in  8.0ms
separate: 72 op-execs in 24.4ms
static:    3 bound ops either way (one listing)
```

Dispatch halved, flops identical — the 3x wall gap at this toy size
is per-op dispatch overhead, not compute savings. No latency claim
beyond this row; placement honesty applies (cf. protocol reports).

## Phase 2 needs

- Flagship seam: the image pipeline's ops (SPLAT_BLUR, GAIN,
  BETA_V5, ...) must each pass the same stacked-shape trial —
  any refusal scopes the work (split program? per-op batch rule?).
- Cross-program alignment: two DIFFERENT listings fused needs
  stream-name/shape alignment first (Phase 1 only covers same-text).
- Block-diagonal weights (different weights per inference) stays
  the documented alternate for the non-shared-weights case.
