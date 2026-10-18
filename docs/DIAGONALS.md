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

Step 1 result (STOP — refusal found, no Step 2): the flagship seam
refuses at SPLAT_BLUR (`tests/test_diagonal2.py` pins it green).

Per-op seam table (two 288-row images stacked, split vs separates):

```
rgb LIN Y A          fully split-exact (pointwise prefix)
AS COH (SPLAT_BLUR)  halo exactly rows {287,288} (radius-1 bleed,
                     max 233 int units)
D BEFF BD AE YENH G OUT  contaminated exactly at the halo;
                     OUT exact on 574/576 rows
fused rerun          byte-identical; 11 vs 22 op-execs
```

Scoping proposal (pick one before any Phase-2 fusion work):
(a) guard-band halo — drop 1 row each side post-split; exact
elsewhere, costs 2 rows per seam, zero code changes;
(b) split program — pointwise prefix fused (exact), SPLAT_BLUR +
tail per image (two PROGRAMs, honest seam handling);
(c) per-op batch rule — pointwise ops batch freely, spatial ops
need halo exchange or separate execution (a batching contract per
opcode for future emitter work).

Step 2 (cross-program alignment via block-diagonal weights) stays
parked behind full Step-1 green, which this is not. Stream-name /
shape alignment rules and the emitter-change question travel with
whichever scoping option is chosen.
