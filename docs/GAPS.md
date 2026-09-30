# GAPS (from the transformer-block exercise — measured, 2026-09-30)

Method: wrote programs/xf_block.asm (Qwen-style encoder block) using NEEDED
mnemonic names, ran the assembler, neutralized each failure, re-ran. The
assembler's fail-loud IS the enumerator (11 rounds). What passed also matters.

## Measured missing mnemonics (assembler-reported, in firing order)

| mnemonic | status | path forward |
|----------|--------|--------------|
| RMSNORM | EXPOSURE (phi-core rmsnorm_int exists) | wire + gate |
| MATMUL | EXPOSURE (phi-core matmul_int exists) | wire + gate |
| ROTARY | GENUINE (nothing exists: needs sincos LUT + pair-rotate op, sigmoid pattern) | author as structure |
| BATCH_MATMUL | EXPOSURE+ (matmul exists 2D; heads need SPLIT convention) | wire + head-axis rule |
| SOFTMAX | EXPOSURE (phi-core softmaxN_triples exists) | wire + gate |
| SILU | EXPOSURE (phi-core silu_int exists; alt: MUL+SIGMOID needs SIGMOID too) | wire silu_int direct |

## Present and correct (exercise also validates coverage)

ADD (residuals), MUL (scaling), CONFIG/IN/multi-output machinery, all four
assembler discipline gates. SwiGLU's skeleton (MUL after activation) is
already expressible; only the activation itself is missing.

## Structural gaps (analysis, not assembler-reported -- v0.1 has no shapes)

1. TYPED STREAMS: layouts (HWC vs seq×dim vs heads) unchecked anywhere.
   compose.Stream HAS metadata; assembly ignores it. Needed before MATMUL
   means anything.
2. SPLIT/RESHAPE/TRANSPOSE mnemonics (IR has them as moves; heads need them).
3. BRANCH: general verdict-gated select (MIXDYAD hardcodes the pattern).
4. ITERATE + loop-carried state + dynamic shapes (KV-cache needs all three;
   driver loops today, language can't).
5. Static verifier: scales/geometry/compat at parse time (range estimator +
   seam chart in one hat).

## Priority (exposure before invention)

Batch 1 (wire+gates, no new math): MATMUL, SOFTMAX, RMSNORM, SILU.
Batch 2 (first genuinely-new structure): ROTARY (sincos LUT + rotate op,
same authoring bar as everything else).
Batch 3 (language features): TYPED STREAMS first (blocks everything else),
then SPLIT family, BRANCH, static verifier; ITERATE last (needs dynamic
shapes -- biggest design decision in the list, do not rush).
