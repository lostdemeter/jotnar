# GAPS (from the transformer-block exercise — measured, 2026-09-30)

Method: wrote programs/xf_block.asm (Qwen-style encoder block) using NEEDED
mnemonic names, ran the assembler, neutralized each failure, re-ran. The
assembler's fail-loud IS the enumerator. Re-run after Batch 1 (2026-09-30):
remaining distinct gaps are ROTARY and BATCH_MATMUL (+ cascade); SILU never
fired (cascade) but is now wired -- verify when its inputs survive.

## Batch 1 CLOSED (2026-09-30): MATMUL, SOFTMAX, RMSNORM, SILU wired + gated

Wrappers add NOTHING (0-diff vs phi-core fns, test_asm.py). One authored
mini-decision inside SOFTMAX: num/den->probs via tdiv-to-2^-18 +
from_fixed@BIAS (overflow-asserted). One CONTRACT found by measurement:
to_fixed saturates above 1.0 at BIAS scale, so this softmax REQUIRES
pre-scaled inputs (T-transformation doctrine); out-of-contract behavior is
PINNED by gate (saturates to softmax of clipped inputs), not barred.
Full-range attention needs the T-transform path (backlog). Also: eps_c is
scale-regime-dependent by nature (CONFIG default 4514 = promoted-test
convention); per-model eps calibration is backlog, stated.

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
