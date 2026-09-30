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
   CLOSED v1 2026-09-30: layouts "KIND:GEOM" on IN, per-op sigs with
   $VAR/$VAR^T/*, parallel layout dict (values untouched), concrete
   mismatches fail naming op+line+stream, UNKNOWN unifies silently (all
   suites pass with partial annotations = backward compat proven).
   Negative gates + gradual gate in test_asm.py. Stated limits: no rank
   arithmetic (MATMUL wildcard; phi-core asserts fail loud inside), checks
   at execute time (layouts ride payloads), POS/metadata as "*".
2. SPLIT/RESHAPE mnemonics (TRANSPOSE done; heads need SPLIT).
3. BRANCH: general verdict-gated select (MIXDYAD hardcodes the pattern).
4. ITERATE + loop-carried state + dynamic shapes (KV-cache needs all three;
   driver loops today, language can't).
5. Static verifier: scales/geometry/compat at parse time (range estimator +
   seam chart in one hat).

## Priority (exposure before invention)

Batch 1 (wire+gates, no new math): MATMUL, SOFTMAX, RMSNORM, SILU. CLOSED.
Batch 2 (ROTARY + BATCH_MATMUL/TRANSPOSE, xf_block runs). CLOSED (see above).
Pile A exposure batch CLOSED 2026-09-30 (12 mnemonics, 41 total):
ARGMAX (only new math in the batch: exact lattice ordering, tie->first,
negatives/zero handled; gated vs numpy), SLICE (bounds-checked windows),
CLIP/DIV/SIGMOID/RESCALE/GATHER/PRELU/POOLAVG/DECONV/INTERP/CONV (all 0-diff
vs source fns). Two mini-doctrines landed with it: non-dyadic INTERP is a
LOWERING ERROR (fail loud, never approximate); POOLAVG asserts HWC (SEQ
layouts fail loud rather than mis-average). RESCALE puts the only-scale-
changer in-language (needed where listings cross m_acc/m_cov).
Batch 3 (language features): TYPED STREAMS v1 CLOSED (see above). SPLIT
family CLOSED as RESHAPE2/RESHAPE3/PERMUTE3 exact moves (+TRANSPOSE; N-way
split deferred -- no demand, heads need reshape+permute only, stated).
BRANCH CLOSED as SELECT (verdict-gated general select; MIXDYAD's hardcoded
pattern generalized). LOOP CLOSED v1 as STATE + repeat() (fixed geometry
refused loudly upfront; STATE must be IN-seeded and OUT-assigned, both
gated; repeat == manual unroll bit-exact). Static verifier v1 CLOSED:
verify() replays unification over declared layouts with zero execution
(flagship 13/13, bad listings caught) + ranges.estimate() (hull intervals,
RANGE declarations, M.json coverage; asymmetric doctrine -- saturation flags
on any exceedance, underflow only whole-range-below; tensor_disc pins the
hull limit: precision loss inside spanning ranges stays with unit gates).
Remaining: static verifier extensions (per-op m override, typical-magnitude
reasoning -- stated limits) + WHILE (data-dependent termination: needs a
termination semantics + verdict integration design conversation first; all
current iteration needs are bounded, so no demand yet -- stated, not missing).
