# Resonance–depth correlation — does phase-similarity track geometric similarity?

## Setup provenance (read-only sibling, never vendored)

`resonant_array` lives at `/home/balrog/Documents/OpenCode/resonant_array`
(same pattern as `dav2_revisited`/`marigold-v2`: import path only, zero
files copied into this tree). Remote `git@github.com:lostdemeter/
resonant_array.git`, commit `4c4a749`, license GPL-3.0, same author —
import-only use documented here. Only `resonant_array.py` primitives
are used (`phase`, `circ_dist`, `ResonantArray.resonate`); nothing
else in their tree is imported, let alone modified. Bytecode writes
suppressed on import (`sys.dont_write_bytecode`) so not even
`__pycache__` lands in their tree (verified via their `git status`
after runs). `phi_constructive.py` read for context (LM-oriented
4-state construction — not for use this round).

## Key hypothesis (FROZEN — the key choice IS the hypothesis)

`phase(key, γ) = γ·key mod 2π` takes a scalar, so a patch must reduce
to one. Three variants, frozen (mid-round key invention voids):

- V1 `luma`: key = round(mean patch luma × 1000) (int 0..1000).
  Pure appearance, no position. Rationale: same-depth surfaces
  (walls, ground) share albedo → same key → same phase.
- V2 `lumapos`: key = luma_q × 4096 + cell_index (injective packing;
  depth is spatially smooth AND appearance-linked — tests whether
  locality+appearance beats appearance alone).
- V3 `rgbpack`: key = R×65536 + G×256 + B of quantized mean RGB
  (color-sensitive vs luma-only; same-depth-different-hue is the
  known failure mode of V1, stated in advance).

All keys < 2^24, so γ·key < 1.2e9 — double-exact (no float dust;
determinism leg guards it). Heads H=4 (first four zeros), threshold
τ=0.1 (`resonate` defaults) — sibling defaults, not tuned.

## Metric + bar (FROZEN, with derivation)

Pairs of patches: phase-distance d (min circ_dist over heads, i.e.
the resonate relation) vs teacher-depth-difference Δ = |mean depth
a − mean depth b|. Primary: same-depth-pair resonance lift —
`lift = P(resonate | Δ ≤ ε) − P(resonate | shuffled keys)`,
ε = 0.1 teacher units (~5% of the signed [-1,1] range).
Bar: **lift ≥ 0.10** (10pp absolute). Derivation: chance rate for
uniform phases is 1−(1−2τ)^H = 1−0.8^4 ≈ 0.59, so the null sits
near 0.59, not near 0 — the bar must be an absolute lift, not a
ratio. Binomial noise at N≈10^5+ same-depth pairs is σ≈0.002, so
0.10 is ~50σ above noise (no fluke passes) yet small enough to
catch weak-but-real signal: any reliable lift nominates a
resonant-attention student round; the bar is a gate, not a grade.
Null baseline = keys shuffled across patches (seed 0, stated —
deterministic null, cross-run stable). Secondary (reported, never
ruling): Spearman(d, Δ) per variant.
Fixtures: 6 banked real fixtures (church/dogs/squirrel/train +
2 heldout, 672×1024), 32px cells (672 patches/frame, 4032
tokens), teacher depth = patch-mean of banked .npy. No new runs.

## Winner rule

Any variant clears lift ≥ 0.10 → resonance carries geometric
signal (nominates a resonant-attention student round — nominated,
not run here). All below → dead for depth. The repo's other
resonance claims (dedup/search) are separate verdicts,
explicitly out of scope — a dead-for-depth finding touches
nothing there.

## Results (TBD — analysis lands here)
