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
Bar: **lift ≥ 0.10** (10pp absolute). Derivation: the honest
baseline is empirical unpaired keys (seeded shuffle), never a
formula. Correction (post-measurement, method and bar untouched —
Phase-19 precedent for open corrections): the constitution's first
draft derived a 0.59 chance rate from 1−(1−2τ)^H, which assumes
INDEPENDENT uniform phases per head. Wrong: heads share the key
(phase_h = γ_h·key), so per-head phases are deterministically
linked. Measured uniform-key rate is 0.343, matching the
seeded-shuffle nulls (~0.35) — the gate cross-checks the two
independent null constructions agree. Binomial noise at N≈10^5+
same-depth pairs is σ≈0.002, so 0.10 is ~50σ above noise (no
fluke passes) yet small enough to catch weak-but-real signal:
any reliable lift nominates a resonant-attention student round;
the bar is a gate, not a grade.
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

## Results (1 analysis run, 12s CPU — `scripts/analyze_resonance_depth.py`)

4032 tokens (6 frames × 672 cells), 8,126,496 pairs, 865,371
same-depth (Δ ≤ 0.1), of which 668,510 cross-frame. Null = seeded
shuffle (seed 0); V1 lift stable +0.0074/+0.0075 across seeds
0/1/2 (diagnostic reruns, /tmp-only — genuine weak echo, not
seed luck).

```text
variant     same   cross  null   lift    spearman  bar?
V1_luma     0.3654 0.3563 0.3579 +0.0075 +0.0039   fail
V2_lumapos  0.3445 0.3445 0.3442 +0.0003 +0.0006   fail
V3_rgbpack  0.3440 0.3433 0.3440 -0.0000 -0.0002   fail
```

Verdict: **dead-for-depth**. No variant clears lift ≥ 0.10 —
V1's +0.0075 is ~15σ real but two orders of magnitude under any
student-nominating strength; V2/V3 are null (±0.001 across
seeds). Notably V2's position multiplex adds nothing over V1
(0.3445 vs 0.3654 same-rate — locality does not rescue it) and
color (V3) adds nothing over luma. Sibling tree untouched
(`git status` clean apart from pre-existing untracked files; no
`__pycache__` — gated). The repo's dedup/search resonance claims
are untouched by this finding (separate verdicts, out of scope
as frozen).
Gate `tests/test_resonance_depth.py` (CPU, <1s): phase
determinism (in-process + fresh subprocess), sibling-pristine,
null cross-check (synthetic-uniform 0.343 vs seeded-shuffle
0.349), schema, lifts vs bar, verdict, null-documented.
Neighbors: none touched (strata read-only reuse only —
`test_strata_marigold.py` green, pasted below).

## Discord — the complement test (constitution, FROZEN pre-run)

Resonance asked whether NEAR phases share depth (+0.0075 max —
dead). Discord asks the complement: do ANTI-phase pairs share
depth? Same three key variants V1/V2/V3 (comparability — no new
keys; mid-round metric invention voids the round).
Discord metric frozen: discordant ⟺ min_h d_h > 1−τ (far in EVERY
head — the exact complement of resonate's ∃-head-closeness),
τ=0.1, H=4, same tokens/pairs/ε as the resonance round.
Discord lift = P(discord | same-depth) − P(discord | shuffled
null), same seeded-shuffle nulls (seed 0), same 0.10 nominating
bar. The heads-share-key correction carries over (cited, not
re-derived): no independence formula anywhere near this metric —
empirical nulls rule, now and always.
Winner rule: discord lift ≥ 0.10 on any variant = signal
(nominates the routing/content-split investigation — nothing
more; no student round, no machinery). Below = dead, and the
resonance pair is complete (similarity AND complement both
measured — the phase-depth question closed in both directions).
Note on power (stated pre-run, not a hedge): discord pairs may
be rare (uniform-independent guess said 1e-4; linkage decides
empirically) — N_discord is banked alongside; a tiny N with zero
same-depth hits is itself a finding (anti-phase pairs barely
exist under these keys), reported not smoothed.

### Discord results (TBD — analysis lands here)
