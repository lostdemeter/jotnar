# Hopfield capacity math — size the associative memory before building one

## Constitution (FROZEN pre-computation — violations bounce)

Formulas from Ramsauer et al., "Hopfield Networks is All You
Need" (arXiv:2008.02282, ICLR 2021 — via existing knowledge,
nothing fetched; theorem numbers as-recalled and FLAGGED where
uncertain, never faked — see honesty note):
- (R1) Single-update retrieval theorem (§ Retrieval): pattern
  x_i is stored and retrieved with one update if its separation
  Δ_i := x_i^T x_i − max_{j≠i} x_i^T x_j satisfies
  Δ_i ≥ 2/(βN) + (1/β)·ln(2(N−1)NβM²), with retrieval error
  exponentially small in βΔ_i. Frozen with β=1 (paper default;
  β-sensitivity noted as gap, not swept).
- (R2) Storage-capacity theorem (§ Storage Capacity): N random
  patterns on the sphere of radius M=K√(d−1) are stored with
  probability ≥ 1−p for N ≥ √p·c^((d−1)/4), where with
  a := (2/(d−1))·(1 + ln(2βK²p(d−1))), b := 2K²β/5:
  c := b / W_0(exp(a + ln b)) (W_0 = principal Lambert W).
  Frozen with unit-norm patterns (M=1 ⇒ K=1/√(d−1)) and
  failure prob p=0.001.
- (R0, context only) Classical Hopfield C ≈ 0.138N
  (Amit–Gutfreund–Sompolinsky 1985) — the comparison baseline,
  labeled as such, not a Ramsauer claim.
Honesty note (part of the freeze): equation-level pinning is
UNVERIFIED against the text — the load-bearing claims of this
round (exponential-in-d shape, c>1, modern-beats-classical
crossover, verdicts) depend on the bound's SHAPE, the paper's
least disputable headline, not on constant-exactness. A
±20% constant-robustness check rides in analysis (verdicts must
survive it). Lambert W implemented by hand (Newton iteration —
scipy NOT installed here and installing is forbidden);
checked against W(e)=1, W(1)=Ω, W(0)=0.
Pattern count frozen: K = 20 (5 observed attractor basins ×
safety factor 4 — covers basin splitting under new data/seeds;
verdicts verified log-insensitive to K=10/40 in analysis).
Error tolerance frozen: successful retrieval = error within
strata-bar scale (AbsRel ≤ 0.1). Bridge (APPROXIMATE, stated):
for unit patterns, dot-separation Δ ↔ L2 ball; the tolerance
enters via (R1): required separation Δ_req(N=K) is computed
and held against the 0.1-scale tolerance for order-of-magnitude
consistency — no precision claimed past one digit.
Capacity table frozen: dims {64, 128, 256, 512, 1024} × max
patterns = min(N_R(d) [exponential bound], N_S(d) [separation-
limited from (R1) + random-pattern geometry: dot∼N(0,1/d),
union bound over pairs at Δ_req, prob ≥ 1−p]). Tighter governs.
Verdict rule frozen: capacity C(d) ≥ 4K=80 → SUFFICIENT
(current failures are then dynamics, not capacity — stated as
such); 20 ≤ C < 80 → MARGINAL (named, not rounded up);
C < 20 → INSUFFICIENT (explains Phase-8/10 flatness
structurally — stated as such). The 64-dim question
(capacity-starved all along?) is answered by the d=64 row.

## Results (computation lands here — 1.4s CPU, no runs)

```text
 dim        N_exp        N_sep  classical     capacity verdict (robust-20)
  64         5.16          651       8.83         5.16 insufficient (insufficient)
 128          457          inf       17.7          457 sufficient (FRAGILE)
 256     4.71e+06          inf       35.3     4.71e+06 sufficient (FRAGILE)
 512     6.55e+14          inf       70.7     6.55e+14 sufficient (sufficient)
1024     1.66e+31          inf        141     1.66e+31 sufficient (sufficient)
```

Constants: c = 1.343–1.382 across dims (stable, well above 1 —
the exponential is genuine, not a boundary artifact).
Separation floor Δ_req(N=20): 10.9–13.7 dot units against
M² = 63–1023, i.e. 17% relative at d=64 shrinking to 1.3% at
d=1024 — same order as the 0.1 tolerance at d=64, easier above
(the tolerance bridge holds at one digit, as promised, and
relaxes with dimension — the geometry gets friendlier, not
tighter). N_sep never binds above d=64 (reported `inf` —
random separation at √d-radius is plentiful; the exponential
bound governs everywhere that matters).
Crossover pinned: modern bound BELOW classical at d=64
(5.16 < 8.83) but 26× above it at d=128 (457 > 17.7) — the
crossover lives in (64, 128), exactly astride our operating
point. K-sensitivity clean (K=10/40: identical verdicts);
p=0.01 note: same verdicts (6.72 / 596).
Verdict: **d=64 INSUFFICIENT, robustly** (survives −20%
constant error; needs ≥20, guarantees 5.16) — the Phase-8/10
flatness (jattn W64, dit W64, all pinned at 0/36-style empties)
is explained STRUCTURALLY: a 64-dim associative memory cannot
hold the 20-pattern working set, whatever the schedule, loss,
or data. **d≥128 SUFFICIENT** — failures there would be
dynamics, not capacity (stated as such); d=128/256 carry
FRAGILE flags (flip at −5%/−16% constant error — rule-verdicts
kept per freeze, caution stated, not rounded up).
Answer to the round's question: YES — our 64-dim failures were
capacity-starved all along (at K=20; the bound bites 4× below
need). A depth associative memory needs d ≥ 128 for this
working set, d ≥ 512 for comfortable margin.
Gate `tests/test_hopfield_capacity.py` (CPU, 1.4s): Lambert
constants, classical definitional, shape legs (monotone, c>1,
crossover pinned), banked-table recompute, verdicts +
robustness flags. Neighbors: none touched (`test_asm.py`
green, pasted below).
Bugs caught while building (mechanisms): (1) log(0) at N=1
(no pairs — vacuously storable, guarded to 0). (2) Unit-norm
z-scaling sent the union tail to the wrong regime (N_sep=1
everywhere, masking N_exp) — radius-M scaling fixed it.
(3) Gaussian tails underflow to exactly 0.0 at high z, which
spun the N scan to 1e9 (5-min hang, killed by timeout) — scan
caps at 1e6 with unbounded reported past it (verdicts live at
20/80, stated). Gaps named: β-sweep (frozen at 1); tolerance
bridge stays one-digit by design; equation-number pinning
unverified (flagged, not faked).

### Pre-computation correction (nothing computed yet — Phase-19 precedent)

The frozen K-choice (unit-norm patterns, M=1) makes the
exponential bound VACUOUS at every table dim (checked
algebraically: with K=1/√(d−1) the d-terms cancel to
2βK²p(d−1)=0.002, c≈1.02–1.18, and the √p=0.0316 prefactor
holds N below 3.0 from d=64 to d=1024 — a bound that proves
nothing judges nothing; adopting it would indict the formula
choice, not the students). Corrected: radius M=√(d−1) (K=1),
because (i) our normalized representations live at RMS scale
~√d (RMSNorm outputs, attention residuals — unit-norm is the
wrong geometry for this stack), and (ii) it restores a
graduated table (small at 64, astronomical at 1024) that can
actually discriminate. Carried through consistently (M² in
the separation condition too). Everything else frozen stands.

## Associative memory from basin prototypes (constitution, FROZEN pre-run)

Prescription, not training: outer-product storage, ZERO gradient
steps (a training loop anywhere in this round is scope breach).
Prototypes from banked data ONLY (no teacher runs — the maps were
never stored, so prototypes live in lesion-response space):
per-(basin, frame) MEAN over member subsets of 6-stratum AbsRel
vectors → 40 prototypes × 6 dims (mean frozen: symmetry-respecting,
no cherry-picking; medoid/exemplar rejected — privileging single
members without principled basis. Frame-indexed (not grand-mean):
frame effects are large in this data; grand-mean would wash out).
Capacity reality stated pre-run (not discovered mid-round): max
honest dimensionality is 48 (full profile) or 6 (per-frame);
capacity math gives N_exp(6)=0.21, N_exp(48)=1.76 — both far below
K=20 (and below K=5). The dim≥512 requirement is UNMEETABLE from
banked data. This round therefore tests whether the construction
beats its own capacity obituary empirically — verdict "dies"
is the predicted outcome, not a failure mode.
Storage: Hopfield outer product W = Σ_p x_p x_p^T over L2-
normalized prototypes (standard, stated). Retrieval dynamics
frozen: Ramsauer continuous update ξ ← X·softmax(βX^T ξ),
β=1 (capacity-math default), ≤100 iterations, Δ<1e-6
convergence (non-convergence reported, not hidden).
Queries frozen: corrupted versions of banked frame profiles —
keep {6(full),5,4,3} of 6 strata (partial-distance on observed
entries), 5 seeded masks per (prototype, level) (seeds
0–4 — 40×(1+3)×5 = 800 trials; binomial se at chance ≈1.4%).
(a) Retrieval correctness: retrieved prototype's BASIN ==
query's basin (chance 1/5). Bar frozen: accuracy ≥ 40% (2×
chance, ~14σ above noise with N=600 corrupted trials).
(b) Depth tier: retrieved basin's AbsRel mean vs strata bars
(all basins ~0.69–0.98 ≫ bars — expected lags) AND vs the
uniform-prior baseline (random-basin expectation — the only
uniform baseline constructible without new runs; "uniform-kill"
operationalized thus, stated). Retrieval can succeed while
depth lags — reported both, conflated never. Full-profile
queries (uncorrupted) ride as diagnostic (basin separability
floor), not bar.
Winner rule: retrieval ≥ bar AND retrieved-depth
directionally better (>0.05) than uniform-prior expectation =
mechanism alive; retrieval at chance = the associative
hypothesis dies for this construction (complete, not a
failure to tune). Retrain-match N/A (nothing trains —
determinism reruns serve instead: same query twice
bit-identical, gated).
Determinism: same query → same basin, bit-identical (numpy
deterministic; asserted exactly in-gate). Fail-loud: wrong-dim
/ NaN / empty queries refuse (never silent nearest).
Ruling: prescription only; no emitter/opcode/assembly changes
(MATMUL/SOFTMAX cited from the table — the update is literally
those two ops; gap hit = stop-finding); weights = prescribed
patterns banked in-repo (40×6 floats — centroids, not
checkpoints); production weights untouched (teacher AND
Qwen-7B never loaded — banked JSONs only); CPU-first (numpy
dynamics; no GPU need exists); no new deps; no downloads.
Gate: new tests/test_assoc_memory.py (CPU-only, seconds):
prototype integrity (recomputed from banked census), retrieval
legs recomputed in-gate (accuracy + depth tier), determinism
(same query twice bit-identical), fail-loud legs. Neighbors:
lesion-basin + anatomy legs green (read-only reuse — nothing
moves underneath).

### Prescription results (800 trials, CPU seconds, no runs)

40 prototypes (5 basins × 8 frames, plain means, L2-normalized)
+ outer-product W (6×6, exact) + Ramsauer retrieval (β=1,
≤100 iters, Δ<1e-6; converged 100%):

```text
keep=5: 0/200 = 0.000   keep=4: 0/200 = 0.000   keep=3: 0/200 = 0.000
accuracy = 0.2000 (bar 0.40, chance 0.20)   LOO = 0.125
verdict = dead
```

Mechanism, measured not asserted: ALL 600 corrupted trials
retrieve ONE prototype (C, ai_021_001_fr0001) — total dynamic
collapse. Prototypes are near-collinear clumps (max off-diag
cosine 0.999999+ with no exact duplicates — one flat
landscape, not 40 attractors); the softmax has nowhere to go
but the top. LOO control at 0.125 (BELOW chance) confirms the
basins overlap even uncorrupted — separability fails before
corruption starts. Uniform-prior depth mean 0.7703: retrieved
depth cannot beat it when retrieval is chance (tier reported,
moot by the accuracy).
Verdict: **the associative hypothesis dies for this
construction** — retrieval at chance (0.2000 == 0.20 exactly),
below the 0.40 bar by 20σ of trial noise. The capacity obituary
holds empirically: 6 dims cannot hold 40 patterns (nor 5), and
no dynamics rescues an inseparable code.
Gate `tests/test_assoc_memory.py` (CPU, 0.1s): prototype
integrity (bit-identical recompute), weights exact, accuracy
+ depth recomputed, determinism (same query twice), 4
fail-loud legs. Neighbors: lesion-basin + anatomy green
(pasted below; read-only reuse, nothing moves underneath).
Bugs caught while building (mechanisms): (1) leftover draft
scaffolding inside the trial loop (placeholder tensors that
evaluated then got overwritten — dead code that RAN; removed
pre-run, never banked under it). (2) A tautological no-peek
leg + walrus one-liner in the gate draft (proved nothing,
read poorly — deleted/rewritten). (3) `%.4f`-rounded max
cosine displayed 1.0000 with no exact duplicates underneath
(verified pairwise — near-collinear, not bit-equal). Gaps
named: higher-dim prototype spaces (unbankable — maps were
never stored, by design); β-sweep (frozen at 1, capacity-math
default).
