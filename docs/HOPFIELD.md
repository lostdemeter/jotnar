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

## Hopfield prototypes — harvest real-dim patterns, no training (constitution, FROZEN pre-run)

Harvest, not training: forward capture hooks ONLY (zero zeroing/
lesioning — `requires_grad_(False)`, `no_grad` + bf16 autocast exactly
like `infer_one`; a loss/backward/optimizer anywhere is scope breach).
Harvest layers frozen pre-run (≤3, exact module paths in the banked
anatomy suffix convention after "Diffuser." per
`teacher_anatomy.json:99`; no exploring mid-round — a path matching
nothing is a stop-finding, fail loud like the lesion typo-guard):
(1) `transformer_blocks.29` — mid-stack DiT block output, image-stream
hidden_states (output tuple index 1), dim 3072;
(2) `transformer_blocks.59` — final DiT block output, image-stream
hidden_states, dim 3072;
(3) `norm_out` — final AdaLayerNormContinuous head output (pre-proj_out
tensor), dim 3072.
Images frozen: the 8-frame Hypersim eval set (first 2 frames of
ai_018_001, ai_021_001, ai_022_001, ai_023_005) — identical inputs to the
lesion/basin evidence the prototypes must explain, so no new-image
confound (one-line justification as required; banked church/dogs/
squirrel fixtures NOT used — stated).
Prototype construction frozen: per-image exemplars — mean-pool over the
image-token sequence of each capture → 8 prototypes × 3072 per harvest
point (24 total). Per-basin means REJECTED (banked basins partition
lesion subsets, not images — no basin→image map exists without inventing
one = peeking-class violation). Medoids REJECTED (privileging single
members without principled basis — assoc-memory precedent). Pooling fixed
before the first run — no peeking at which separates best.
Capacity check against the banked math: harvested dim 3072 vs the d≥512
robust-sufficient row (N_exp(512)=6.55e+14 ≫ K=20) — stated as PASS/FAIL
per harvest point, not argued (expected PASS ×3: 3072 > 512 and N_exp
grows in d).
Determinism: same image twice → bit-identical prototypes; harvest runs
ai_018_001 frame 0 twice and banks per-point max-abs-diff (bar exactly
0.0); the gate reads banked evidence — heavy reruns are NOT routine.
Ruling: hooks only; weights read-only out-of-tree (sha256 of the
established 7 files — 5 transformer shards + trainables + VAE — pre AND
post, banked; a single modified byte voids the round); no training; no
emitter/opcode/assembly changes; no new dependencies (marigold-env stack
already used by `teacher_lesion.py`: torch/diffusers/peft/bitsandbytes/
cv2/numpy); production weights untouched; teacher AND Qwen-7B never
co-loaded (text encoder never loaded — precomputed embeds feed all
blocks; VRAM math: 4-bit NF4 20B ≈10GB + LoRA trainables 1.85GB + bf16
VAE ≈0.2GB + embeds 19MB + single-frame activations ≈ 13GB total < 24GB
3090Ti — stated pre-run); GPU solo for hook sessions (nvidia-smi pre-run:
zero compute processes, 451MiB display-only, 23.6GB free — stated);
no downloads; MUST run under `~/marigold-env/bin/python` with a CLEAN
env (`cuda_env.sh` poisons it — established, never re-proven).
Gate: new `tests/test_hopfield_prototypes.py` (fast legs: determinism
evidence + schema + capacity-check legs reading banked prototypes +
banked math; heavy reruns are NOT routine). Neighbors: anatomy +
assoc-memory legs green (read-only reuse — nothing moves underneath).

### Harvest results (GPU solo, hooks only, no training)

Run: `~/marigold-env/bin/python scripts/harvest_hopfield_prototypes.py`
(clean env, GPU solo — zero compute procs pre-run, 451MiB
display-only, 23.6GB free). VRAM measured alloc=13.7GB reserved=14.9GB
(< 24GB 3090Ti — matches the ≈13GB pre-run math; text encoder never
loaded, teacher+Qwen-7B co-load impossible by construction).
Eval context mirrors `infer_one` exactly (no_grad + bf16 autocast,
LANCZOS4 768x576, t=0.499, precomputed embeds).

```text
harvest points matched 1:1: ['transformer_blocks.29', 'transformer_blocks.59', 'norm_out']
determinism ai_018_001/fr0000 x2: transformer_blocks.29=0.00e+00 transformer_blocks.59=0.00e+00 norm_out=0.00e+00
harvested ai_018_001_fr0000..ai_023_005_fr0001 (8 frames, ~1s each)
sha post == pre (7 files, nothing written)
POINT dim=3072 K=20 requirement=d>=512: transformer_blocks.29=PASS transformer_blocks.59=PASS norm_out=PASS
```

Harvest table (per-point dims + capacity verdicts):

| harvest point | capture | dim | N_exp(dim) | verdict |
|---|---|---|---|---|
| transformer_blocks.29 (block out[1], image stream) | [1,1728,3072] → mean-pool | 3072 | 1.3e+97 | PASS (3072≥512) |
| transformer_blocks.59 (block out[1], image stream) | [1,1728,3072] → mean-pool | 3072 | 1.3e+97 | PASS (3072≥512) |
| norm_out (head output, pre-proj_out) | [1,1728,3072] → mean-pool | 3072 | 1.3e+97 | PASS (3072≥512) |

24 prototypes banked (`samples/marigold/hopfield_prototypes.json`,
1.15MB): 8 frames × 3 points × 3072, all finite + nonzero.
Determinism bar met exactly (0.00e+00 all points — same image twice
bit-identical). Checksums pre==post (7 files).
What the listing port inherits: (1) dims PASS with astronomic margin
— capacity can never explain a future retrieval miss at these points
(failures would be dynamics, stated as such); (2) L2-NORMALIZE before
outer-product storage — banked pooled norms are raw residual-stream
magnitudes (b29 ≈1.5e8, b59 ≈1.3e10, norm_out ≈148; no inf, but 8 orders
apart across points), while Ramsauer patterns live on the sphere and the
capacity math assumes radius-M geometry — storing raw would let b59
swallow everything (assoc-memory collapse precedent: one flat winner);
(3) point choice is a dynamics question now, not capacity — b29 (mid)
vs b59/norm_out (late) differ in abstraction, identically in verdict.
Gate `tests/test_hopfield_prototypes.py` (CPU, 0.03s): schema, values,
shapes, determinism evidence, capacity recompute, checksums. Neighbors:
anatomy + assoc-memory green (read-only reuse, nothing moves underneath).
Bugs caught: none in-harvest (paths matched 1:1 first try, recipe mirror
exact). Gaps named: basin→image map still absent (per-basin means remain
unconstructible without invention); retrieval dynamics over these 24
patterns untested (next round's question, not this one's).

## Hopfield retrieval dynamics — prove the product retrieves (constitution, FROZEN pre-run)

Dynamics only: Ramsauer update on L2-normalized banked prototypes
(normalize-before-store invariant from the harvest round), β=1
(capacity-math default), ≤100 iterations, Δ<1e-6 convergence
(non-convergence reported, not hidden). Three independent 8-pattern
memories (one per harvest point: transformer_blocks.29/.59, norm_out);
per-point tables; the listing port inherits the winning point(s).
Queries frozen (exact construction, no peeking at which retrieves well):
(a) corrupted variants — additive zero-mean Gaussian, σ = f×RMS of the
query prototype, f ∈ {0, 0.01, 0.05, 0.1, 0.2} (f=0 = exact-query sanity,
must self-retrieve), numpy RandomState(1000+10·seed+level_idx), seeds
0–4 per (frame, level, point) → 8×5×5 = 200 trials/point, 600 total;
(b) held-out LOO — each frame's exact harvested pattern queried against
the other 7 of the same point; correct = same-scene sibling frame;
chance 1/7 (7 stored, 1 sibling — derived); DIAGNOSTIC (no bar — stated,
assoc-memory LOO precedent).
Correctness (family a): retrieved frame == query frame. Chance 1/8 =
0.125 (8 stored patterns, uniform random guess — derived). Bar 0.25 =
2× chance (assoc-memory precedent: 2× chance at 1/5→0.40 — derived, not
just stated).
Depth tier, secondary and explicit: retrieved prototype's frame →
banked baseline mean AbsRel (`teacher_baseline_H.json`) vs strata bars
(`test_strata_marigold.py:34-35`: near/far/planar/smooth 0.09,
nonplanar 0.08, tex 0.07 — every frame mean 0.20–0.40 misses them all,
expected, teacher misses its own bars) AND vs the uniform-kill baseline
operationalized as the random-frame expectation (uniform mean 0.2936 over
the 8 frames — assoc-memory precedent). Reported alongside retrieval,
never conflated — retrieval can succeed while depth lags, and that split
is itself a finding.
Winner rule: accuracy ≥ bar AND retrieved-depth directionally better
(>0.05 lower mean AbsRel) than the uniform mean = product alive;
accuracy at chance = dynamics dead for these prototypes (complete, not a
failure to tune).
Ruling: dynamics only (no training, no emitter/opcode/assembly changes,
no new dependencies — numpy only); banked prototypes read-only
(recompute-integrity leg, never re-harvest); production weights untouched
(teacher AND Qwen-7B never loaded — banked data only); CPU-first
(microsecond dynamics: 600 trials × ≤100 iters × 8×3072 MATMUL ≈
milliseconds numpy — no GPU need exists, stated pre-run); no downloads.
Gate: new `tests/test_hopfield_retrieval.py` (CPU-only, seconds):
prototype-integrity (recomputed from banked harvests), correctness legs
(per corruption level vs frozen bar), determinism (same query twice
bit-identical), convergence legs (iters ≤ cap honored), fail-loud legs
(malformed queries refuse). Neighbors: assoc-memory + prototypes +
lesion-basin green (read-only reuse — nothing moves underneath).

### Retrieval results (TBD — dynamics land here)

### Retrieval results (600 trials + 24 LOO, CPU 0.2s, banked data only)

`python3 scripts/hopfield_retrieval.py` — three independent 8-pattern
Ramsauer memories (L2-normalized banked prototypes, β=1, ≤100 iters,
Δ<1e-6; converged 100%, max 2–4 iters — the dynamics converge fine,
separability is what fails):

```text
transformer_blocks.29: f=0.0:5/40 f=0.01:5/40 f=0.05:5/40 f=0.1:5/40 f=0.2:5/40
  acc=0.1250 (bar 0.25, chance 0.125) loo=1.000 (chance 0.143) conv=1.000 maxit=4 rdepth=0.3334 uniform=0.2936 delta=-0.0397 verdict=dead
transformer_blocks.59: (identical rows) acc=0.1250 loo=1.000 conv=1.000 maxit=2 rdepth=0.3334 delta=-0.0397 verdict=dead
norm_out: (identical rows) acc=0.1250 loo=1.000 conv=1.000 maxit=4 rdepth=0.3334 delta=-0.0397 verdict=dead
```

Retrieval table per corruption level (all three points identical —
every cell 5/40 = 0.125 = chance exactly):

| level f | b29 | b59 | norm_out | bar |
|---|---|---|---|---|
| 0.0 (exact) | 5/40 | 5/40 | 5/40 | 10/40 |
| 0.01 | 5/40 | 5/40 | 5/40 | 10/40 |
| 0.05 | 5/40 | 5/40 | 5/40 | 10/40 |
| 0.1 | 5/40 | 5/40 | 5/40 | 10/40 |
| 0.2 | 5/40 | 5/40 | 5/40 | 10/40 |

Mechanism, measured not asserted: ALL 200 trials/point retrieve ONE
frame (ai_021_001_fr0000) — total dynamic collapse at every point,
including exact queries (f=0 winners: all 8 → frame 2, never self).
Normalized prototypes are near-collinear clumps (max off-diag cosine
0.9974 b29 / 0.99999 b59 / 0.9985 norm_out) — one dominant attractor,
not 8; the softmax has nowhere to go but the top (assoc-memory collapse
precedent, now at real dims: capacity PASSed, separability failed).
The 0.125 is structural, not sampling: the single winner contributes
exactly its own 25/200 queries. LOO same-scene 1.000 (diagnostic):
scenes cluster locally, but globally one attractor wins — separability
fails across scenes, not within. Depth tier: retrieved 0.3334 =
winner-frame mean (vs uniform 0.2936, delta −0.0397 — worse, not >0.05
better; vs strata bars 0.07–0.09 — every frame misses, as expected).
The split finding (retrieval-at-chance WITH perfect convergence) is the
point: dynamics dead, optimizer-of-dynamics cannot help — there is
nothing to tune toward, only codes to separate.
Verdict: **dynamics dead for these prototypes** — 0.1250 < 0.25 at all
points and levels, depth delta −0.04. Complete per the frozen winner
rule, not a failure to tune. Capacity obituary hierarchy stands:
6-dim died of capacity (N_exp=0.21), 3072-dim dies of separability
(N_exp=1.3e+97, cosine 0.99999) — the bound was necessary, never
sufficient.
Gate `tests/test_hopfield_retrieval.py` (CPU, 0.3s): prototype
integrity (unit rows), per-level recompute, determinism (same query
twice), convergence (cap honored), 4 fail-loud legs, frozen-rule
verdicts. Neighbors: assoc-memory + prototypes + lesion-basin green
(pasted below; read-only reuse, nothing moves underneath).
Bugs caught: none (exact-query collapse first looked like a corrupt()
bug — f=0 returns the untouched copy, verified; the collapse is the
dynamics, confirmed by the cosine audit). Gaps named: code separation
(contrastive/whitened prototypes untested — next construction's
question); β-sweep (frozen at 1); per-point memories only (joint
24-pattern memory untested — capacity covers it, separability
wouldn't improve, stated not run).

## Separable prototypes — whitening, token-level, or bust (constitution, FROZEN pre-computation)

Analysis + prescription only (numpy/stdlib — scipy absent on this box,
stated; no training, no emitter/opcode/assembly changes; banked
prototypes read-only, never re-harvested; teacher AND Qwen-7B never
loaded; CPU-first — microsecond dynamics plus one 3072² eigh, GPU need
none exists, stated pre-run; no downloads).
Prototype constructions frozen pre-run (≤3). Pool adjudicated first,
then frozen — mid-round constructions void the round:
(C1) ridge-ZCA-whitened mean-pooled — ADOPTED: symmetric whitener
W=(C+εI)^{−1/2} via eigh, C = covariance of all 24 L2-normalized banked
patterns (shared fit across points — frozen), ε=1e−4 (order of the mean
eigenvalue ≈3e−4: stabilizes the ~3049 empty directions without erasing
the whitening), applied per point to its 8 patterns, re-normalized.
Dim stays 3072 → capacity N_exp=1.3e97 ≫ 24 PASS (stated, verified
in-analysis).
(C2) point-mean-centered — ADOPTED: subtract the point's own 8-pattern
mean (removes the dominant shared direction — the poor man's whitener),
re-normalize. Dim 3072 → PASS.
(C3) scene-contrastive (SIBLING-subtracted) — contrastive-separated
means ADOPTED in this exact form with reason: LOO showed scenes cluster
(1.000 same-scene), so remove the scene direction — per frame subtract
its same-scene sibling pattern, re-normalize (α=1 exact, no tuned
knob). Dim 3072 → PASS.
TOKEN-LEVEL — VOID pre-run with reason: token tensors were never banked
(harvest banked mean-pooled prototypes only) and re-harvest is forbidden
— unconstructible from banked data, stated not attempted (no silent
substitution).
Separability bar frozen alongside (adopted with reason): max off-diag
cosine < 0.9 AND mean pairwise cosine < 0.5. Reason: 0.9 restores
1−0.9=0.1 of separation margin — 10⁴× the dead baseline's 1−0.99999=1e−5
(where β=1 weight ratios sit at e^{1e−5}≈1, pure collapse); mean<0.5
ensures the clump disperses as a whole, not one outlier patched. The
0.99999 baseline is what must move — per construction, per point.
Winner rule: a construction clearing the bar (both clauses, all 3
points) earns the full retrieval battery rerun (same frozen 600-trial
construction + LOO diagnostic, same judge bar 0.25); retrieval above bar
= product alive; separability fails everywhere = prototypes unfixable
this way (complete — and the joint-24-pattern + β-sweep gaps inherit the
question EXPLICITLY, not hopefully: stated as the named next holders).
Gate: EXTEND `tests/test_assoc_memory.py` in place (new separability
key — construction integrity recomputed from banked harvests +
separability legs reading banked tables + retrieval-rerun legs for any
construction clearing the bar; existing legs untouched and green).
Neighbors: assoc + prototypes + lesion-basin green (read-only reuse).

### Separability results (TBD — computation lands here)

### Separability results (CPU seconds, banked data only)

`python3 scripts/hopfield_separable.py` — raw baselines reproduce the
dead round (0.9974/1.0000/0.9985), then the three frozen constructions:

```text
transformer_blocks.29:raw: maxcos=0.9974 meancos=+0.9651 fails
transformer_blocks.59:raw: maxcos=1.0000 meancos=+0.9999 fails
norm_out:raw:            maxcos=0.9985 meancos=+0.9831 fails
transformer_blocks.29:c1: maxcos=0.9977 meancos=+0.9964 fails
transformer_blocks.59:c1: maxcos=1.0000 meancos=+0.9999 fails
norm_out:c1:            maxcos=0.9990 meancos=+0.9976 fails
transformer_blocks.29:c2: maxcos=0.9151 meancos=-0.1332 fails
transformer_blocks.59:c2: maxcos=0.7833 meancos=-0.1377 CLEARS
norm_out:c2:            maxcos=0.9316 meancos=-0.1364 fails
transformer_blocks.29:c3: maxcos=0.4325 meancos=-0.1429 CLEARS
transformer_blocks.59:c3: maxcos=0.7298 meancos=-0.1429 CLEARS
norm_out:c3:            maxcos=0.9315 meancos=-0.1429 fails
constructions clearing all 3 points: []
```

Separability table (bar: maxcos<0.9 AND meancos<0.5):

| construction | b29 (max/mean) | b59 (max/mean) | norm_out (max/mean) | clears |
|---|---|---|---|---|
| raw | 0.9974/+0.97 | 1.0000/+1.00 | 0.9985/+0.98 | 0/3 |
| C1 ridge-ZCA | 0.9977/+1.00 | 1.0000/+1.00 | 0.9990/+1.00 | 0/3 |
| C2 centered | 0.9151/−0.13 | **0.7833**/−0.14 | 0.9316/−0.14 | 1/3 |
| C3 sibling-sub | **0.4325**/−0.14 | **0.7298**/−0.14 | 0.9315/−0.14 | 2/3 |

Mechanisms: (C1) ZCA-whitening barely moves anything — verified
mechanism, not a bug (W symmetric to 9e−15; the whitener is CORRECT):
ZCA is the whitening rotation CLOSEST to the originals, so
near-collinear inputs stay near-collinear; with 24 samples spanning ≤23
dims, equalizing variance cannot invent directions that aren't there.
Robustness note (analysis, NOT a new construction): even mean-centered
application reaches only 0.96 maxcos — still fails the bar, so the
verdict doesn't hinge on the centering choice. (C2/C3) removing the
shared direction disperses the clump whole (mean −0.13/−0.14 — the mean
clause passes everywhere) but one stubborn pair survives per failing
point (0.9151/0.9316/0.9315 — all norm_out-adjacent misses by a hair;
nearest miss overall C3 2/3).
No construction clears all 3 points → NO reruns earned per the frozen
winner rule (retrieval battery stays unrunned — stated, not skipped).
Verdict: **prototypes unfixable this way — construction-dead.**
Complete: the joint-24-pattern + β-sweep gaps inherit the question
EXPLICITLY (named next holders — hopefully nothing).
Hierarchy closed: 6-dim died of capacity, 3072-dim of separability,
separability survives whitening/centering/scene-contrast. The code is
entangled at the representation level (cosine 0.99999 across scenes at
every depth and every head output) — no post-hoc prototype surgery
separates what the encoder never pulled apart.
Gate: `tests/test_assoc_memory.py` extended in place (assoc-sep-*
key: constructions recomputed from banked harvests, bar legs, vacuous
rerun legs; all 10 pre-existing legs untouched and green). Neighbors:
assoc + prototypes + lesion-basin green (read-only reuse).
Bugs caught: none (C1's non-movement first looked like a transposed-W
bug — symmetry + covariance diagnostics clear it; the failure is the
geometry). Gaps named: joint-24 memory, β-sweep (explicit inheritors);
PCA-whitening (rotation-arbitrary — NOT a loophole, it scrambles the
frame identity the queries rely on, stated not run).
