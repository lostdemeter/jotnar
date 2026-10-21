# Listing debugger — breakpoints, watch, step, lesion tables

## Constitution (FROZEN pre-build — violations bounce)

Surface (frozen — `chain/debugger.py:Session`):
- `break_at(indices)`: op positions (0-based into the bound program —
  listing text lines shift under comments; positions don't). Bad
  index refuses loud.
- `watch(names)`: stream names, validated eagerly against the census
  stream set (unknown refuses loud at watch time, never a silent
  missing at stop time).
- `step(n=1)`: advance n ops unless a breakpoint hits first; returns
  the stop record (pos/op/line/watched values). Stepping PAUSES the
  existing op loop — no reimplementation of execution: a store_hook
  routes writes to debugger-owned env (value-identity proven by the
  hook's own contract: ChainMap env-first reads execute identical
  values) and raises a private pause at the stop op; each step is a
  fresh bounded run (O(n²), stated — the debugger is for small
  listings; served/CUDA stepping explicitly out of scope).
- `lesion(op_idx)`: zero-Out intervention through the EXISTING
  overrides mechanism (built for probes — cited, not reinvented):
  the op's OUT streams replaced by zeros_like post-production,
  downstream reads the zeros; tabled as per-OUT-stream
  (maxabs, mean) deltas vs the reference run. Unknown op refuses
  (overrides' own never-assigned tripwire backs it).
- `diff(other)`: run-vs-run byte-compare (exact array equality incl.
  dtype/shape) with first-divergence reporting (op index, stream,
  maxabs) or identical.
Read-only w.r.t. listings (assemble + run only — a mutation
primitive is scope breach). CPU-first, lattice listings.
Determinism: same debugging session twice → identical tables
(byte-digests; flagship 24×32 verified 3/3 bit-identical
pre-round, so strictness is earned, not hoped).

## Lesion semantics

A lesion answers "what does the output owe this op": zero its
OUTs, rerun, diff the final OUTs. Zeros are zeros_like the
reference values (per-component for triples tuples) — same
shapes/dtypes/kinds, so kind checks pass and only VALUES change.
Deltas in float64 (maxabs for the worst pixel, mean for the mass).
Zero-op control (empty overrides) must read all-zero deltas —
the tripwire that the rig measures the lesion, not itself.

## Example table (flagship 24x32 synthetic rgb — `lesion_table()`)

```text
op_idx op           outs            worst_maxabs sink:maxabs,mean...
     0 SRGB_DECODE  LIN             255          OUT:255,128.9
     1 LUMA         Y               83           OUT:83,36.36
     2 SQRT         A               83           OUT:83,36.36
     3 SPLAT_BLUR   AS,COH          47           OUT:47,23.05
     4 SUB          D               26           OUT:26,11.06
     5 BETA_V5      BEFF            26           OUT:26,11.06
     6 MUL          BD              26           OUT:26,11.06
     7 ADD          AE              83           OUT:83,36.36
     8 SQUARE       YENH            83           OUT:83,36.36
     9 GAIN         G               255          OUT:255,128.9
    10 SRGB_ENCODE  OUT             255          OUT:255,128.9
control: OUT maxabs 0.0 mean 0.0 (zero-op tripwire)
self-diff: identical (first None)
```

Read: the output owes 255 to the format edges (decode/encode,
gain), 83 to the luma trunk, 47 to the splat field, 26 to the
detail chain — every op's causal share, measured not asserted.
Gate `tests/test_debugger.py` (CPU, ~1s): step-values byte-equal
to direct `run()` at every stop, 11/11 lesionable with finite
positive deltas, control exactly zero, self-diff identical,
perturbed-IN first-divergence at op 0, session-twice identical
tables, 7 fail-loud legs. Neighbors green: `test_asm.py`,
`test_census.py` (shared read paths, untouched).
Bugs caught while building (mechanisms): (1) hook-env snapshots
miss IN streams on paused runs (inputs live in run()'s internal
feeds, stranded by the pause exception) — pre-seed env from the
payload rule, shadowed streams still read latest via write
order. (2) Test compared intermediate-stop watches against
FINAL ref values — watches must expect MISSING for unproduced
streams (prefix-correct expectation, sources from census).
Gap named: O(n²) stepping (fresh bounded run per step) + no
served/CUDA stepping — fine for small lattice listings, stated.

## Teacher-side hooks (named, not built)

What teacher debugging would need: per-layer activation capture
(twin forward hooks, no_grad), prompt-override interventions
(the overrides analogue for teacher inputs), align-and-compare
vs banked teacher outputs with first-divergence reporting.
Named here so the shape is on record; none of it is built.

## Torch lesions (constitution, FROZEN pre-build)

Surface mirrors the listing semantics exactly (`chain/debugger.py`,
torch section — same module, not a second tool): `TorchSession`
over a fixed net + a reference `run()` returning named output
tensors. `lesion(module_path)` zeroes that LEAF module's outputs
via a forward hook (zeros_like — shapes/dtypes/kinds preserved,
only values change) and reruns; tabled as per-output
(maxabs, mean) deltas vs the reference run. Zero-module control
(empty hook set) must read exactly zero. Same session twice →
identical tables. Bad module paths refuse loud (no silent
skips — every leaf lesionable is gated, not claimed).
Validation targets, frozen: mscale64 + jattn64b2 BANKED weights
(read-only out-of-tree `~/student_runs/<tag>/weights.pt`,
`weights_only` load, eval + no_grad throughout — no training,
no weight writes, ever). Lesion tables banked per model
(`samples/marigold/lesion_<tag>.json`); shapes compared across
families with the open question stated, never assumed: do
lesion profiles differ the way failures do (MLP mixer vs joint
attention — uniform spread vs load-bearing blocks)?
Reference runs (frozen): mscale64 = full-frame MLP forward on
banked church fixture (stride math untouched — full frame, not
the stride sample); jattn64b2 = rgb-path encode+head on church
(Phase-8 shape: W64/B2/H1/s32/C4). CPU suffices (MLP/head
forwards are seconds on CPU — no GPU need established, none
used). Qwen and teacher weights never loaded (students only —
teacher lesion is the next round, explicitly not this one).

## Torch results (students, CPU seconds — `scripts/bank_student_lesions.py`)

mscale64 (church full-frame, 5 leaves, all fired; outputs ~3.6e5 —
blown-up like the arc's constant-attractor findings, so every
lesion ≈ total-kill and biases vanish relatively):

```text
module          fires worst_maxabs
0                   1 3.652e+05
1                   1 3.652e+05
2                   1 3.656e+05
3                   1 3.656e+05
4                   1 3.652e+05
control: exactly zero
```

jattn64b2 (rgb path, 22 leaves; unfired exactly {row, col,
dino_proj} — row/col BYPASSED (encode reads `.weight`
directly, never calls the modules), dino_proj genuinely unused
in rgb path; the fire-count tripwire tells them apart from
measured zeros):

```text
module            fires worst_maxabs
blocks.0.q/k/v/o    1   4.3-4.4e6
blocks.0.n1         1   3.3e6
blocks.0.m1/m2/n2   1   2.2-2.3e6
blocks.1.m1/m2/n2   1   2.6e6
blocks.1.q/k/v/o/n1 1   6.2-7.4e4
h1/h2/tok           1   4.4e6
row/col/dino_proj   0   0
control: exactly zero
```

Profiles compared (the open question, answered): YES, they
differ the way failures do. mscale = uniform total-kill (chain
MLP, no residuals — every layer load-bearing, outputs
astronomical). jattn = differentiated (block-0 attention path
~1e6, head h1/h2 + tok ~4.4e6 load-bearing, block-1 qkv/o/n1
~70× smaller than block-1 MLP — the second attention block is
near-dead weight; only h1/h2 carry the head). The mixer fails
as one fused rope; the attention stack fails in named places.
Gate `tests/test_debugger.py` extended in place (CPU ~9s):
lesion-every-module, control-zero ×2, live==banked both models
(cross-process determinism), session-twice identical,
container/root/bad-path fail-loud, dino_proj exact-zero,
unfired-set exact. Listing legs green unmodified (shared code
untouched behaviorally — proven by the unchanged legs, not
claimed). Neighbors green: `test_asm.py`, `test_census.py`.
Bugs caught while building (mechanisms): (1) mscale rows
printed identically at 4 sig figs — full precision shows
bias-scale differences (3.652e5 vs 3.656e5); the model IS blown
up, the printer hid it. (2) row/col zeros are hook-BYPASS
artifacts (encode reads `.weight`, never calls forward), NOT
causal findings — fire counts added to every row; the gate pins
the unfired set exactly. (3) Test helper `sess()` shadowed by a
loop variable of the same name (UnboundLocalError) — renamed.

## What teacher lesion will need

Per-layer hooks on the teacher twin (no_grad, eval), a
reference run returning named decisions (argmax-rank/fluency —
serving grades decisions, never exact strings, per house
contract), banked teacher outputs to diff against, and the same
fire-count tripwire (dead vs bypassed). Qwen-7B stays unloaded
until that round's constitution says otherwise.

## Teacher lesion Phase A — block sweep, causal table (constitution, FROZEN pre-run)

Target list frozen (exact paths, no "and neighbors" mid-round):
- 60 × `transformer_blocks.{0..59}` whole-block zero (diffusers
  QwenImageTransformer2DModel, 60×3072 — banked blueprint).
- VAE-decoder stages (9): `decoder.{conv_in, up_blocks.0..3,
  mid_block, norm_out, conv_out}` + `post_quant_conv` (grouped
  from the 108 banked vae_pairs — full-weight tuned, not LoRA).
- Top-5 tuned families by banked rel LoRA-delta, GROUP lesions
  (zero across all 60 blocks at once — instance-level would be
  300 runs, out of budget, stated): `attn.add_q_proj` (0.0558),
  `attn.to_add_out` (0.0492), `attn.add_v_proj` (0.0455),
  `attn.add_k_proj` (0.0452), `txt_mlp.net.0.proj` (0.0436).
Baseline first: unlesioned teacher AbsRel on the eval set,
banked before any lesion runs. Eval set: 8 Hypersim frames
(first 2 available of ai_018_001, ai_021_001, ai_022_001,
ai_023_005 — deterministic, no cherry-picking), GT-anchored
(masks & valid intersect — no teacher-mimicry circularity;
webcam fixtures excluded: no GT there, stated reason).
Metric: AbsRel-delta per lesion vs baseline, relative %
((les−base)/base); alignment lstsq (same doctrine as every
student round + Marigold's own lstsq-scale recipe — RANSAC
unjustified here, not run). Removal criterion frozen: delta <
5% nominates removable (below single-seed noise family; the
criterion judges, Phase C validates).
Harness (read-only sibling use): custom loader mirroring their
component path (4-bit NF4 transformer + LoRA trainables +
bf16 VAE + cached prompt embeds — never the text encoder) +
encode → ONE flow step (t=0.499, velocity) → decode →
channel-mean (mirroring QwenImageEncode/2509Step/Decode/
SelectChannel node-for-node) + seeded VAE sampling. PARITY GATE
first: custom-unlesioned == sibling-CLI output on 2 frames
(else stop — cross-harness deltas are incomparable, no rescue).
Lesions ride forward hooks (zeros_like block outputs);
peft/quant wrappers respected (fire-count tripwire carried
from student lesions — bypassed vs dead distinguished).
Budget: per-eval wall measured on 2 pilot blocks FIRST (early
idx 3 + late idx 56 — not 0/59: edges are atypical, stated),
then total = per-eval × 75 configs (1 baseline + 60 + 9 + 5);
proceed iff projected ≤ 12h else shrink the eval subset first
(documented fallback) or stop with pilot evidence (timeout
precedent). GPU solo (idle + VRAM stated at run); Qwen-7B never
co-loaded (VRAM math: ~10GB 4-bit DiT + ~2GB VAE + embeds +
activations « 24GB; text encoder and Qwen-7B stay on disk).
Weights read-only out-of-tree (lesion at runtime — a single
modified weight file voids the round; sha256 of the 7 files —
5 transformer shards + trainables + VAE safetensors — banked
pre AND post, gate compares).
Ruling: hooks only; no emitter/opcode/assembly changes (gap
hit = stop-finding); no new deps (torch/diffusers/peft live in
marigold-env); production weights untouched (teacher AND
Qwen-7B); no downloads (shard suffices).
Gate: new tests/test_teacher_lesion.py (CPU-fast legs +
banked-table integrity; heavy reruns NOT routine — checksums,
schema, baseline-recompute-from-banked, delta arithmetic
re-derived in-gate). Neighbors: anatomy + teacher gates green
(read-only reuse).
Spec: this section + baseline + per-block delta table +
removable nominations per the frozen criterion — nothing more.

### Phase-A results (TBD — pilots first, then the run)

## Teacher lesion Phase A — block sweep, causal table (constitution, FROZEN pre-run)

Target list frozen (exact paths, no "and neighbors" mid-round):
- 60 × `transformer_blocks.{0..59}` whole-block zero (diffusers
  QwenImageTransformer2DModel, 60×3072 — banked blueprint).
- VAE-decoder stages (9): `decoder.{conv_in, up_blocks.0..3,
  mid_block, norm_out, conv_out}` + `post_quant_conv` (grouped
  from the 108 banked vae_pairs — full-weight tuned, not LoRA).
- Top-5 tuned families by banked rel LoRA-delta, GROUP lesions
  (zero across all 60 blocks at once — instance-level would be
  300 runs, out of budget, stated): `attn.add_q_proj` (0.0558),
  `attn.to_add_out` (0.0492), `attn.add_v_proj` (0.0455),
  `attn.add_k_proj` (0.0452), `txt_mlp.net.0.proj` (0.0436).
Baseline first: unlesioned teacher AbsRel on the eval set,
banked before any lesion runs. Eval set: 8 Hypersim frames
(first 2 available of ai_018_001, ai_021_001, ai_022_001,
ai_023_005 — deterministic, no cherry-picking), GT-anchored
(masks & valid intersect — no teacher-mimicry circularity;
webcam fixtures excluded: no GT there, stated reason).
Metric: AbsRel-delta per lesion vs baseline, relative %
((les−base)/base); alignment lstsq (same doctrine as every
student round + Marigold's own lstsq-scale recipe — RANSAC
unjustified here, not run). Removal criterion frozen: delta <
5% nominates removable (below single-seed noise family; the
criterion judges, Phase C validates).
Harness (read-only sibling use — weight FILES + cached prompt
files only, zero sibling imports): custom loader mirroring
their component path (4-bit NF4 transformer + LoRA r128/a128
trainables + bf16 VAE + VAE full weights + cached realimg512
prompt embeds — never the text encoder) + RGB[-1,1]/lanczos to
768×576 (their Hypersim res) → encode → ONE flow step
(t=0.499, velocity) → decode → channel-mean → Bilinear up to
GT res (mirroring QwenImageEncode/2509Step/Decode/SelectChannel
node-for-node) + seeded (2025) VAE sampling. PARITY GATE
first: custom-unlesioned vs sibling-CLI .npy on 2 frames ≥60dB
(else stop — cross-harness deltas are incomparable, no rescue).
Lesions ride forward hooks (zeros_like block outputs);
peft/quant wrappers respected (fire-count tripwire carried
from student lesions — bypassed vs dead distinguished).
Budget: per-eval wall measured on 2 pilot blocks FIRST (early
idx 3 + late idx 56 — not 0/59: edges are atypical, stated),
then total = per-eval × 75 configs (1 baseline + 60 + 9 + 5);
proceed iff projected ≤ 12h else shrink the eval subset first
(documented fallback) or stop with pilot evidence (timeout
precedent). GPU solo (idle + VRAM stated at run); Qwen-7B never
co-loaded (VRAM math: ~10GB 4-bit DiT + ~2GB VAE + embeds +
activations « 24GB; text encoder and Qwen-7B stay on disk).
Weights read-only out-of-tree (lesion at runtime — a single
modified weight file voids the round; sha256 of the 7 files —
5 transformer shards + trainables + VAE safetensors — banked
pre AND post, gate compares; pre hashes landed before any
build, trainables hash matches banked anatomy provenance).
Ruling: hooks only; no emitter/opcode/assembly changes (gap
hit = stop-finding); no new deps (torch/diffusers/peft/bnb
live in marigold-env); production weights untouched (teacher
AND Qwen-7B); no downloads (shard suffices).
Gate: new tests/test_teacher_lesion.py (CPU-fast legs +
banked-table integrity; heavy reruns NOT routine — checksums,
schema, baseline-recompute-from-banked, delta arithmetic
re-derived in-gate). Neighbors: anatomy + teacher gates green
(read-only reuse).
Spec: this section + baseline + per-block delta table +
removable nominations per the frozen criterion — nothing more.

### Phase-A results (TBD — parity first, pilots, then the run)

### Phase-A correction (pre-run, nothing executed — Phase-19 precedent)

Parity bar revised ≥60dB → ≥40dB with reason: VAE latent sampling
consumes RNG, and RNG streams cannot be aligned across harnesses
(their validation RNG state at sample time depends on dataloader
internals — unknowable without tracing); seeded determinism holds
WITHIN each harness (their gate proves theirs; mine seeds per
frame). 40dB is the repo's own cross-implementation bar
(torch-twin-vs-CUDA) — systematic harness drift (wrong
normalization/resize/order) still fails it by orders of
magnitude, while VAE sampling noise passes inside it. Lesion
deltas are computed WITHIN one harness (lesioned vs unlesioned,
same RNG handling), so parity only guards systematic offset.
Frozen bar: custom-unlesioned vs sibling-CLI .npy ≥ 40dB on 2
frames, else stop.

### Phase-A correction (pre-lesion: baseline ran, zero lesions run)

Judge fixed with visualization proof: lstsq-align + AbsRel in
METRIC space scores ~1.0 on a near-perfect map (pasted trio:
pred structure matches GT wall-for-wall) because the
relative→metric relation isn't affine (disparity-like
compression) — the judge was broken, not the teacher. Fixed
judge mirrors THEIR protocol exactly: GT percentile-affine
normalized per image (p2/p98 → [-1,1] on valid pixels, their
NormalizeDepthPercentileAffine), PLAIN AbsRel per stratum
(their metric has no alignment — `_abs_relative_difference`),
masks & valid. Old baseline numbers discarded (same run,
re-judged? NO — re-run baseline under the fixed judge; nothing
downstream exists yet, so nothing is contaminated). Removal
criterion unchanged (5% relative — same family, healthier
scale). Everything else in the constitution stands.

### Phase-A results (1 baseline + 74 lesion evals, ~15 min wall, GPU solo)

Baseline banked first: mean AbsRel 0.2936 over 8 frames (per-frame
0.20–0.40). Per-eval wall ~10s (8×1s frames + overhead); total ≈
15 min — inside the 12h ceiling with two orders of magnitude to
spare, no subset shrink needed. VRAM 13.7GB/24GB (4-bit DiT +
bf16 VAE, text encoder + Qwen-7B never loaded).
Parity (custom vs sibling CLI, 2 frames): 46.7dB / 45.3dB ≥ 40dB
bar — harness validated, then trusted for deltas.
Per-block deltas, relative % (baseline 0.2936):

```text
 0:+153  1:+129  2:+197  3:+175  4:+140  5:+168  6:+92  7:+147  8:+166  9:+132
10:+106 11:+127 12:+123 13:+180 14:+152 15:+133 16:+154 17:+154 18:+166 19:+158
20:+118 21:+125 22:+160 23:+167 24:+159 25:+189 26:+126 27:+129 28:+123 29:+172
30:+141 31:+138 32:+140 33:+123 34:+160 35:+146 36:+141 37:+142 38:+180 39:+143
40:+150 41:+144 42:+141 43:+173 44:+154 45:+154 46:+149 47:+147 48:+156 49:+157
50:+189 51:+154 52:+156 53:+160 54:+140 55:+127 56:+142 57:+234 58:+139 59:+135
VAE: conv_in +84.7, up.0 +91.7, up.1 +132.3, up.2 +91.8, up.3 +94.9,
mid +98.2, norm_out +94.9, conv_out +95.2, post_quant +86.1
families: txt_mlp.net.0.proj +11.7, add_q +24.5, add_v +59.1,
add_k +102.2, to_add_out +110.6
```

Removable nominations per the frozen <5% criterion: NONE (min
+11.7% — txt_mlp family closest, still 2× over bar). Phase C
has nothing to validate (stated outcome, not an omission).
Verdict: the teacher is DENSE — every one of 60 blocks moves
AbsRel +92% to +234%; all VAE stages +84–132%; all 5 families
+12–111%. Structure inside the density: late blocks worst
(57: +234%, 50: +189%), block 6 least (+92%); attention-add
families cheaper than to_add_out; txt_mlp family cheapest of
all. Nothing here resembles the students' differentiated
pins — the teacher has no dead weight at block granularity.
Gate `tests/test_teacher_lesion.py` (CPU, 0.02s — heavy reruns
NOT routine): schema 60+9+5, fires (blocks 8/8, families
60×8), baseline recompute, nominations re-derived (0), sha256
pre==post over 7 files. Neighbors green: anatomy
(`test_teacher_anatomy.py`) + teacher (`test_teacher_marigold.py`)
— read-only reuse, nothing moves underneath.
Bugs caught while building (mechanisms): (1) timestep/dtype
break at the first block — `prepare_model_for_kbit_training`
upcasts non-quantized params to fp32 while their validation
runs under autocast-bf16; mirrored the eval context (no_grad +
autocast bf16) and the break vanished — the loader was right,
my eval context was wrong. (2) Prompt batch mismatch (their
node truncates embeds to batch, mine didn't) — mirrored
`_match_batch` verbatim including the truncate. (3) My judge
(lstsq-align + AbsRel in meters) scored ~1.0 on a
wall-for-wall-correct map (pasted trio) — the relative→metric
relation isn't affine; fixed to their protocol (GT
percentile-affine + plain AbsRel), old baseline discarded
pre-lesion, rerun banked. (4) VAE targets needed the VAE.
namespace prefix in matching (exact paths matched nothing) —
caught by --dry-run validation before the full run (74/74
resolve). (5) Fires banked per-frame-last instead of summed —
accumulated across the eval. (6) Smoke-test glob collided
across scenes (fr0000 × 4) — scene-qualified.
Gaps named: sub-block granularity (qkv vs MLP inside a block)
is Phase B; per-instance (not family) LoRA-target lesions
would be 300 runs — priced, not run.

## Teacher holography — subset-lesion curve (constitution, FROZEN pre-run)

Question: does teacher quality degrade GRACEFULLY (holographic —
every block carries a little) or CLIFF (modular — some subset is
load-bearing)? Phase A proved every SINGLE block matters
(+92–234%); this round lesioned RANDOM SUBSETS and reads the
curve shape.
Design frozen: subset sizes {0, 5, 10, 20, 30, 40} of 60 (0 =
unlesioned baseline RE-BANKED in this round, not inherited —
guards harness drift between rounds), 3 seeds per nonzero size
(subset RNG: `random.Random(28000 + size*100 + rep).sample(
range(60), size)`, rep ∈ {0,1,2} — stdlib RNG only, torch/numpy
streams untouched so inference RNG matches Phase A exactly).
16 configs total (1 + 15). VAE untouched (DiT-block question).
Metric: mean AbsRel over the Phase-A fixture set (same 8
Hypersim frames, same relative-space judge — comparability
with Phase A is the point). Fired-block sets banked per row
(integrity: row claims match hook counts).
Verdict rule frozen: LINEAR-ish degradation (least-squares fit
of mean-vs-size over the 6 sizes: R² > 0.8 AND no single step
>2× the previous size's mean = holographic) vs CLIFF (any step
>2× = modular, not holographic). Seed-agreement gate: max/min
across 3 seeds per size < 1.5× (subsets shouldn't wildly
disagree). Characterization round — no removal nominations,
no Phase-C anything.
Budget: per-eval wall KNOWN (1s/frame from Phase A — no new
pilots needed, stated reason): 16 configs × 8 frames × ~1s +
load ≈ 5 min. GPU solo (idle + VRAM stated at run); Qwen-7B
never co-loaded (~14GB footprint as measured); no downloads.
Weights read-only out-of-tree (checksums pre/post in-gate,
Phase-A precedent). Ruling: hooks only; no emitter/opcode/
assembly changes; no new deps; production weights untouched.
Gate: extend tests/test_teacher_lesion.py in place
(subset-curve legs: schema, linear-fit R², cliff check,
seed-agreement — same integrity shape as Phase A; heavy
reruns NOT routine). Neighbors: lesion + anatomy + teacher
gates green (read-only reuse).
Spec: this section + curve table + verdict (holographic or
modular) — nothing more.

### Holography results (TBD — run lands here)

### Holography results (1 baseline re-bank + 15 subset evals, ~6 min wall, GPU solo)

Baseline re-banked in-round: mean 0.2936 (EXACT match to Phase-A
0.2936 to 4 decimals — cross-round reproducibility, same harness/
seed/frames; determinism corroborated, not just claimed).
Curve (size: rep0 rep1 rep2 | mean | seed max/min | step-ratio):

```text
 0: baseline                                   0.2936
 5: 0.7241 0.6902 0.6902 | mean 0.7015 | 1.049 | x2.39
10: 0.9813 0.7458 0.7458 | mean 0.8243 | 1.316 | x1.18
20: 0.6902 0.6902 0.9813 | mean 0.7872 | 1.422 | x0.96
30: 0.6902 0.7103 0.7458 | mean 0.7154 | 1.081 | x0.91
40: 0.6902 0.6902 0.6902 | mean 0.6902 | 1.000 | x0.96
linear fit: slope +0.0053, R² = 0.1860 (needs >0.8)
```

Frozen-rule verdict: **modular** (R² 0.186 fails holographic;
0→5 step ×2.39 fails no-cliff — doubly determined).
Diagnosis — the curve is NOT a dosage response, it's attractor
membership: 8 configs snap to ONE bit-identical per-frame
output (0.6902 attractor), and subset_10_0 (contains block 57 +
9 others) reproduces Phase-A block-57-alone per-frame EXACTLY —
block 57 totally dominates its subset (9 extra dead blocks
change nothing). Lesioned DiT falls into a few discrete
degenerate states rather than decaying gracefully: no dosage,
no hologram. Consistent with Phase A (every block load-bearing
alone) and sharper: not only does each block matter, but
beyond ~5 dead blocks the model stops degrading FURTHER —
collapse saturates at shared attractors. Seed agreement holds
(max/min ≤ 1.422 < 1.5 — seeds agree the curve is flat
garbage, not signal).
What this closes: graceful-degradation/holographic storage for
this teacher at block granularity. What it doesn't touch:
sub-block structure (Phase B), VAE/family subsets (unrun),
recovery dynamics (can the model re-converge post-lesion?
untrained here — training is out of scope).
Gate `tests/test_teacher_lesion.py` extended in place (CPU,
0.06s): schema (15 rows, frozen subsets re-derived from the
frozen RNG scheme), finite, R²/cliff/seeds recomputed,
verdict equality. Neighbors green: lesion (Phase-A legs),
anatomy, teacher — read-only reuse, nothing moves underneath.
Bugs caught while building (mechanisms): (1) gate `_cm`
assumed wrapped strata-dicts; banked frames are flat (fixed).
(2) Step loop off-by-one (6 points → 5 steps, not 6).
Gaps named: none new — the round is self-closing by design.

## Attractor census — map the basins, compare against graceful (constitution, FROZEN pre-run)

Basin census over the 15 holography subsets (+ unlesioned
reference): which subsets fall into which basins, where basin =
identical-output equivalence class (sha256 of model-res float32
pred bytes, concatenated over the 8 eval frames — counted
exactly like the 8-way tie, no thresholds). Reruns bank map
digests + per-frame stats (mean/std at model res; corr vs GT
and vs unlesioned at native res, computed in-run) into NEW
`teacher_basins_H.json` (banked lesion files NEVER overwritten).
New lesion runs needed (maps weren't banked): 15 subsets +
baseline rerun + 3 same-seed stability reruns (subsets 5_0,
20_2, 40_1 — spread across the curve), ~19 evals ≈ 4 min GPU
solo (known 1s/frame; cap 60 min). GPU need stated pre-run
here: 20B 4-bit DiT + VAE (same ~14GB footprint as Phase A;
CPU cannot serve it — no CPU path exists, stated not assumed).
Characterize each basin (frozen rules): constant map?
(model-res std ≈ 0, tolerance 1e-6); scene mean? (|basin mean
− GT mean| < 5% of GT mean); else structured remnant (report
corr-vs-unlesioned, no threshold asserted). Stability frozen
(two legs): (i) per-frame partition consistency — subsets tied
overall must tie on EVERY frame (else the basin is an
averaging artifact, reported as such); (ii) rerun determinism —
3 stability reruns land in their banked basins exactly.
Yarnball comparison (frozen sources, stated deviation from the
letter): NO single banked "ball degradation curve" exists on
`research` (verified by grep: only passing "graceful" mentions
in retrieval-store notes + an lm_dualhead comment — pasted in
build notes; the diagonals branches hold seam-exactness proofs,
not damage curves). The comparison therefore reads the nearest
banked graceful-degradation evidence, named here so the gap is
checkable: flagship listing lesion profile (debugger round —
graded shares 255/83/47/26, every op a continuous share),
mscale64 student lesions (uniform total-kill 3.65e5 — brittle),
jattn64b2 lesions (70× differentiated). Numbers cited, never
adjectives. Dedup/search claims untouched (out of scope).
Winner rule: N/A (characterization — completeness = every
banked subset in exactly one basin + stability green; empty
basins reported, not hidden).
Ruling: hooks only (`--mapstats` additive flag on the existing
harness — same tool); no training/emitter/opcode/assembly/deps;
weights read-only out-of-tree (checksums discipline as
established — sha re-verified post-run, any drift voids);
production weights untouched (teacher AND Qwen-7B never
co-loaded); no downloads; `research` read-only via git show
(never checkout/merge/push — verified above without touching).
Gate: extend tests/test_teacher_lesion.py in place
(basin-census legs: partition integrity — every subset in
exactly one basin, basins pairwise distinct; stability legs;
yarnball-comparison legs reading banked numbers from both
sides). Neighbors: lesion + anatomy + teacher gates green
(read-only reuse).
Spec: this section + basin catalog with characterization +
graceful-vs-snap comparison table + what the architecture
implies for distillation (stated as implication, not plan).

### Attractor results (TBD — reruns land here)
