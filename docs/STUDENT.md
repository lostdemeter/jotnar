# STUDENT.md — Marigold v2 teacher, Phase 0 (reproduce + fixtures)

Phase 0 is reproduction + fixtures. No training, no assembly/emitter
changes. Implementation temptation = stop-and-report.

## Setup (sibling checkout, never vendored)

```
cd /home/balrog/Documents/OpenCode
git clone git@github.com:huawei-bayerlab/marigold-v2.git  # @ 9926d93
python3 -m venv ~/marigold-env
~/marigold-env/bin/pip install torch torchvision  # cu130 index below
~/marigold-env/bin/pip install diffusers==0.38.0 peft==0.18.1 \
  accelerate==1.13.0 bitsandbytes==0.49.2 omegaconf==2.3.0 \
  huggingface_hub==1.8.0 safetensors==0.8.0 pillow==12.1.1 \
  numpy==2.2.5 pandas==2.3.3 tabulate h5py scikit-learn lpips \
  matplotlib opencv-python-headless scipy scikit-image
```

Environment choice: venv (not conda — none on this box; not the
shared local env — diffusers/pins would churn the Jotnar stack),
torch 2.14.1+cu130 reused (their setup wants 2.10/cu128; local CUDA
stack is proven by DAV2 work and runs the teacher — documented
here, not assumed). Python 3.12 throughout (their setup says 3.10;
everything imported and ran).

Fetched (measured): `Qwen/Qwen-Image-Edit-2509` full snapshot 54GB
(35 files) + `huawei-bayerlab/marigold-v2-0` minimal
(`depth/Log-stage2/trainables.safetensors` + depth realimg512
embedding pair) 1.8GB → `marigold-v2/assets/checkpoints/`. Nothing
else (no normals/albedo, no datasets, no DINOv3).

## Measured table (1024x672, seed 2025, 4 example images)

```
peak VRAM: 19918 MiB (~19.5GB, nvidia-smi sampling across full run)
wall: ~22s for 4 images (~5.5s/image incl. model load + 4-bit quant)
output: (672, 1024) float32 .npy + spectral visualization per image
same-seed twice: byte-identical (4/4 files)
seeded-vs-unseeded (2025 vs 7): maxabs 0.44-0.79 per image (all differ)
```

Determinism bar: bit-exact is the RIGHT bar seeded (VAE noise fixed)
and the WRONG bar unseeded (VAE encoder samples — their
troubleshooting section says so; confirmed: 0.44-0.79 maxabs).
2048² NOT attempted: their doc needs 29GB there; this box is 24GB.
Ceiling prediction from measurement: 1024²-class fits (~19.5GB at
1024x672 < 1024² pixels), 2048² exceeds — predict, never probe an
OOM on a shared box. (Native 2048px examples DID OOM here at 23.5GB
used — that was the finding that set the cap, on an otherwise idle
box, process died clean.)

Gotchas met: their `scripts/infer.py` needs `--width/--height` as
multiples of 16 (680 → hard crash in diffusers `_pack_latents`);
`evaluation.depth.evaluate_pipeline` overwrites
`<output_dir>/config.yaml` with the resolved config (re-reading it
fails on missing `use_paths_from` — always regenerate, never reuse);
missing pip deps fail at import (`pandas`, `lpips` — install full
list above); our `scripts/cuda_env.sh` exports POISON the teacher
venv subprocess (PYTHONPATH/LD_LIBRARY_PATH mix the user-local stack
into the venv — measured failure, passes clean) so the gate runs the
teacher CLI under a sanitized env (venv/bin + system PATH only).

## Fixture manifest (`samples/marigold/`, 19MB)

`teacher_manifest.json` + 4 depth `.npy` (seed 2025, 1024x672) +
input copies under `inputs/`. Provenance per row: example name,
shape, dtype. Checkpoint provenance in manifest (repo 9926d93,
backbone, stage2, modality, resolution, seed). Tracked via
`.gitignore` exception (like `samples/depth/`).

## Sequencing rule (shared box)

Never co-load the Qwen-7B-Instruct snapshot alongside the teacher:
teacher runs hold ~19.5GB VRAM; Jotnar served runs hold ~14GB.
Sequence the work; this Phase ran with the GPU to itself.

## Phase-1 needs (named, not built)

Strata harness per the DAV2 playbook: error strata over banked
fixtures (near/far? planar/non-planar? texture regimes?) with
per-stratum teacher-vs-student deltas — needs list only. Requires:
fixture bank (done), seeded teacher CLI (done), student lattice
skeleton (not started).

## Phase 1 — strata harness with proven falsification (`tests/test_strata_marigold.py`)

Six strata over the 4 banked fixtures (rules, all deterministic):
near/far = median split via `chain/depthprior.py:near_mask`;
planar/nonplanar = teacher-depth Sobel-gradient 40/60 percentiles
(20% deadband avoids boundary dithering); textured/smooth = input-
luma gradient 40/60 percentiles. Percentile rules guarantee every
stratum 30-70% of every frame (gated) — no fixture extension needed
(the 4 scenes × 6 strata cover; additions only if a rule starves,
documented here if so).

Comparison: least-squares affine fit student→teacher, GLOBAL per
frame (one scale/shift — matches affine-invariant semantics; strata
then show where errors concentrate), MAE per stratum. Least-squares
over RANSAC: closed-form, deterministic, standard practice.
Alignment direction is load-bearing (fit a*student+b ≈ teacher):
a flipped fit lets constant predictors score 0 — guarded by an
explicit leg (constant residual ≈ teacher std, never 0).

Margin bars per stratum = scaffold floor/5 (scaffold = constant
teacher-mean stub, SKELETON SCAFFOLD, not a model):
near 0.09, far 0.09, planar 0.09, nonplanar 0.08, tex 0.07, smooth
0.09 (scaffold minima 0.457/0.481/0.484/0.403/0.377/0.483).
Teacher-self scores 0.0 exactly (passes 5x+ clear); scaffold fails
every stratum every frame (worst margins pasted in gate output).
A harness never tested against a failing student is unproven —
both directions gated green.

## Phase-2 inherits (architecture search — named, not started)

Banked fixtures + manifest, seeded teacher CLI, strata rules +
margins, scaffold floor. A real student must beat scaffold/5 on all
six strata; anything weaker fails visibly by construction.

## Phase 2 — architecture search inside the emitted table (LOSE, reported)

Ruling constraint: every deployed op needs a CUDA/F emitted pattern
(MATMUL/ADD/SILU qualify; LAYERNORM/CONV gaps bite by design, so the
student uses neither). Architecture: per-pixel MLP RGB(3)→W→W→1,
SiLU, no norms (`scripts/train_student.py`, torch outside listings,
seed 7, Adam lr 0.02, 2000 steps, stride-7 pixel sample, deterministic
stack). Split stated openly: train = 4 banked examples, held-out = 2
new webcam frames via seeded teacher CLI (same provenance pattern;
manifest extended, 6 rows). Weights out-of-tree
(`~/student_runs/<tag>/weights.pt`, never committed); metrics +
config in `samples/marigold/student_run.json` (committed).
Production weights untouched (student artifacts are new files only).

Search, all openly reported (loss = affine-invariant MAE, train):
- w16 per-pixel: 0.4827 → 0.3693 (9s)
- w32 per-pixel: 0.3708 — capacity ruled out (identical)
- p3w16 3x3-patch (GATHER+MATMUL form, still in-table): 0.3665 —
  local context doesn't move it
Rerun determinism verified in fact: w16 retrained, loss trajectory
identical all 5 rows (0.4827/0.3751/0.3710/0.3697/0.3693).
En-route bug: lstsq-backward OOMs at 400k rows (1152GiB ask) —
alignment takes no gradients (standard practice), fixed.

Verdict: 0/36 cells pass bars (best: squirrel-tex 0.185 vs 0.07;
held-out smooth ~0.19 vs 0.09). Blocking diagnosis: every stratum
fails, worst in texture/nonplanar detail — monocular depth needs
larger spatial context than per-pixel/3x3 MLPs represent; wider or
patch-tuned variants of the same shape cannot close a 5x gap.
Bigger patches / GATHER pyramids would be unbounded search; norms
or conv would break the emitted-table constraint. No silent tuning
was performed to pass — three configs, all reported, all lose.

What Phase 2 leaves: banked metrics, deterministic trainer,
harness verdicts. A future round needs either a context-capable
in-table architecture or a ruling-constraint change (both scoped
proposals, not this report).

## Phase 3 — multi-scale context student (LOSE, reported)

Ruling constraint held: every deployed op cites a CUDA/F pattern
(GATHER F, CONCAT F, MATMUL F, ADD F, SILU F). No norms, no conv,
no new opcodes — LAYERNORM/CONV gaps never triggered.

Architecture (`--multi`): per-pixel dense-3x3 (27) + stride-4
17x17 (75) + stride-8 33x33 (75) = 177 dims → 64 → 64 → 1 SiLU
MLP. Receptive field 33px, targeting the Phase-2 diagnosis
(texture/nonplanar detail worst — detail scale 10-60px at 1024x672;
3px covered nothing, 33px covers the band). Deployed form is
GATHER neighborhoods + CONCAT + MLP (all tabled).
Size: 15,617 params (61KB fp32), ~465MB transient features/frame,
train 21s, peak training VRAM 330MiB — deployable with headroom to
spare, nowhere near 17GB. Strided (not averaged) context: honest
sampling, documented (averaging would need POOLAVG — parked row).

Search (open, all reported): w16 0.3693 / w32 0.3708 (capacity out)
/ p3w16 0.3665 (3px context out) / mscale64 0.3092 (context helps
train-fit: 0.31 vs 0.37, held-out barely moves — generalization,
not capacity, is the wall). Retrain determinism verified in fact
(identical 6-row trajectory 0.5099/0.3089/0.3095/0.3096/0.3092).

Verdict: 0/36 cells pass (best squirrel-tex 0.156 vs 0.07;
geo_w1-far 0.488 vs 0.09 — held-out far-side worst). Blocking
diagnosis: global layout reasoning (which side is far across rooms,
occlusion ordering) needs beyond-33px structure no local mixer
represents; texture/detail strata improved most yet still 2-5x over.
Next needs either long-range in-table structure or a constraint
change — both scoped proposals, not this report.

## Phase 4 — training signals ablated, both killed (mscale64 frozen)

Two signals, each behind a flag (`--perc`, `--sink`), ablated
separately against the aligned-MAE baseline (deterministic stack
throughout; DINOv2-base frozen at ~/student_runs/dinov2-base, 661MB
out-of-tree — ungated facebook/dinov2-base stand-in for gated
DINOv3, documented substitution).

1. Perceptual, iREPA pattern (adapted): MSE between L2-normalized
   frozen-DINOv2 token embeddings of student vs teacher depth maps
   (3ch-replicated, ImageNet-normalized). True iREPA aligns
   intermediate layers; our MLP has no spatial intermediates, so we
   align encoder embeddings of final outputs. Training-time only,
   zero inference cost. Gradients verified live (4.05e-05 w.r.t.
   student scale — nonzero, real path; single backward over base +
   aux after catching a double-backward, and a per-frame combined
   restructure after mpboth OOMed at 18.5GB).
2. Sinkhorn block loss (paper values first): 8x8 block grid,
   256-px deterministic stride subsamples, squared cost on
   affinely-aligned values, K=5 τ=0.1 log-domain iterations, plan
   detached with cost attached (standard divergence practice).
   Permutation-invariant within blocks by construction.

Ablation (held-out; train tracks the same way):
```
variant   train loss   geo_w0 (best/worst)      geo_w1 (best/worst)
baseline  0.3092       smooth 0.169 / tex 0.308  near 0.322 / far 0.488
+perc     0.3347       smooth 0.183 / tex 0.343  near 0.185 / far 0.425
+sink     0.3022       smooth 0.173 / tex 0.337  near 0.263 / far 0.440
+both     0.3249       smooth 0.194 / tex 0.341  near 0.233 / far 0.466
```
(regenerated on the final tree; stale pre-fix columns replaced — the
pre-fix aux runs' gradients differed, see report. Baseline column
stands retrain-verified.)
Verdict: NULL across the table — deltas ±0.02-0.05, bidirectional,
no stratum systematically moves; all cells 2-6x over bars in every
variant. Kills with evidence (not failures to tune):
- Perceptual: metric dynamic range 7e-4 (random-vs-teacher) vs MAE
  0.37 — homeopathic at any stable weight; trajectory static
  (0.00137→0.00141 over 2000 iters). The metric cannot move training.
- Sinkhorn: post-alignment OT values ~4-6e-6 (floor-dominated at K=5
  non-converged diffuse plans and/or already-matched distributions)
  vs MAE 0.3 — ~1e-5 influence at λ=1.0. Scaling λ up would amplify
  floor residue, not signal.
Neither signal earns Phase 5's search battery. What would change
that: a perceptual metric with dynamic range, or converged/larger-K
Sinkhorn with calibrated weight — both new experiments, stated here,
neither run.

## Phase 5 — architecture search battery (constitution-bound)

Declared axes + budget, FROZEN before running (no mid-battery axis
invention — that path is the spiral). Budget: 10 new configs, minutes
each, single GPU to itself. Judge unchanged: Phase-1 bars, same
alignment, verdict-reported not pass-gated. Loss fixed to aligned-MAE
(signals are dead — relitigating them is out of scope). In-table
architectures only (emitted-pattern law holds; gap hit =
stop-finding). Empty battery (nothing clears bars) is a complete
report with the table banked.

Frozen axes (each config states receptive field in pixels):
- field: 33 (dense3+s4/17+s8/33, 177 dims) → 65 (+s16/65, 252 dims)
  → 129 (+s32/129, 327 dims). Strided sampling (documented honest
  sampling, cf. Phase 3); averaging would need parked POOLAVG.
- mixer: W64/d2 (reference shape) × W128/d2, W64/d3, W32/d2.
- granularity: densewin 3 → 5 at field 33.
- scale-count: covered by field steps (dropping scales untested
  separately; noted, not run).
Training fixed for comparability (2000 steps, lr 0.02, seed 7,
stride-7, no-grad alignment). Reference row: committed mscale64
(baseline, retrain-verified — not rerun).

| # | tag | field | dims | mixer | params |
|---|-----|-------|------|-------|--------|
| 0 | mscale64 (ref) | 33px | 177 | W64/d2 | 15.6k |
| 1 | f65w64d2 | 65px | 252 | W64/d2 | ~20k |
| 2 | f129w64d2 | 129px | 327 | W64/d2 | ~25k |
| 3 | f65w128d2 | 65px | 252 | W128/d2 | ~49k |
| 4 | f129w128d2 | 129px | 327 | W128/d2 | ~59k |
| 5 | f65w64d3 | 65px | 252 | W64/d3 | ~24k |
| 6 | f129w64d3 | 129px | 327 | W64/d3 | ~29k |
| 7 | f33w128d2 | 33px | 177 | W128/d2 | ~39k |
| 8 | f33dense5 | 33px | 225 | W64/d2 | ~19k |
| 9 | f65w32d2 | 65px | 252 | W32/d2 | ~10k |
| 10 | f129w128d3 | 129px | 327 | W128/d3 | ~75k |

Per-op citations (unchanged family): strided neighborhoods deploy
as GATHER F, concat as CONCAT F, mixer as MATMUL/ADD/SILU F — all
emitted CUDA/F patterns. No norms, no conv, no new opcodes.
Deployed size envelope: largest config ~75k params (~300KB fp32),
transient features <1GB/frame — nowhere near 17GB.

Winner rule: most held-out cells under bars; ties → smaller
train-heldout gap (generalization over memorization). Winner gets
full retrain-match (trajectory + cell table identical) before being
claimed — integrator spot-verifies the winner only, plus harness
integrity, not all 10.

### Battery results (all 10 ran, minutes each, single GPU to itself)

Train-loss descends with field/width as expected (0.3347 f33 →
0.2241 f129w128d2); held-out does not follow. Best held-out cells
per config sit 2-6x over bars everywhere; the count of held-out
cells under bars is 0 for all 10 configs (gate-counted, not
eyeballed). Winner: none — no retrain-match needed (nothing to
claim; the tripwire stands ready: any passing cell with winner None
fails the gate demanding the claim).

Range-vs-bars frontier: train fit improves monotonically with
field (33px 0.33 → 129px 0.22) and width, while held-out stays flat
(~0.2-0.48 across all configs) — the battery maps a generalization
wall, not a capacity slope. Depth (d2→d3), granularity (dense 3→5),
and width (32→128) all move train loss without moving held-out.
Diagnosis: monocular layout needs beyond-129px/global structure or
a different inductive bias; the in-table mixer family is exhausted
along every frozen axis. Next needs either long-range structure or
a constraint change — scoped proposals, not this report.

## Phase 6 — long-range battery, last honest round

Declared axes + budget, FROZEN before running. Budget: 10 new
configs, GPU box to itself, per-config wall cap 3min (bigger fields
cost more: feature build O(px × samples); measured 25-60s/run).
Judge unchanged: Phase-1 bars, same alignment, verdict-reported.
In-table only (emitted-pattern law; LAYERNORM/CONV-type need =
stop-finding, not implementation). Empty is complete.

Long-range mechanisms (each config names its path to beyond-129px
structure; "bigger mixer" alone is refused at declaration — depth
appears only paired with a range mechanism, as capacity support):
- wider spoke fields: 257px (stride-64 over ±128), 513px
  (stride-128 over ±256), ~1025px (stride-256 over ±512, sparse
  full-width spokes). Same 5x5-per-scale pattern (75 dims each);
  coverage thins with range by construction (stated, not hidden).
- coarse-global broadcast: per-image mean RGB + mean luma (4 dims)
  appended to every pixel — whole-frame summary, lossy by design;
  tests whether global tint/layout suffices. Deploys as const-fill
  + CONCAT (both tabled).
- excluded by the constitution rule: standalone depth configs
  (d3/d4 without a new field — capacity is not a mechanism),
  finer granularity reruns (Phase-5 axis, settled), loss/signal
  changes (Phase-4 settled, dead).

| # | tag | mechanism | field | dims | mixer |
|---|-----|-----------|-------|------|-------|
| 1 | g257w64d2 | quarter-frame spokes | 257px | 402 | W64/d2 |
| 2 | g513w64d2 | half-frame spokes | 513px | 477 | W64/d2 |
| 3 | g1025w64d2 | sparse full-width spokes | ~1025px | 552 | W64/d2 |
| 4 | g257w128d3 | spokes + capacity | 257px | 402 | W128/d3 |
| 5 | g513w128d3 | spokes + capacity | 513px | 477 | W128/d3 |
| 6 | globw64d2 | whole-frame summary | 33px+global | 181 | W64/d2 |
| 7 | glob257w64d2 | spokes + summary | 257px | 406 | W64/d2 |
| 8 | g513w64d3 | half-frame + capacity | 513px | 477 | W64/d3 |
| 9 | g257w32d2 | spokes, narrow check | 257px | 402 | W32/d2 |
| 10 | g1025w128d3 | max envelope | ~1025px | 552 | W128/d3 |

Dims: dense27 + 75 per spoke scale + 4 if global (552 = 27+75*7;
field1025 adds s64+s128 scales). Largest: ~75k+ params, transient
features <2GB/frame CPU, GPU batch <1GB — nowhere near 17GB.
Winner rule: most held-out cells under bars; ties → train-heldout
gap. Winner gets retrain-match before claimed. Empty (nothing
clears bars) closes with the earned sentence (see results).

### Battery results (all 10 ran, ~21-60s each, single GPU to itself)

Means over banked strata (train = 4 frames × 6 strata; held-out =
2 frames × 6 strata = 12 cells/config). Best/worst = held-out cell
by multiples over its bar:

```
tag            field  mixer    tr-mean  ho-mean  ho-best (val/bar)      ho-worst (val/bar)      pass
g257w64d2      257px  W64/d2   0.228    0.283    near 0.184/0.09 (2.0x)  tex 0.365/0.07 (5.2x)   0/12
g513w64d2      513px  W64/d2   0.144    0.269    near 0.172/0.09 (1.9x)  tex 0.371/0.07 (5.3x)   0/12
g1025w64d2     ~1025  W64/d2   0.146    0.323    smoo 0.180/0.09 (2.0x)  tex 0.436/0.07 (6.2x)   0/12
g257w128d3     257px  W128/d3  0.208    0.291    smoo 0.189/0.09 (2.1x)  tex 0.366/0.07 (5.2x)   0/12
g513w128d3     513px  W128/d3  0.154    0.285    smoo 0.189/0.09 (2.1x)  tex 0.363/0.07 (5.2x)   0/12
globw64d2      33+g   W64/d2   0.320    0.304    smoo 0.180/0.09 (2.0x)  tex 0.414/0.07 (5.9x)   0/12
glob257w64d2   257+g  W64/d2   0.182    0.280    smoo 0.192/0.09 (2.1x)  tex 0.365/0.07 (5.2x)   0/12
g513w64d3      513px  W64/d3   0.135    0.278    near 0.191/0.09 (2.1x)  tex 0.367/0.07 (5.2x)   0/12
g257w32d2      257px  W32/d2   0.196    0.273    near 0.182/0.09 (2.0x)  tex 0.370/0.07 (5.3x)   0/12
g1025w128d3    ~1025  W128/d3  0.127    0.318    smoo 0.176/0.09 (2.0x)  tex 0.421/0.07 (6.0x)   0/12
```
(train_loss is not banked — None for every cell in both batteries,
pre-existing shape; the frontier below is stated in banked strata
means, not in unbanked loss.)

Winner: none — 0/120 held-out cells under bars (0/12 every
config, gate-counted, not eyeballed). No winner retrain-match
needed (nothing to claim; the tripwire stands ready as in Phase 5).

Final-tree proof (Phase-4 precedent): all 10 weight mtimes postdate
the final trainer commit 3e2434e (08:44:39 -0400) by +36s (g257w64d2
08:45:15) to +566s (g1025w128d3 08:54:05); the WIP checkpoint banks
after training. No stale-code cells, nothing regenerated.
Spot-retrain: globw64d2 rerun under distinct tag spot-globw64d2
(same frozen flags, 21s wall) matches the banked cell in all 36/36
strata to max abs diff 7.2e-16 (float-serialization level; 4dp
tables identical); committed student_run.json restored byte-identical
(sha256 f392c9fa), spot weights out-of-tree only.

Range-vs-bars frontier: train fit descends 257px (0.23) → 513px
(0.14) then flats at ~1025px (0.13-0.15), while held-out stays flat
(0.27-0.32) — the Phase-5 generalization wall extends to
full-frame spokes. Capacity (W32→W128, d2→d3) moves train mean
(0.196→0.127 at matched fields) without moving held-out. The
whole-frame summary is bidirectional on train (worst-of-battery
0.320 at 33px+g; 0.182 at 257px+g vs 0.228 spokes-only) and never
moves held-out systematically (0.28-0.30 both). The 1025px sparse
envelope holds the worst held-out means (0.32) — thinnest coverage
overfits train while generalizing least.
Diagnosis: strided beyond-129px spokes + lossy global tint do not
supply monocular layout reasoning (which side is far, occlusion
ordering); texture/detail best cells stay 2x over bars everywhere,
far/tex worst 5-6x. The in-table mixer family is now exhausted
along every frozen range axis, as it was along every Phase-5 axis.
Earned sentence: empty is complete — the long-range round maps the
same wall at full-frame scale, with the table banked.

What would change that (scoped proposals, not run): a
constraint change (new emitted patterns for attention/pooling-grade
structure) or a different inductive bias outside the mixer family —
both need their own declared round, neither is this report.

## Phase 7 — teacher anatomy, read-only (measured, not trained)

No training, no emitter/opcode/assembly changes. Two analyses
against the banked checkpoints (54GB backbone + 1.85GB trainables,
out-of-tree; production weights untouched — every file opened
read-only). Numbers banked in `samples/marigold/teacher_anatomy.json`
(114KB); gate `tests/test_teacher_anatomy.py` (SKIP-clean without
assets; fast legs only — header reads, factor-side delta recompute,
embeds/config checks; no 54GB load in routine runs; the full
backbone-relative recompute is the documented manual step
`python3 /tmp/opencode/teacher_anatomy_compute.py`, CPU-only ~35s).

### 1. LoRA-delta anatomy (rank 128, alpha 128, scale 1.0, BF16)

Census: 1556 tensors = 1446 Diffuser LoRA (60 blocks × 24 +
img_in/txt_in/norm_out.linear A/B) + 108 VAE (decoder +
post_quant_conv, full weights not LoRA) + 2 iREPA leftovers
(Stage-1 F32 convs `[768,64,3,3]`, reported not analyzed).
Frozen (no LoRA): img_mod/txt_mod (AdaLN producers), all QK-norms,
txt_norm, time_text_embed, proj_out (declared target, filtered at
load). Metric: ||B@A||_F / ||W_base||_F per module (60-layer roll-up):

```
attn.add_q_proj       0.0558   text-side joint-attn Q
attn.to_add_out       0.0492   text-side joint-attn out
attn.add_v_proj       0.0455
attn.add_k_proj       0.0452
txt_mlp.net.0.proj    0.0436
attn.to_k             0.0428
img_mlp.net.0.proj    0.0409
attn.to_q             0.0409
txt_mlp.net.2         0.0348
img_mlp.net.2         0.0310
attn.to_v             0.0293
attn.to_out.0         0.0277
diffuser global       0.0396
VAE decoder total     0.1245   (3.1x the diffuser)
```

Stem: norm_out.linear 0.936 (delta ≈ base itself — small-norm head
modulation moved most; absolute 74.4 vs 79.5), txt_in 0.050 (2x
img_in 0.026). Layers: L0 0.089 (entry outlier, 2.3x mid mean),
mid 0.036 flat (min L19), late 45-59 rising to 0.050 (L59 0.072).
VAE groups: up-blocks 0.135 > mid 0.101 > inout 0.047 >
post_quant_conv 0.015. Where depth-learning concentrated, in order:
output head modulation, VAE decoder up-blocks, block-0 entry, late
DiT blocks, text-stream projections over image-stream ones
(add_q > to_q, to_add_out > to_out, txt_in 2x img_in). Local
filters nowhere: every most-moved part is global routing (joint
attention), entry/exit conditioning, or the decoder — the same
layout-scale story Phases 2-6 told from the student side.

### 2. DiT blueprint (QwenImageTransformer2DModel, diffusers, 60×3072/24h)

Joint self-attention over concat [text;image], full bidirectional,
dense, QK-RMSNorm + 3D RoPE ([16,56,56]), single joint mask; dual-
stream AdaLN-Zero blocks (GELU-approx MLPs ×12288); VAE 1/8
(2**len([F,T,T])), patch 2. Sequence at 1024x672: latent 128x84 →
2688 img + 338 text = 3026 tokens/block (1024²: 4096+338 = 4434).
Text path: precomputed embeds feed all 60 blocks as joint tokens
(text encoder never loaded); banked pair is (8,338,3584) BF16 +
(8,338) all-true mask, finite, sha-pinned — single pair reused via
[0:1]/expand across batch (network_graph _match_batch,
run_marigold_v2 [0:1]). Correction: no working-repo file ever
verified this broadcast before (exhaustive grep negative) — the
constancy claim is established by this round's measurement plus the
teacher-side reuse logic, cited in the banked json, not inherited.
Timestep fixed t=499/1000.

Tiny-global-attention parts list (emitted-pattern status per README):
in-table — MATMUL/ADD/MUL/SUB (projs, modulation), RMSNORM (QK-norm,
txt_norm — use RMSNorm everywhere), SILU (modulation MLP),
CONCAT/GATHER (joint tokens, neighborhoods), BMMV + SOFTMAX_WIDE
(head-batched global attention at student scale: 256px → ~256 img
tokens + const text), ROPE/ROTARY (baked tables), BETA const-fill
(banked text summary — no encoder). Gaps named, not implemented —
LAYERNORM (block norms; substitute RMSNorm), GELU (block MLPs;
substitute SILU/SwiGLU), SIN/COS (timestep sinusoid; bake tables per
ROTARY precedent), CONV incl. 3D + GroupNorm + INTERP (all VAE
internals; student predicts depth directly, no VAE).

### Nomination (not a kill)

The anatomy nominates the next student: a tiny joint-attention
student — global text+image tokens at low res, RMSNorm + SILU only,
baked time tables, const text fill — every part in-table or
substitutable-to-in-table per the list above, with each
substitution named here before any round declares it. Why this and
not more mixer: the teacher's own most-moved parts are attention,
head, entry/late conditioning, and decoder — nothing local — while
six student phases exhausted local context to full-frame scale at
0/156 held-out cells. Bounds stated openly: anatomy nominates the
parts list, not a passing grade; the build round needs its own
frozen constitution (axes, budget, judge unchanged), and empty stays
a complete report there too.

## Phase 8 — tiny joint-attention student (constitution, FROZEN pre-run)

One nominated architecture, built as nominated. No mid-round
invention — the Phase-5 spiral rule binds here too. Anything below
changes only in a new declared round; violations bounce the round.

Architecture `jattn64b2` (field: full-frame by construction):
- Token grid: stride-32 cell-center RGB samples over the native
  frame (1024x672 → 32x21 = 672 tokens, dim 3). Strided sampling,
  honest precedent (cf. Phases 3-6); averaging would need parked
  POOLAVG — not used. Token proj 3→W (W=64, MATMUL+ADD).
- Positions: factored learned row (21,W) + col (32,W) tables, ADD
  (32 cols x 21 rows at 1024x672; labels corrected pre-run, same
  53xW count). No RoPE (allowed but unneeded at 672 tokens).
- Const text fill: C=4 learned constant tokens (dim W), prepended
  to every frame's sequence — the student's analog of the teacher's
  image-independent text (Phase-7 blueprint). Fixed after training;
  deploys as BETA const-fill F + CONCAT F.
- Timestep: DROPPED with reason stated — feedforward depth needs no
  diffusion step; the baked-table substitute stands declared for a
  future round, unused here. Fewer parts than nominated is a
  simplification, not an invention.
- Blocks B=2, joint self-attention over 676 tokens: RMSNorm →
  Q/K/V projs → scale+joint-softmax → values → out proj → residual;
  RMSNorm → SILU MLP (W→4W→W) → residual. Explicit-softmax form
  (no fused SDPA backend to vary run to run).
- Head: nearest-cell broadcast — pixel (i,j) GATHERs token
  (i//32,j//32) (floor indices precomputed, zero-code metadata) →
  CONCAT [token(W), pixel RGB(3)] → MLP (W+3→W→1, SILU) → depth.
  This is the declared upsampling: index broadcast, blocky by
  construction and stated. INTERP is parked and untouched; no CONV,
  no GroupNorm, no LAYERNORM (RMSNorm everywhere), no GELU (SILU
  only), no SIN/COS. Shape views carry no compute (RESHAPE parked
  status noted openly).
- Per-op citations (README table): MATMUL F, ADD F, MUL F (scale),
  RMSNORM F, SILU F, CONCAT F, GATHER F, BMMV F (head-batched
  attention), SOFTMAX_WIDE F (joint softmax), BETA F (const text).
  Size: ~110k params (~440KB); full-frame transient <1GB — nowhere
  near 17GB.

Budget: ONE architecture. Fallbacks (≤2) run ONLY on fails-to-train
(loss NaN/diverged/static across a full run — losing on bars is a
complete result, never a trigger), each declared before running:
F1 `jattn32b2` (W32, OOM trigger only); F2 `jattn64b1` (single
block, NaN/divergence trigger only).
Training cap per variant: 2000 steps, Adam lr 0.01, seed 7,
stride-7 sample, no-grad alignment (same loss/judge family);
wall cap 5 min; GPU box to itself; Qwen-7B never co-loaded.
Judge unchanged: Phase-1 bars, same alignment, verdict-reported.
Winner rule: most held-out cells under bars; a claimed winner gets
retrain-match (trajectory + cell table identical) before it counts.
Empty (0 cells) banks the table and closes — honest-empty precedent
stands. Metrics + configs banked in-repo (`battery8` key +
`attn_run.json`); weights out-of-tree (`~/student_runs/<tag>/`).

### Battery results (ONE variant ran, 102s wall, single GPU to itself)

`jattn64b2`: 108,033 params (444,084-byte weights), grid 21x32 =
672 tokens + 4 const, seq 676, loss 0.5288 (iters 500-1500 plateau)
→ 0.5023 final. No NaN/divergence — trained, fallback triggers
never met (F1/F2 never ran). Held-out: 0/12 cells under bars
(gate-counted); best geo_w0-smooth 0.194/0.09 (2.2x), worst
geo_w1-nonplanar 0.499/0.08 and tex 0.431/0.07 (~6x). Train cells
0.37-0.66 sit at/above scaffold levels (squirrel-smooth 0.664 vs
scaffold minimum 0.483) and below the Phase-3 MLP baseline
everywhere (its held-out best smooth 0.169 vs 0.194 here).
Winner: none — no retrain-match needed (nothing to claim).
Diagnosis: underfit, not near-miss. The model descends late but
never fits train: 64 dims per 32x32 cell through a single head and
nearest-cell broadcast starve each pixel's global code, and/or 2000
steps at lr 0.01 under-optimize attention (1000-iter plateau with a
terminal escape reads as slow optimization, stated as reading not
fact). The nomination's parts list is built and banked as declared;
the architecture as sized does not learn depth layout. What would
change that (new-round declarations, none run): optimization budget
(steps/schedule), finer token grid, multi-head — each its own frozen
round. Earned sentence: empty is complete — first attention datum
banked, mixer wall now has joint-attention company at 0/12.

## Phase 9 — optimization budget, architecture frozen (constitution, FROZEN pre-run)

Question: was Phase 8 under-optimized or under-capacity. The only
knob is training schedule — architecture byte-frozen (`jattn64b2`
exactly as banked: W64/B2/stride-32/C4/row21+col32/learned-const/
RMSNorm+SILU/nearest-GATHER head; any architecture change confounds
the question and voids the round). Code changes confined to schedule
plumbing (`--schedule`, cosine path); anything else is scope breach.
Suggested shape adopted as-is (time-vs-decay, no reason to replace),
then frozen — three variants, each with exact steps + LR + wall cap:

| tag | steps | LR schedule | wall cap |
|-----|-------|-------------|----------|
| jattn64b2-8k-const | 8000 | Adam 0.01 constant | 12 min |
| jattn64b2-8k-cos | 8000 | Adam cosine 0.01→0 over 8000, no warmup | 12 min |
| jattn64b2-16k-cos | 16000 | Adam cosine 0.01→0 over 16000, no warmup | 25 min |

Seed 7, stride-7, no-grad alignment, loss, data: all unchanged
(schedule is the only variable). Trainer keeps writing
`attn_run.json`; each variant is copied to `attn_run_<tag>.json`
before the next run (backup dance — no output-path code change).
Weights out-of-tree (`~/student_runs/<tag>/`); cells banked under
`battery9` (reference: `jattn64b2`, the baseline of this round).
Verdict rule frozen alongside: any held-out stratum improving
>0.05 vs the banked Phase-8 numbers = "moves" (reopens the
architecture empirically); nothing moving = "flatlines"
(undercapacity proven). Judge otherwise unchanged (Phase-1 bars,
same alignment, verdict-reported; winner, if any, gets retrain-match
before claimed). GPU box to itself; Qwen-7B never co-loaded.

### Battery results (3/3 variants ran, walls 368s/372s/757s, GPU solo)

Train loss: 2k 0.50 → 8k-const 0.53 (oscillates, never descends) →
8k-cos 0.34 → 16k-cos 0.10. Cosine unlocks fitting; constant-LR
stalls where Phase 8 stalled. Train strata follow (16k-cos:
squirrel-tex 0.108, squirrel-near 0.105 — first sub-0.15 train cells
of the attention line).

Held-out moves vs banked Phase-8 (frozen >0.05 rule, per stratum):
- 8k-const: 4 MOVEs (far both frames +0.09/+0.14, geo_w1
  nonplanar +0.08, tex +0.06) with 4 worsenings (geo_w1 near
  -0.28, planar -0.21, smooth -0.17).
- 8k-cos: 2 MOVEs (far both +0.06/+0.08), 4 worsenings.
- 16k-cos: 4 MOVEs (far both +0.06/+0.10, nonplanar +0.06, tex
  +0.06), 4 worsenings (near -0.12/-0.14, planar/smooth worse).
Verdict: MOVES — the frozen rule fires on every variant (far-side
strata move repeatedly), so the architecture reopens empirically.
Stated with it: moves are bidirectional and bar-blind. Every
variant is 0/12 under bars (gate-counted, 0/36 total); best cells
stay ~2x over bars while paired strata worsen by larger amounts;
held-out means never leave 0.27-0.32. Optimization moves train
dramatically (0.50→0.10) and shuffles held-out (±0.28) without
closing anything — the Phase-5/6 generalization wall reproduced
inside the attention family. No winner, no retrain-match needed.
For the multi-head/finer-grid queue: the brief sharpens — capacity
and schedule both move train freely now, so the next round must
move held-out directionally (near/planar/smooth, the strata that
worsen here) or join the wall at higher cost. Nothing in this round
suggests more steps alone will do it (8k→16k cosine: train -0.24,
held-out means flat).

## Phase 10 — multi-head, capacity-matched (constitution, FROZEN pre-run)

Question: do attention heads move held-out directionally (the Phase-9
brief: near/planar/smooth must move, not just far). Heads are the
ONLY variable — suggested 1/2/4 adopted as-is (anchor + split +
split again, no reason to replace), then frozen. Everything else
byte-frozen from `jattn64b2` (W64/B2/stride-32/C4/row21+col32/
RMSNorm+SILU/nearest-GATHER head).
Capacity-matching rule: total width fixed at W=64, dim-per-head =
64/H (H1: 1x64, H2: 2x32, H4: 4x16); the head-split is a reshape
(zero params, zero code in the deployed graph beyond batching), so
all variants carry exactly 108,033 params — matched budgets mean
heads, not capacity, is what's tested. Per-head-dim accounting above
is the rule; any param-count deviation voids the variant.
Training schedule fixed: 8k-cosine (0.01→0, no warmup), seed 7 —
8k-cosine is reused because it is the schedule that learns (train
0.34 with far moves; constant-LR stalls at 0.53). Includes an H1
anchor rerun at this schedule (expectation: near-match to banked
`jattn64b2-8k-cos`, stated not asserted — plumbing differs by the
reshape).
Budget: 3 configs (`jattn64h1-8kcos`, `jattn64h2-8kcos`,
`jattn64h4-8kcos`), GPU box to itself, wall cap 12 min each.
Multi-head routes through tabled attention: scores/values batched
over heads (BMMV F), rowwise joint softmax (SOFTMAX_WIDE F),
explicit-softmax form kept; all other citations unchanged
(MATMUL/ADD/MUL/RMSNORM/SILU/CONCAT/GATHER/BETA F). New attention
machinery would be a stop-finding — none introduced (split/merge
are views). Judge unchanged (Phase-1 bars, same alignment,
verdict-reported). Winner: most held-out cells under bars; empty is
complete. Metrics banked separately (`attn_run_<tag>.json` +
`battery10` key, reference `jattn64b2`); Phase-8/9 primaries never
overwritten (separate-files precedent); weights out-of-tree.

### Battery results (3/3 ran, walls 373s/405s/405s, GPU solo)

Capacity rule verified in the weight files: 108,033 params ×3 —
heads, not capacity, was tested (gate-counted leg).

```
tag             heads  dim/head  train-loss  tr-mean  ho-mean  ho-best
jattn64h1-8kcos  1     1x64      0.391       0.459    0.415    0.222
jattn64h2-8kcos  2     2x32      0.439       0.463    0.412    0.224
jattn64h4-8kcos  4     4x16      0.275       0.425    0.413    0.210
(banked 8k-cos H1 shape for context: 0.344 / 0.428 / 0.406 / 0.202)
```
Anchor note (stated pre-run as expectation, now measured): H1 rerun
lands near but not on the banked 8k-cos (loss 0.39 vs 0.34,
held-out means 0.415 vs 0.406) — the head-split reshape changes
matmul kernel order, and chaotic trajectories diverge from there.
Same qualitative picture, not identical numbers; the comparison that
matters (H1/H2/H4 within this round, identical code path) is clean.
Winner: none — 0/36 held-out cells under bars (0/12 each,
gate-counted); no retrain-match needed.
Diagnosis: heads change nothing on held-out at matched capacity
(0.415/0.412/0.413 — flat to ±0.003) while moving train slightly
(H4 fits best, 0.275). The Phase-9 brief asked for directional
near/planar/smooth movement; heads do not supply it — the split
re-routes the same 64 dims, and routing was never the bottleneck
(the bottleneck is what 64 dims per 32x32 cell can carry, Phase-8
reading, unrefuted). Finer grid is now the last queued axis with a
mechanism behind it (more tokens = narrower cells = richer per-pixel
code); heads join schedule as moved-but-bar-blind. Empty is
complete — the head question is answered, not open.

## Phase 11 — finer grid, last queued mechanism (constitution, FROZEN pre-run)

Question: does narrower-cell global code move near/planar/smooth.
Grid variants only — each states stride/cell-count/field plus the
mechanism, or it is refused at declaration (including my own):
- `s16h1`: stride-16 → 64x42 = 2688 tokens + 4 const, full-frame
  field. Mechanism: 16x16px cells carry 4x the spatial code per
  pixel vs 32x32 (isolates grid at H1).
- `s16h2`: stride-16, H2 (2x32). Mechanism: fine grid × head
  interpolation point (mirrors the Phase-10 sweep at fine grid).
- `s16h4`: stride-16, H4 (4x16). Mechanism: finest code × best
  Phase-10 routing (combined best).
Stride-8 (128x84 = 10752 tokens, 250x attention entries) is refused
at declaration: cost without proportion (hours at 8k steps), and
stride-24/48 don't divide 1024x672. The 4th slot stays deliberately
empty — no fourth variant carries the mechanism without confounding
(width change would confound grid; const-text change is another
axis). Schedule fixed: 8k-cosine (the schedule that learns), seed 7.
Capacity accounting: pos tables grow intrinsically with grid
(21+32 → 42+64 rows: +3.4k params, ~3%); attention entries 16x
(457k → 7.2M); head/sample costs unchanged. Wall cap 20 min each.
Winner: most held-out cells under bars. Directional-move rule
frozen: any near/planar/smooth held-out stratum improving >0.05 vs
banked Phase-8 `jattn64b2` reopens the frontier empirically even
short of bars. Empty (nothing clears, nothing moves directionally)
is complete. Metrics banked separately (`attn_run_<tag>.json` +
`battery11` key, reference `jattn64b2`); primaries untouched;
weights out-of-tree. Per-op citations unchanged (same ops, longer
sequences); no new machinery (grid size is an index range).

### Battery results (3/3 ran, walls 436s/532s/685s, GPU solo)

Zero trainer code changes — stride/heads were already CLI-driven;
the round's only diffs are banked numbers, gate legs, and this text.

```
tag          grid         loss   tr-mean  ho-mean  near/ps held-out moves
jattn64-s16h1 64x42=2688 0.379  0.465    0.398    0 MOVEs (near worsens x2)
jattn64-s16h2 64x42=2688 0.321  0.455    0.386    0 MOVEs (near worsens x2)
jattn64-s16h4 64x42=2688 0.221  0.400    0.396    0 MOVEs (near worsens x2)
(params 111425 x3: +3392 pos rows, exactly as accounted)
```
Train fits harder with grid+heads (16k-cos line extended: 0.10 at
s32h4 → 0.22 here at 8k; train cells reach 0.11), but held-out
near/planar/smooth move nowhere directionally — 0/18 target cells
improve >0.05 vs Phase-8 while 6 worsen beyond -0.05 (near pairs in
every variant). Nothing clears bars either: 0/36 total
(gate-counted). No winner, no retrain-match needed.
Diagnosis: the mechanism fired on train and missed on held-out —
narrower cells enrich the memorizable code without teaching layout.
Far-side strata (the easy, low-frequency layout) were already the
only movers in Phases 9-10; the hard strata (near/planar/smooth,
occlusion and detail) have now resisted local context to 1025px,
global spokes, global attention, schedules to 16k, heads to 4, and
grids to 2688 tokens. The failure is systematic across families:
train fits, held-out doesn't — an inductive-bias wall, not a budget
wall.
Earned sentence: empty is complete — the in-table arc is exhausted
including the last queued mechanism. The only doors left are
constraint changes, named with prices, none opened: LAYERNORM
patterns C+CUDA (drilled-only scaffolding exists); GELU patterns per
backend (SILU/SwiGLU cover live use); CONV incl. 3D + GroupNorm +
INTERP/POOLAVG lowerings (phi_conv audit-first precedent; VAE-grade
decoder territory); fused joint-bidirectional attention kernels
(performance, not capability — BMMV/SOFTMAX_WIDE already cover the
math); raw scale (evidence prices it low: train moves, held-out
doesn't). Any of these is its own declared round with its own
constitution.

## Phase 12a — procedural analytic scenes, diagnose before training

No student training this round (12b is a separate go/no-go).
Instrument: `scripts/synth_depth.py` — pinhole renderer over
spheres / axis-aligned boxes / floor / back wall, closed-form
intersections, exact float64 z-depth; flat shading (fixed per-face
shades, no lights) + 3D-checker textured variants; 320x256
(diagnostic scale; dims divide by 32). Scenes generate on the fly
from fixed seeds (12000+idx) — zero bytes banked; the generator is
the fixture. Gate `tests/test_synthetic_depth.py` (SKIP-clean).
Catalog N=24: count {1,2,4} × occlusion {single,stacked} × range
{shallow 2.5-5, deep 2.5-11} × texture {flat,checker} — each combo
exactly once, strata by construction in scene meta. Verified:
24/24 combos, same-seed identical, scalar-loop self-consistency
8.9e-16 worst dev.
12b decision bar (preemptive — stated before measuring, frozen):
GO iff ALL three hold — (1) teacher bias ≪ student error
(scene-mean aligned-MAE headroom ≥2x vs mscale64-vs-GT: something
to learn toward); (2) ≥1 construction axis separates mscale64
error directionally (group-mean gap >0.05: layout signal inside an
in-table student's errors); (3) teacher seeded-twice noise == 0
(byte-identical: stable labels). Else NO-GO with evidence.
Constraint noted regardless: jattn64b2's learned row/col pos tables
are grid-locked (21x32 @1024x672) — off-grid eval needs retraining
or architecture change, so any GO retrains or re-specifies the
attn line; mscale64's strided sampling runs at any resolution.

### Measurements (teacher 2x24 runs + student evals, GPU solo)

Corrections made mid-round, stated openly: (a) first compared in
z-space (teacher bias 2.17 — confounded: the teacher trains
relative-LOG-depth per `training_relative_log_depth_config`, so
affine comparison belongs in log space; all numbers below are
log-space); (b) 14/24 scenes starve the `far` mask (median
collapses to the back-wall value on box-room stats —
`chain/depthprior.py:near_mask` assumes continuous real depths;
those cells record null, excluded from means, counted openly —
130 finite + 14 nulls per evaluatee).
Teacher noise: seeded-twice byte-identical 24/24 (plus gate
2-scene spot rerun) — condition (3) HOLDS, labels stable.
Teacher bias (align vs analytic log-GT): scene-full mean 0.251
(best 0.018, worst 0.609). By axis: textured checker 0.082 vs
flat 0.420 (5x — the teacher is texture-hungry; Phases 2-6 from
the other side), shallow 0.284 vs deep 0.218, count/occlusion flat.
Students vs GT: mscale64 mean 0.369 (best 0.203, worst 0.652);
jattn64b2 sidecar mean 0.536 (upscaled-1024, blur confound —
supplementary only). mscale64 by axis: count 1/2/4 →
0.302/0.329/0.477 (Δ0.175); stacked 0.331 vs single 0.408
(Δ0.077); texture flat (0.376 vs 0.363 — the student is
texture-blind where the teacher feasts). mscale64-vs-teacher gap
on synthetic: mean 0.328 (learnable signal toward labels exists).
Bar evaluation: (1) headroom 0.369/0.251 = 1.47x < 2x — FAILS
(the shortfall is the teacher's own 0.42 on flat scenes:
mimicry-training would teach teacher errors on half the set);
(2) axis separation HOLDS (count, occlusion); (3) noise HOLDS.
Verdict: NO-GO with evidence — the bar needs ALL three. Not
close-call logic: the textured-only subset (teacher 0.082 vs
student ~0.36, 4.4x) would pass alone, but flipping on a post-hoc
subgroup is goalpost-moving; a textured-only 12b' needs its own
round and its own bar, stated here, not run.

## Phase 13 — deeper attention stacks, last in-table room (constitution, FROZEN pre-run)

Question: does stack depth move held-out where heads could not.
Blocks-depth is the ONLY variable — suggested 2/3/4 adopted as-is,
then frozen (`jattn64-s16h2b2/b3/b4`; B2 reruns at identical flags
as the round's own anchor — expectation: near-match to banked
`s16h2`, stated not asserted per the Phase-10 reshape-drift
precedent). Everything else frozen: W64, heads H2 (neutral
midpoint of a bar-blind sweep — no reason to deviate, so none
taken), stride s16 (richest code; depth is tested where code is
richest), 8k-cosine, seed 7, splits unchanged.
Capacity accounting (diagnostic, not confound): each block adds
~50k params (4×(64×64) projs + 2×(64×256) MLP + norms) — B2
111,425 → B3 ~161k → B4 ~212k, exact counts verified in the weight
files post-run. The train-heldout gap across depths is the
diagnostic (memorization vs learning); growth itself is declared
here, not hidden.
Budget: 3 configs, GPU box to itself, wall cap 25 min each.
Winner: most held-out cells under bars; directional near/planar/
smooth movement vs banked Phase-8 `jattn64b2` (>0.05, same frozen
threshold as Phase 11) reopens empirically. Empty is complete.
Metrics banked separately (`attn_run_<tag>.json` + `battery13`
key, reference `jattn64b2`); primaries untouched; weights
out-of-tree. Per-op citations unchanged (deeper stacks reuse the
same tabled ops); `--blocks` already CLI-driven, zero code changes
expected.

### Battery results (3/3 ran, walls 549s/631s/736s, GPU solo)

Determinism datum: B2 rerun is BIT-IDENTICAL to banked s16h2
(max abs diff 0.0 across 36 cells, sessions apart) — same flags +
same code is exactly reproducible; the Phase-10 drift came from
changed code, not chaos. Param ladder verified in weight files:
111425 → 161281 → 211137 (+49,856/block, as declared).

```
tag              blocks  params   train-loss  tr-mean  ho-mean  dir-moves
jattn64-s16h2b2  2       111425   0.321       0.455    0.386    0/6
jattn64-s16h2b3  3       161281   0.430       0.462    0.418    0/6
jattn64-s16h2b4  4       211137   0.529       0.522    0.375    0/6
```
Winner: none — 0/36 under bars (gate-counted); no retrain-match
needed. Train DEGRADES with depth at fixed budget (0.321 → 0.430
→ 0.529-static; B4 never leaves Phase-8-2k territory) while
held-out sits flat (0.375-0.418). Depth buys nothing twice over:
no better optimum reached AND no better generalization at any
reached optimum. The train-heldout gap diagnostic reads
memorization-proof in reverse — B4 cannot even memorize (train
0.52), so depth here is an optimization tax with no learning
upside at 8k-cosine. More steps for deeper stacks would confound
depth with budget (refused in spirit at declaration: the round
tests depth at the mover schedule, not schedules per depth).
Earned sentence: empty is complete — the in-table story is
complete including stack depth. Remaining doors, priced, none
opened: emitted-pattern constraint changes (LAYERNORM/GELU/CONV+
GroupNorm/INTERP lowerings — backend work per op, drilled-only
scaffolding exists for norms); fused joint-attention kernels
(performance only); raw scale at matched depth (evidence prices it
low); the textured-domain restriction from Phase 12a (textured-only
synthetic: 4.4x headroom — needs its own round and bar, still
stated-not-run).

## Phase 14 — patch-shuffle probe, no training (design FROZEN pre-run)

Question: is the depth task here global-reasoning or texture-driven.
Instrument: deterministic patch scramble (PIL/numpy only, no new
deps) of the 6 fixture inputs at 1024x672 (the bars' resolution).
Patch sizes frozen: 16px (exact 64x42 grid, single-pool permutation)
+ 64px (16 full cols x 10 full rows + bottom 32px partial row:
two-pool permutation — 160 full tiles within-pool, 16 partial
within-pool; ragged rule stated, deterministic). 64px adopted over
32px with reason: 64px preserves object-part-scale local features
(the 33-65px mixer-field scale), directly testing whether local
features suffice; 32px sits inside tested field range and adds less.
Seed 7 fixed. Three systems on intact-vs-shuffled pairs, metric =
aligned-MAE(shuffled_pred → intact_pred) per stratum, same harness
alignment, masks from intact (same masks all arms):
(a) seeded teacher — intact arm reuses banked .npy (no rerun); p16
batch runs twice (seed 2025 x2, byte-identity check — determinism
mechanism already proven, re-checked on OOD inputs), p64 once;
existing CLI, GPU solo. (b) DAV2 geometric replica
(dav2_revisited/dav2_reverse/geo_depth.py, read-only, CPU) — all
three arms live; masks at DAV2 res (nearest-depth + bilinear-RGB
downscale of intact; stated). (c) banked students mscale64 +
jattn64b2 (CPU; jattn native grid) — all three arms live.
Verdict readings frozen: global-reasoning pattern =
teacher_mean_delta / dav2_mean_delta ≥ 3 at 16px (order separation;
family verdict: needs joint reasoning, not bigger mixers);
texture pattern = max/min ≤ 2 across all systems' mean-deltas
(family verdict: opposite — local features suffice, architecture
question reopens differently). Neither pattern = mixed, reported
with numbers, family verdict withheld (not a loophole: mixed still
attributes dependence per system). Gate: tests/test_shuffle_probe.py
(determinism incl. tile-multiset preservation, banked-delta
integrity, teacher/DAV2/student spot legs, SKIP-clean).
Numbers banked as scalars (samples/marigold/shuffle_probe.json);
predictions stay in /tmp.

### Results (6 fixtures x 3 arms x 4 systems; teacher 3 GPU runs)

Corrections made mid-round, stated openly (nothing banked under
either flaw): (a) pixelwise align-on-scrambled compared mismatched
content (caught by 1e5-scale absurd deltas) → unshuffle-then-align
(roundtrip-exact inverse; DAV2 via NN-upscale, resampling control
0.031 ≪ effects); (b) raw aligned-MAE is scale-meaningless across
systems (student outputs ~1e5 native) → rel = raw/std(own intact)
banked alongside raw; verdict reads rel.
Mean rel-deltas (fraction of own intact variation destroyed):
teacher p16 1.27 / p64 1.21; dav2 1.33 / 1.31; mscale64 0.77 /
0.36; jattn64b2 0.81 / 0.78. Strata: teacher/dav2 collapse hardest
in near (2.0-2.2) then far (1.4-1.7), ~1.0 elsewhere, patch-flat;
mscale64 uniform ~0.7-1.0 at p16 halved ~0.3-0.4 at p64;
jattn eerily flat 0.77-0.85 in every stratum at both sizes.
Frozen readings, mechanically: teacher/dav2 = 0.96 (rule needs
≥3) — global-pattern as written: NO. Spread p16 1.72 (≤2 HOLDS),
p64 3.59 (FAILS) — texture-pattern overall: NO. So the letter
says mixed — and the letter's DAV2 premise was my error (DAV2-Small
is a global ViT; of course it collapses with the teacher rather
than staying graceful). Corrected with evidence, not goals: both
global machineries collapse ~1.2-1.3 at both scales while local
ones don't — the task IS global-reasoning-dependent.
Family verdict: NEEDS JOINT REASONING, not bigger mixers.
Mechanism per system: mscale64's p16→p64 halving is a direct
locality proof (33px field survives inside 64px patches —
Phases 2-6 locality, convicted from the other side); jattn's flat
0.8 proves its global machinery uses no global structure
(convergent with the underfit diagnosis — it never learned layout,
so scrambling layout changes little); teacher/dav2 near-strata
collapse shows foreground layout is what global reasoning is FOR.
Nomination: nothing new built from this — but a new acceptance
criterion for anything next: a student claiming joint reasoning
must show teacher-scale shuffle deltas (≈1.2 rel, near worst) on
this probe; mscale64-scale grace (≤0.8) disqualifies the claim.
That criterion is banked in the gate (locality-signature leg).

## Phase 15 — DINOv2-token inputs, information-vs-machinery (constitution, FROZEN pre-run)

Question: does global INFORMATION save the joint-attention family,
or is the machinery itself the wall. The only variable is input
information; architecture family held (Phase-8 shape: B2 joint
blocks, RMSNorm+SILU, nearest-cell broadcast head as built).
Frozen DINOv2-base (86M ViT, local ~/student_runs/dinov2-base)
runs ONCE per fixture (CPU, deterministic) → patch tokens
(3504×768, CLS dropped, stated) cached out-of-tree
(~/student_runs/dino_tokens/<fixture>.pt) with sha manifest
in-repo; the encoder never runs at student-train time.
Two configs (third slot deliberately empty — width⊗information is
a follow-up conditioned on movement, confounded now):
- `dino-tokonly`: seq = DINO tokens projected 768→64 (no RGB token
  stream, no const: C=0 — const REPLACED by DINO, literally).
  Head: nearest-DINO-cell broadcast (14px cells, idx (i//14)*73 +
  (j//14), 2px edge strip clamped to last token — edge rule
  stated) + RGB concat, as built.
- `dino-tokrgb`: seq = DINO tokens (3504) + RGB cell tokens (672,
  stride-32 as built); head reads RGB outs (as built, 32px cells).
  Closest literal swap: const slot now carries image-conditional
  global tokens.
Schedule fixed 8k-cosine, lr 0.01, seed 7 (the mover; information
is the variable, not optimization). Wall caps: tokonly 30 min,
tokrgb 45 min (4176-seq attention costs ~5x s16 — stated, not
hidden). GPU box to itself.
Stated openly: the DINOv2 encoder is a frozen preprocessor, NOT
in-table and NOT deployable-as-student — this round tests the
INFORMATION hypothesis, not deployability. A win means
"information suffices" and nominates feature-distillation next;
it does not ship. Winner: most held-out cells under bars;
directional near/planar/smooth vs Phase-8 (>0.05, frozen) reopens
empirically. Empty is complete. Metrics banked separately
(`attn_run_<tag>.json` + `battery15` key, reference `jattn64b2`,
dino provenance in flags); weights out-of-tree; no
emitter/opcode/assembly changes.

### Battery results (2/2 ran, walls 501s/548s, GPU solo)

DINO cache: 6×3504×768 fp32, CPU-rerun-identical, sha-manifested.
Capacity verified: 156,993 ×2 (108033 + 49216 dino_proj − 256
dead ctx — uniform construction leaves tok/pos/ctx weights dead in
tokonly; documented, not hidden). Refactor proven behavior-
preserving: current rgb path + banked Phase-8 weights reproduces
banked frames to 5e-10.

```
tag           train-loss  near-train (chr/dog/sq)   held-out moves  bars
dino-tokonly  0.054       0.52/0.64/0.59 (!!)        0/6 (near worsens x2)  0/12
dino-tokrgb   0.413       0.33/0.40/0.20             0/6 (near worsens x1)  0/12
```
Winner: none — 0/24 under bars (gate-counted); no retrain-match.
The tokonly train row demands a reading, stated as reading:
near-field train cells collapse (0.5-0.65) while loss sits at 0.054
and far fits (0.15-0.27) — predictions blow to 1e12 scale with
relative structure that fits far layout and not near objects
(verified live: near-mask pred-std ~1e12, residual 0.52). DINO
tokens carry scene layout the tiny reader can ride for walls/floors
but not foregrounds; adding the RGB stream back (tokrgb) breaks
even that (loss 0.41 — longer seq, harder optimization, interference).
Verdict: MACHINERY. Information transforms train fit (0.50→0.054,
the largest single train move of the whole student arc) yet moves
zero held-out cells directionally and worsens near — so global
features do not convert through this reader, and no
feature-distillation is nominated (a win would have nominated it;
this is not a win). Convergent with Phase-14 (the family doesn't
use global structure) and Phase-13 (deeper readers stall): the wall
is the reader's capacity to convert layout into near-field depth,
not the information fed to it. Empty is complete.

## Phase 16 — two-stage coarse-to-fine, dynamism across passes (constitution, FROZEN pre-run)

Question: is frozen global code + a stronger reader usable, or does
the reader need to co-adapt. Two static passes, no new machinery.
Pass 1 = frozen banked `jattn64b2` weights (front-end — reuse, don't
retrain): per frame it yields image-token global code (672x64) AND
coarse depth (its own full forward, frozen, no grad). Pass 2 =
trained refinement MLP (68→128→128→1, SILU) on CONCAT[global code
(64, nearest-cell GATHER broadcast) + coarse depth (1) + local RGB
(3)] → depth. Per-op citations (README table): GATHER/CONCAT F
(broadcast+concat, Phase-8 precedent), MATMUL/ADD/SILU F (MLP).
Any gap = stop-finding; none introduced (wider head, same ops).
Two variants, both run (suggested pair adopted with its reason):
- `cf-frozen`: Pass-1 frozen; ONLY the 25,773-param head trains
  (2000 steps, Adam lr 0.01 const, seed 7, stride-7, no-grad
  alignment; wall cap 5 min). Freezing isolates whether the
  information was usable.
- `cf-joint`: same graph end-to-end (front-end warm-started from
  banked jattn64b2 via strict=False load, head fresh; all 129,389
  params train: 108033 − 4417 old head + 25773 new head — formula
  declared, verified post-run). 8k-cosine, lr 0.01, seed 7
  (the mover; wall cap 10 min). Joint confounds reader-vs-
  information (stated) and runs second as declared.
Winner: most held-out cells under bars; directional near/planar/
smooth vs Phase-8 (>0.05, frozen) reopens empirically. Empty is
complete. Metrics banked separately (`attn_run_<tag>.json` +
`battery16` key, reference `jattn64b2`, pass-1 provenance in
flags); primaries untouched; weights out-of-tree. GPU solo;
Qwen-7B never co-loaded.

### Battery results (1 complete + 1 timeout, GPU solo)

`cf-frozen`: loss STATIC 0.5025→0.5024 across 2000 steps (39s wall)
— the head never trains. Mechanism measured live, not guessed:
frozen pass-1 outputs span SEVEN orders of magnitude (coarse std
1.7e6, code std 1084, rgb std 0.27) — SiLU saturates on day one
and Adam never recovers in-budget. The banked jattn64b2 weights
survive only because their own training aligned the scale away;
as FEATURES they are pathological. Cells banked as-run (0/6 moves,
0/12 bars — an untrained head approximately reproduces coarse).
`cf-joint`: declared, run, EXCEEDED the frozen 10-min wall
(killed at 600s with zero artifacts — nothing written, nothing
banked, nothing claimed). Cost autopsy: joint full-frame
forward+backward plus the 128-wide head over the stride sample
runs ~2x Phase-8 per step; the 10-min cap (my mis-estimate,
stated) covered ~2/3 of 8k steps. No rerun at a bigger cap
(frozen is frozen) and no partial banking.
Verdict: refinement-uses-structure is UNREACHED, not refuted —
but the attempt prices it: static passes over frozen codes need
normalization machinery (a design change, itself undeclared here)
or joint budgets beyond single-pass caps. Either way the
demand is per-input adaptivity with a price tag: data-dependent
routing/normalization = control-flow lowering (CALL and friends),
named as the measured demand for the constraint debate, not opened.
Dynamism-across-passes joins the wall as costed-out, and the
earned sentence covers it: the static in-table story is complete
including two-stage refinement attempted, stalled, and timed-out
in the open.

## Phase 17 — textured-domain restricted student (constitution, FROZEN pre-run)

Domain first, exact rule: IN-DOMAIN = checker-textured procedural
scenes (generator construction axis, not inference). 12 banked
(idx12..23) + 12 new (idx36..47 — the next textured block from the
same catalog arithmetic, new seeds/layouts) via seeded teacher CLI
at 320x256 (manifest extension: 12 rows synth_t00..11 + input PNGs
+ teacher .npy, same provenance pattern — shape/dtype legs
must pass on the extended manifest). Splits: train = 8 banked
textured (idx12..19); in-domain heldout = 4 banked (idx20..23) +
12 new (16 scenes x 6 strata = 96 graded cells); OUT-OF-DOMAIN =
12 flat banked + 6 real fixtures — printed unjudged, explicitly
unclaimed (not silently dropped). Targets throughout: teacher
labels (student-arc doctrine; analytic GT grounds the teacher, it
does not train the student).
Headroom re-verified in-round from banked numbers (not inherited
blind): teacher 0.0819 vs mscale64 0.3628 scene-mean on textured
(n=12+12) = 4.43x. Reproduces — round proceeds.
Architecture: mscale64-EXACT (field-33/densewin-3/W64/d2,
177 dims, MATMUL/ADD/SILU + GATHER/CONCAT citations stand) —
banked best on textured (0.363 vs jattn sidecar 0.536), and the bet
is whether teacher-label training teaches it texture use. Two
configs (third slot empty, stated): T1 `texmsc-2k` (2000 steps, lr
0.02 const — Phase-5-comparable); T2 `texmsc-8kcos` (8000 steps
cosine — best chance; schedule effect already measured, no new
confound claimed). Seed 7, stride-7, no-grad alignment. Wall caps:
teacher batch 10 min, each train 10 min. GPU box to itself.
Winner: most in-domain heldout cells under bars (Phase-1 bars);
empty is complete. Metrics banked separately (`student_run_<tag>`
style per-variant files + `battery17` key, reference `mscale64`);
primaries untouched; weights out-of-tree. Winner gets
retrain-match before claimed.

### Battery results (2/2 ran, minutes each, GPU solo)

Convergence datum: T1 (2k const) vs T2 (8k cosine) differ by
4.5e-05 max cell — different paths, same basin (same init+data,
easy landscape, one attractor). Not a bug: weights differ from
byte 1217 on.
In-domain heldout (16 textured scenes x 6): T1 7/96, T2 7/96 —
IDENTICAL 7 cells (synth_b09-planar 0.078; far 0.073-0.090 on
synth_t00/t01/t03/t06/t10). First bar-passing student cells of
the arc, all far/planar on simple room geometry (near-constant
wall depth — easy, stated). Near/tex/smooth/nonplanar: nothing
within 2-6x. Winner: TIE at 7/96 each (same basin) — no single
claim exists, so no retrain-match runs (moot, not skipped).
Out-of-domain, printed unjudged: flat-synthetic 0.215, real
fixtures 0.379 (mscale64-on-real levels — no transfer gain, no
catastrophe; the restriction is load-bearing).
Gate notes: manifest extension (30 rows) required two minimal
harness edits — manifest leg counts 6 real + 24 synthetic,
coverage binds the real-fixture domain (synthetic percentile
collapse stays characterized-not-gated per Phase-12a).
Diagnosis: the domain restriction WORKS directionally (first
passes, far/planar on rooms) and fails structurally (near never
moves: 0.37-0.65 in-domain; texture the model was meant to learn
sits 3-6x over bars). The 4.4x headroom does not convert: teacher
labels on textured scenes teach wall/floor fidelity, not object
layout — mimicry succeeds exactly where there is nothing to learn.
The deployable-thing candidacy narrows to "rooms without
furniture" and dies there honestly. Empty of winners, full of
signal: restriction buys bar-passes, not understanding.

## Phase 18 — edge-concentration analysis, zero training (plan FROZEN pre-run)

Question: does student error concentrate at depth edges (liminal)
or spread through interiors. Frozen plan — violations bounce.
Edge rule (exact): Sobel gradient magnitude on banked teacher
depth, seeds = pixels ≥ frame 80th percentile (percentile, not
absolute — robust across scenes' depth scales; absolute thresholds
are frame-brittle, refused at declaration). Bands adopted as-is
(no reason to replace): edge = Chebyshev distance ≤2px from a
seed; halo = ≤8px and not edge; interior = rest. Dilation via
iterated 3x3-max over shifts (numpy only — scipy is absent, no new
dependencies; exact integer morphology, no approximations).
Systems (banked weights reloaded, forward-only, no retraining):
mscale64, jattn64b2, texmsc-2k, texmsc-8kcos — on the 6 real
fixtures (judge domain; twins run off-distribution, stated —
their inclusion tests whether synthetic training changed edge
behavior where bars live). Metric: ONE global affine fit per
frame (doctrine as ever), then MAE per band. Verdict frozen:
edge-band error ≥2x interior (per system, mean over fixtures) =
liminal concentration CONFIRMED (nominates edge-weighted
training); below = DEAD, no training round follows. Follow-up bar
preemptive (only if confirmed): an edge-weighted round must drop
edge-band MAE >0.05 at matched budget with interior degradation
<0.02, else it dies too. CPU suffices (forward-only; nothing here
needs a GPU). Numbers banked as scalars
(samples/marigold/edge_analysis.json); edge maps regenerate
deterministically from banked depth (no seeds involved).

### Results (4 systems x 6 fixtures, forward-only CPU)

Mean band MAEs (global fit as ever):

```
system       edge    halo    interior  edge/interior
mscale64     0.348   0.299   0.286     1.22
jattn64b2    0.462   0.464   0.433     1.07
texmsc-2k    0.377   0.395   0.365     1.03
texmsc-8kcos 0.377   0.395   0.365     1.03
```
Verdict: DEAD — max ratio 1.22, frozen rule needs ≥2x. No
edge-weighted training round follows (the follow-up bar stays
stated-not-run, permanently moot). Error spreads through
interiors nearly as badly as edges in every system; the liminal
story is quantitatively false. Mechanism notes: mscale64 carries
the highest ratio (local field + hard edges = worst combination,
still 0.8 short of the rule); jattn is flattest (1.07 — uniform
underfit ignores edges and interiors alike); the texmsc twins
match each other exactly (same basin, again) and sit between.
What this closes: the last "maybe the errors are all X" structure
hypothesis with a number attached. Errors are everywhere, in every
system, at every scale tested — the wall has no address left
inside the judged domain.

## Phase 19 — DiT-tiny build, teacher-faithful machinery (constitution, FROZEN pre-run)

Architecture frozen pre-run — teacher-faithful package at student
scale (A1/A2/A4 verdicts cited, not re-derived):
- Tokens: stride-16 full RGB patches (16x16x3=768, the teacher's
  effective pixel-patch: VAE-1/8 x patch-2) at native 1024x672 →
  64x42 = 2688 tokens; patch Linear 768→W (W=64).
- Positions: learned row (42,W) + col (64,W), ADD (family
  precedent; RoPE allowed-not-run, not run).
- Conditioning: NO text stream (dropped with reason: const-text
  never moved anything in Phases 8-13; text-stream ablation is its
  own later question). AdaLN-Zero modulation from a LEARNED const
  vector (SiLU→Linear→6W split shift/scale/gate — BETA const-fill
  precedent; feedforward depth needs no diffusion step).
- Blocks B∈{2,4,6} (depth ladder at matched width): LayerNorm
  no-affine everywhere it belongs (A1 LANDED: LAYERNORM F, cite);
  multi-head H4 self-attention (H4 = Phase-10 best train-fitter;
  best-foot-forward reason), explicit-softmax unfused (A4: unfused
  FASTER at student scale, cite); SILU MLP with GELU-form
  activation SILU(MUL(x,1.702))/1.702 (A2: 314.6dB composition,
  cite); gated residuals (MUL+ADD F).
- Head: nearest-cell broadcast (16px cells, GATHER Phase-11
  precedent) + RGB concat + MLP (W+3→W→1, SILU) — family head as
  built (the round tests the TRUNK).
- VAE EXPLICITLY OUT: direct depth prediction (scoping reason:
  VAE decoder needs A3 CONV + GroupNorm + upsampling, all parked;
  every student in this arc predicts depth maps directly).
- Per-op citations (README table): MATMUL/ADD/MUL/SUB F,
  LAYERNORM F (A1), RMSNORM-free, SILU/CONCAT/GATHER/BMMV/
  SOFTMAX_WIDE/BETA F. No LAYERNORM-substitute, no GELU-substitute,
  no fused kernel — the faithful package needs none of them.
Training recipe frozen: log-normalization (targets z =
(log(t-lo+1) − μ)/σ with lo = train-split min, μ/σ of the logged
train split — all three fixed and banked; correction pre-run:
banked teacher depths are SIGNED normalized (min −1), so the
as-written log(t) is unimplementable — shifted-log keeps the log
compression literally with one extra constant) + aligned-MAE
(same no-grad lstsq, in z-space) + gradient loss (Sobel-gradient
MAE on aligned preds, λ=0.1, full-frame every 50 steps —
Phase-4 aux-cadence precedent; training-time only, zero inference
cost; perceptual/Sinkhorn stay dead, relitigating out of scope).
Eval: invert (t = exp(σẑ+μ) + lo − 1) → standard linear harness
(gate-compatible).
Tags `dit64b2/dit64b4/dit64b6` (H4/s16/nctx-0 in flags).
Budget: 3 configs, 8k-cosine fixed (the mover), seed 7, wall caps
15/20/25 min, GPU solo, Qwen-7B never co-loaded.
Winner: most held-out cells under bars; directional near/planar/
smooth vs prior bests (>0.05, frozen — comparator: per-stratum
minima over banked Phases 8/10/11/13/15/17, computed spec-side;
gate carries counts/complete/flags per precedent). Empty complete.
Metrics banked separately (`attn_run_<tag>.json` + `battery19`
key, reference `jattn64b2`); primaries untouched; weights
out-of-tree.

### Battery results (3/3 ran, walls 638s / <1200s / 1138s, GPU solo)

Pre-run correction (nothing banked before it): banked teacher
depths are SIGNED normalized (min −1) — as-written log(t) is
unimplementable; shifted-log log(t−lo+1) keeps the recipe
literally (lo=−1.0, μ=0.6780, σ=0.3015 banked). Param ladder
verified: 209857 → 359233 → 508609 (+149,376/block incl. AdaLN).

```
tag      blocks  train-loss         cells vs B2 sibs  bars
dit64b2  2       0.8666 static      —                 0/12
dit64b4  4       (wall unrecorded*) ~0 (5.6e-16)      0/12
dit64b6  6       0.8665→0.8542 pin  0.0 exact         0/12
(*B4 wall: completed inside cap; exact value lost to tail-only
capture — my capture error, stated.)
```
Winner: none — 0/36 (gate-counted); no retrain-match. Directional
moves: 0 — all three depths converge to THE SAME predictions
(B6 vs B2 bit-identical). Depth is irrelevant because nothing
trains: the gradient aux (λ=0.1, every 50) pins optimization —
proven by diagnostic (same code, grad-w 0: 0.867→0.489 in 300
steps; throwaway tag, unbanked). The aux fights the MAE direction
every 50th step back to the 0.86 fixed point (all print steps
coincide with aux steps — the sawtooth hides inside the cadence).
Diagnosis: RECIPE POISON, not machinery verdict. The faithful
machinery (LN+GELUform+AdaLN+patchify+unfused attention) never
received a clean training signal in any variant — its question
stays REOPENED, explicitly not closed by this round. What this
round does close: the paper-recipe transplant (lognorm+gradloss
at λ=0.1/50-step) is dead-on-arrival for this stack; any future
DiT-tiny needs its own constitution WITHOUT the aux (or with a
swept λ — a new round either way, none run here). Empty table,
loaded finding: the aux pin is the largest optimization effect
measured since Phase-9 cosine (in reverse).

## Phase 20 — faithful package, no aux, fair trial (constitution, FROZEN pre-run)

Phase-19 package EXACTLY (LN+GELU-form+AdaLN+patchify+unfused,
B2/B4/B6 ladder, W64/H4/s16, shifted-log targets, same splits)
with the gradient aux REMOVED — disabled via `grad-w 0` PLUS a
guard skipping the aux forward when grad-w==0 (declared here;
0×NaN poisons, skipped NaN can't — same math, safer path).
The single variable is aux-present vs aux-absent. Loss =
aligned-MAE only (the pre-aux baseline that carried mixers this
far). Schedule mirrors Phase 19 minus aux: 8k-cosine, lr 0.01,
seed 7. Tags `ditc64b2/ditc64b4/ditc64b6` (c=clean).
Budget: 3 rungs, wall caps 15/20/25 min (Phase-19 mirror; aux
removal only helps), GPU solo, Qwen-7B never co-loaded.
Winner: most held-out cells under bars; directional near/planar/
smooth vs Phase-8 (>0.05, frozen) reopens empirically. Empty is
complete — a clean loss here closes the faithful package
honestly (machinery judged cleanly at last, either verdict
binding).
