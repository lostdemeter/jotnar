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
