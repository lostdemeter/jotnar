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
each, single GPU to itself. Loss fixed to aligned-MAE (signals dead).
Judge: Phase-1 bars, same alignment, verdict-reported. In-table only.

Frozen axes (each config states receptive field in pixels):
- field: 33 (dense3+s4/17+s8/33, 177 dims) → 65 (+s16/65, 252 dims)
  → 129 (+s32/129, 327 dims). Strided sampling (documented honest
  sampling, cf. Phase 3); averaging would need parked POOLAVG.
- mixer: W64/d2 (reference shape) × W128/d2, W64/d3, W32/d2.
- granularity: densewin 3 → 5 at field 33.
- scale-count: (covered by field steps — dropping scales untested
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
integrity, not all 10. Empty battery (nothing clears bars) is a
complete report with the table banked.
