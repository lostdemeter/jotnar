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
