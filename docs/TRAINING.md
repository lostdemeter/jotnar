# Training harness — one place for how student rounds train

## Constitution (frozen before extraction; violations bounce)

Seven hand-rolled trainers (`scripts/train_{student,attn_student,
refine_student,dit_student,dpt_torch,dpt_gt,hyper}.py`) share one
shape: seed → preload frames → stride-sample → Adam loop with cadence
prints → weights out-of-tree → full-frame strata eval at end → metrics
banked in-repo. The harness extracts that shape once. Rules:

1. **Profile-first.** One instrumented run of an existing trainer before
   any extraction. The run prints forward/backward/eval/dataloader/aux
   shares; the harness cuts the biggest slice first, stated from the
   numbers (table in §Profile below). No slice is cut on intuition.
2. **Shared base, one file.** `scripts/train_base.py` (placed with the
   trainers it serves — same import surface as every trainer's
   `sys.path` shim). Contract pieces, all exercised by the backfill:
   deterministic seeding (torch+np, fixed order), stride sampling
   (fixed order, no RNG in-loop), eval cadence with full-at-end
   (mid-run eval is print-only, never changes math), metrics-banking
   schema (tag/arch/config/torch/device/weights/nparams/frames, same
   keys the batteries read), weights-out-of-tree discipline
   (`~/student_runs/<tag>/weights.pt`, never in-repo), gate-hook
   outputs (loss prints + banked JSON the existing harness reads).
3. **Backfill EXACTLY ONE trainer** onto the base as the parity proof.
   Choice + reason in §Backfill. The other five stay frozen — migrating
   proven-gated code for tidiness is churn with risk, explicitly out
   of scope. The backfilled script keeps its module-level API
   (`load_frame`, `_define_model`, `scales_for`, `multiscale_features`,
   `scene_frames`, `load_pair` — other trainers/tests import them).
4. **No science changes.** Same losses, same math, same RNG draw order.
   A "small improvement" to any loss is scope breach, not a bonus.
   No emitter/opcode/assembly changes anywhere near this work.
5. **Bars.** Determinism: same config twice → identical metrics.
   Parity: harness-run vs hand-rolled-run identical on the backfilled
   trainer's smallest config (paste), pinned forever by banked-number
   regression in the gate (transitive parity, stated). Budget: profiled
   shares sum sanely + the cut slice measured, not asserted. Neighbor:
   the backfilled trainer's behavior gate stays green; the only
   existing-results change allowed is none.
6. **Box discipline.** GPU solo for timed runs (no co-runners, pasted
   `nvidia-smi`). Qwen-7B never co-loaded with training. CPU/GPU-fast
   gate: no full training runs in tests (smallest configs only).

## Profile

Method (zero trainer-file changes): `CUDA_LAUNCH_BLOCKING=1 python3 -m
cProfile` on `scripts/train_hyper.py --arch mscale` — blocking
serializes async kernels onto their launching calls so cProfile
attributes GPU work honestly; shares are total-work shares (overlap
ignored), stated. GPU solo (empty compute-app list at launch, pasted
in build notes). Slice: `--steps 30` covers the full 120-frame train
rotation once (scenes rotate deterministically `(it*mb+k) % 120`) plus
the full 180-frame end-eval — the exact per-unit work of the 300-step
run. Tag `hyper-prof` (scratch; banked `hyper_run_hyper-mscale.json`
untouched, scratch JSON removed after).

Calibration (unblocked, wall): 2 steps = 10s train (5.0s/step →
~1500s for 300, matches banked 1451s); end-eval 180 frames = 228s;
startup ≈ 95s. Projected full run ≈ 1780s (train 84% / eval 13%).

Instrumented shares (30 blocked steps + full end-eval, tottime 372.6s):

```text
dataloader   326.5s   87.6%   (multiscale_features 320.7s/300 calls + PIL/PNG)
h2d           35.4s    9.5%   (torch.tensor host->device copies)
forward        2.4s    0.6%   (linear/silu — 15k-param MLP)
backward       1.0s    0.3%   (run_backward + per-scene lstsq align)
eval           3.8s    1.0%   (strata_masks/align_mae/gradmag proper)
other          3.4s    0.9%
```

Per-call multiscale = 1.07s (320.7/300): train loop burns 1200 calls
(~1280s of the 300-step run — recomputed from scratch every scene
every step), end-eval burns 180 calls (~192s of its 228s). It is a
deterministic function of (image, field, densewin) — no RNG, no
weights — so the hoist changes no math.

**The cut, stated from the numbers: feature cache first.**
`FeatureCache` (disk, keyed by frame id + featurizer name/version/args;
versioned keys refuse stale) hoists deterministic featurization out of
the step loop AND the end-eval. Projected cut ≈ 1280s train + 192s
eval ≈ 1470s of ~1780s (≈ 83%). NOT cut: forward/backward (0.9% —
cutting means touching science, out of scope); eval cadence stays
full-at-end (the remaining ~36s eval proper is the price of banked
metrics, kept deliberately). No aux exists in hyper-mscale (0%).

## Base contract

`scripts/train_base.py` (with the trainers, same `sys.path` shim). No
science inside — no losses, models, or metrics math:

- `seed_all(seed)` — torch+np, torch-first fixed order (matches all 7).
- `pick_device()` — cuda-iff-available + the `device=` print.
- `stride_index(n, stride=7, device)` — fixed-order sample, no RNG.
- `apply_schedule(opt, kind, lr, it, steps, warmup=400)` — const /
  cosine / linear / warmcos (covers every trainer's schedule; unknown
  kinds raise). Values verified against the hand-rolled formulas.
- `cadence(it, steps, every)` — print/eval cadence + always last step.
- `PhaseTimer` — `time(name)` blocks over dataloader/forward/backward/
  eval/aux; `summary()` prints per-run shares (profile-first as habit).
- `FeatureCache(root, featurizer, version, args_fp)` — disk `.npy`
  cache keyed by frame + versioned featurizer identity (stale math can
  never hit); same bytes in or out, timing only.
- `save_weights(state_dict, tag)` — `~/student_runs/<tag>/weights.pt`,
  never in-repo.
- `bank_metrics(rep, path)` — schema-enforced (`tag/torch/device/
  weights/frames` present, weights out-of-tree) + indent-1 write.

## Backfill

Choice: `scripts/train_attn_student.py`, with reason. Cheapest
full-contract exercise: stride-7 sampling, const+cosine schedules,
deterministic seeding, weights-out-of-tree, strata metrics banking,
full-at-end eval — all six contract pieces fire on its smallest config
(`--steps 2 --width 8 --blocks 1 --heads 1`, ~3s CPU). Rejected:
`train_student` (aux/scipy/transformers branches — widest, not
cheapest), `train_refine_student` (needs banked pass-1 weights),
`train_dit_student` (8000-step heavy), `train_dpt_torch`/`train_dpt_gt`
(no stride sampling, no schedules — half the contract unfired),
`train_hyper` (the profiled subject — don't touch the measured thing).
Bonus: its model (`_define_model`) + `load_frame` are shared by
`train_refine_student` and hyper-jattn — the first client sits at the
reuse center. The other five trainers stay frozen (out of scope).

Diff (plumbing only, statement order preserved): seed/device/stride/
schedule/cadence/weights-path calls routed to base (identical values —
const sets the same lr Adam already holds); `PhaseTimer` blocks wrap
forward / align+backward / end-eval (no reordering, no RNG, no math);
`bank_metrics` replaces the raw `json.dump` (same keys, same indent).
Module API (`load_frame`, `_define_model`) preserved — refine imports
them (gated). Stdout-only deltas, stated: phase-share lines added,
`wrote` prints the absolute path.

Parity proof (smallest config, CPU; tracked `attn_run.json` guarded +
restored, scratch tags, weights out-of-tree):
hand-rolled run vs harness run — losses `0.5286 / 0.5255` both ways,
all 36 strata match, max-abs-diff **1.25e-09**. Run-to-run CPU floor
(identical binary twice) measured **2.4e-08**, so the backfill sits an
order of magnitude UNDER process noise; gate bar is house float=eps
(1e-6, 1000x margin). Weights `.pt` bytes are compared by effect, not
hash (same CPU-noise floor). Gate banks the harness numbers
(`tests/test_train_harness.py` ANCHOR) — hand == banked at backfill,
banked == now in-gate; any future science change goes red there.
Existing-results change: none (guard byte-verified in-gate).

## What future rounds import

```python
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import train_base as BASE

BASE.seed_all(args.seed)
device = BASE.pick_device()          # (+ BASE.log_gpu_mem() on GPU rounds)
sidx = BASE.stride_index(n, 7, device)
timer = BASE.PhaseTimer()
for it in range(args.steps):
    BASE.apply_schedule(opt, args.schedule, args.lr, it, args.steps)
    ...
    if BASE.cadence(it, args.steps, 500): ...
cache = BASE.FeatureCache("~/student_runs", "multiscale", 1, "33-3")
x = cache.get(frame_id, lambda: multiscale_features(im, 33, 3, False))
BASE.save_weights(net.state_dict(), args.tag)   # out-of-tree
BASE.bank_metrics(rep, path)                    # schema-enforced
```

Next trainer to convert (when a round touches it anyway): `train_hyper`
— the profiled one; its per-scene `multiscale_features` call becomes a
`cache.get`, projecting ~1280s off the 300-step wall with zero math
change. Not done here (frozen per constitution rule 3).
