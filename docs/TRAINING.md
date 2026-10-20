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

(TBD — instrumented run lands here before any extraction.)

## Base contract

(TBD — `scripts/train_base.py` API after the profile table.)

## Backfill

(TBD — choice + reason, parity proof paste.)

## What future rounds import

(TBD — import lines + config surface.)
