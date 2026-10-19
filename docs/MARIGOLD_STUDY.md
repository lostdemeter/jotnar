# Reverse-engineering Marigold v2: the process and what it proved

Twelve rounds (Phases 0–11) distilling a Marigold v2 depth teacher into
a geometric-assembly student. Zero passing students. The process is
the asset — every round banked evidence, and the evidence composes
into a measured boundary. This is that story, written so the method
transfers to the next target.

## The target (established, not assumed)

Marigold v2 repurposes frozen Qwen-Image-Edit-2509 with rank-128 LoRA
adapters (12 module families, measured in-tree — not rank 1) plus a
fine-tuned VAE decoder into a single-step depth predictor. Inference
closure, audited: 54GB backbone + 1.85GB trainables + constant
precomputed text embeddings (broadcast-identical every image, proven
by hash — no per-image information beyond RGB) + seed (VAE samples,
so bit-exact is the wrong bar by the authors' own docs; seeded
bit-exact confirmed 4/4 and 6/6). Runs here at ≤1024² (~19.5GB VRAM).

## The method (why this isn't guessing)

1. **Harness before student.** Strata (near/far, planar/nonplanar,
   tex/smooth) + affine-invariant comparison + margin bars at
   scaffold/5 — proven both directions *before* any candidate: seeded
   teacher-self passes with 5x+ clearance, a cheating-constant stub
   fails every stratum. An apparatus that can't say no is theater;
   ours says no 46+ times and counting.
2. **Constraints as carving.** Emitted-patterns table (architecture
   must compile), frozen constitutions (axes/budget/judge declared
   pre-run — mid-search invention bounces the round), determinism
   (retrain-match before any claim counts).
3. **Kills with evidence, never adjectives.** Perceptual: 7e-4 dynamic
   range vs 0.37 MAE (homeopathic at any stable weight). Sinkhorn:
   ~1e-5 influence (floor residue, not signal). Schedules: train moves
   freely, held-out shuffles bar-blind. Each kill names what would
   resurrect it.
4. **Process tripwires earned en route.** Banked numbers must postdate
   final-tree code (caught once: stale aux columns, bounced,
   regenerated — precedent now structural). Winner-consistency
   (passing cell + winner None fails loud). Spot-retrains by an
   independent executor (matches to 1e-15, 5e-05, 0.0, 7.2e-16).

## The rounds and their verdicts

| Phase | Question | Verdict |
|---|---|---|
| 0 | Teacher reproduces? | Yes — seeded bit-exact, fixtures banked |
| 1 | Harness falsifies both ways? | Yes — self passes, scaffold fails all six |
| 2 | Tiny in-table MLP? | 0/36 — needs context, not capacity (w32 identical) |
| 3 | 33px context mixer? | 0/36 — train fits, held-out flat (generalization wall) |
| 4 | Stolen signals (perceptual/Sinkhorn)? | Both killed with scale evidence |
| 5 | Frozen architecture battery? | 10 configs, empty, gate-counted |
| 6 | Long range (spokes to ~1025px, global summary)? | 0/120 — range moves train only |
| 7 | White-box anatomy — where is the knowledge? | norm_out.head 0.94 ≫ tail; nomination, not kill |
| 8 | Nominated joint-attention student? | 0/12 — underfits (train at/above scaffold) |
| 9 | Optimization budget? | Moves-but-bar-blind — steps won't close it |
| 10 | Multi-head at matched capacity? | Flat to ±0.003 — routing never the bottleneck |
| 11 | Finer grid? | 0 moves (6 worsen) — mechanism fires on train only |
| 12a | Synthetic analytic scenes — mimicry headroom? | NO-GO: headroom 1.47x < 2x; noise 0.00, bias 0.251 (5x texture hunger) |
| 13 | Deeper attention stacks? | 0 moves — depth buys nothing twice over (train degrades, held-out flat) |
| 14 | Patch-shuffle: global-structure dependence? | Teacher 1.27/dav2 1.33 collapse, locals don't — NEEDS JOINT REASONING |
| 15 | DINOv2-token inputs (info vs machinery)? | MACHINERY — train 0.50→0.054, held-out 0 moves; reader is the wall |
| 16 | Two-stage coarse-to-fine? | Dead on arrival (frozen codes 1e4 off RGB scale); CALL demand named |
| 17 | Textured-domain restriction? | Tie 7/96 (wall/floor mimicry); near never moves — candidacy dies at rooms without furniture |
| 18 | Liminal/edge concentration? | DEAD (1.03–1.22 vs ≥2 needed) — errors uniform everywhere |

## What the losses compose into

Train-fit descends monotonically with field, width, range, and steps;
held-out sits at 0.27–0.32 throughout. Best cells 1.9–2.1x over bars
in every round; texture/nonplanar detail worst everywhere. That is a
measured inductive-bias boundary: monocular layout reasoning needs
global structure no local mixer and no small attention has shown.
The teacher's measured anatomy (rank-128 adaptation through the whole
DiT + tuned decoder) prices the gap from the other side.

## The two options (one now opening)

1. **Change constraints — OPENING NOW.** The in-table arc is exhausted
   *including* stack depth, dynamism-across-passes, domain restriction,
   and edge-structure hypotheses (Phases 12a–18). Table gaps under
   examination: LAYERNORM/GELU/CONV patterns, fused attention,
   multi-output kernels, halo exchange, adaptive control flow (CALL
   lowering). Each gap gets priced (design + parity-gate shape + size
   + order) before anything is built — scoping round first,
   implementation only by signature.
2. **Accept.** Fixtures, harness, trainers, 18 rounds of negatives,
   tripwires — all banked, gated, and waiting. The dav2 geometric
   replica already supplies depth today; the student was the
   smaller-faster play. Nothing evaporates; the gates re-prove the
   closures every run.

## Why the process transfers

Black-box target → reproducibility proof → falsifying harness →
constraint-carved search → white-box anatomy → nominated build →
bounded batteries → exhaustion with an earned sentence. Nothing in
that chain mentions depth. The next target inherits the harness
pattern, the constitution pattern, the retrain-verified standard,
and the tripwires — the machinery is the deliverable the losses paid
for.
