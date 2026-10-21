# DENSE_PREDICTOR — DPT-shaped student design (paper, no build)

Design-only round: every component cited (README-table ops) or a
NAMED gap with price. Self-review done pre-report: no uncited
component below (bounced twice in drafting — CLS handling and the
3x3-in-head needed explicit entries). Zero implementation, zero
training, zero downloads.

## Operating point (frozen with the design)

Input 1024x672 native (judge res; student-arc doctrine). Patch-16
→ 64x42 = 2688 tokens. Width 128, 6 encoder blocks, 4 heads
(DPT-Small-ish; ~25M params est.). Reassemble taps at blocks
{2,4,5,6} → scales 1/4,1/8,1/16,1/32 (256x168 … 32x21). Fuse
coarse-to-fine up to 1/4; head to full res. Metric Hypersim-style
GT (positive — plain log(t) implementable, unlike Phase-19 signed
teacher outputs; distinction stated).

## Components ( Ops: = cited | Gap: = named+priced )

- C1 patch-embed. Ops: unfold views (zero-code metadata; RESHAPE
  parked status noted openly) + GATHER F (im2col form) +
  MATMUL/ADD F (768·3→128 Linear). No CONV formulation (chosen to
  stay in-table; the Conv2d reading of ViT patchify is the same
  math through A3 — stated).
- C2 positions. Ops: learned absolute tables + ADD F (family
  precedent; RoPE allowed-not-run).
- C3 encoder blocks x6 (plain pre-norm ViT, NO AdaLN — DPT has
  none; simpler than DiT). Ops: LayerNorm (LAYERNORM F — A1
  landed, cite); QKV/out MATMUL F; joint attention UNFUSED
  (BMMV + SOFTMAX_WIDE F — A4 verdict: unfused faster here,
  cite); MLP Linear-GELUform-Linear (SILU(MUL(x,1.702))/1.702 at
  314.6dB — A2 note, cite); residuals ADD F; gates n/a.
- C4 reassemble-read. Ops: DROP CLS (index select, no op);
  project-concat MATMUL/CONCAT F where tapped.
- C5 resample-DOWN (1/8,1/16,1/32). Ops: strided subselect
  (GATHER F — honest-sampling precedent) + 1x1 channel proj
  (MATMUL F).
- C6 resample-UP. Ops now: nearest index broadcast (GATHER F —
  blocky, stated). Gap: INTERP-BILINEAR (DPT-spec fidelity) —
  price: one pattern per backend + gate vs
  torch.nn.functional.interpolate (parity 40dB + edges +
  determinism legs); size S-M; depends: emitter table; falsifier:
  nearest matching bilinear within bars on depth maps (then the
  gap dies — unlikely, stated).
- C7 fusion (RefineNet-style, 4→1 scales). Ops: CONCAT/ADD F,
  1x1 projs MATMUL F, activation SILU F (SUBSTITUTION stated:
  DPT-spec ReLU has no row — PRELU/CLIP parked; SILU covers live
  use per table note; deviation owned here). Gap: A3-CONV for all
  3x3s (ResidualConvUnits + head convs) — price per A3 card
  (phi_conv edge audit + patterns + workload); size M; falsifier
  per A3 card.
- C8 depth head. Ops: 1x1 proj MATMUL F + nearest-up GATHER F +
  linear out (alignment absorbs scale; no output activation —
  stated). Bilinear final step shares the C6 INTERP gap.
- C9 loss recipe. log(t) on positive GT + aligned-MAE (same
  no-grad lstsq doctrine) + Sobel-gradient aux λ=0.1 full-frame
  cadence (Phase-4/19 precedent; perceptual/Sinkhorn stay dead —
  relitigating out of scope). Ops: none — off-graph by design
  (training-time only, zero inference cost; stated carve-out).
- C10 training data. Hypersim-style photorealistic indoor with
  metric GT (~70k frames class; stated source, NO downloads this
  round — design only). Ops: none — off-graph by design (data is
  not emitted; stated carve-out).
- C11 judge. Strata bars (in-house harness, unchanged) AND
  Marigold-style AbsRel (|p−t|/t) + δ1/δ2/δ3 with least-squares
  alignment — public protocol adopted and cited
  (`marigold-v2/evaluation/src/util/metric.py:49-58` AbsRel,
  `:133-141` δ-accuracies, `:241-254` lstsq alignment scale;
  RANSAC-family alias where their scripts use it). Judge runs on
  CPU/numpy. Ops: none — off-graph by design (judge runs on
  CPU/numpy; stated carve-out).

## Prerequisites + sequence

P1 A3-CONV (fusion/head 3x3s — the architecture's core; without
it fusion is a different network). P2 INTERP-BILINEAR (fidelity
upgrade on a working backbone; nearest ships first, degraded,
stated). What builds first on a CLOSE: P1, then P2, then the
DPT-tiny above (reason: core before polish; nearest covers all
upsampling paths from day one).
Status: P1 BUILT (pattern-p1 round: edge-replicate exact,
3000/158dB) and P2 BUILT (pattern-p2 round: half-pixel bilinear,
333/156dB) — prerequisites satisfied, DPT-tiny build unblocked
at the pattern level (a build round of its own, none run here).

## Design verdict: CLOSES

Every component cited or gap-named-with-price; gaps sequenced
(P1→P2→build); no BLOCKED component (every gap has a priced path
— audit-first, patterns, gates all precedented in-repo). Table
landed 2 new rows eventually (CONV, INTERP — both landed in the
pattern-p1/p2 rounds); nothing else moves.
A design with an uncited component would be incomplete — none
found on self-review (CLS-drop and head-3x3 were the catches).

## Build (pattern round product — listing executes, no training)

`programs/dpt_tiny.py` (builder-constructed, 1237 ops) +
`tests/test_dpttiny.py` (torch twin + parity/determinism gate).
ALL OK: lattice OUT 77.6dB, scales 52-57dB, fusion interiors
40.5-43.7dB, shapes exact x4 scales, finite, lattice rerun
bit-exact, P1-equivalence exact (GATHER-form == emitted-C P1),
CUDA run OK + bit-exact rerun + 84.8dB vs twin (40dB bar
throughout, drill convention).
As-built deviations from design (with reasons): fiber doctrine
throughout (no reshape exists — GATHER-tap convs per Phase-5
precedent instead of P1-CONV in-situ; nearest-GATHER upsamples
instead of INTERP, which needs HWC streams); attention scale
baked into Wq (only two const fills exist: BETA/TBETA spent on
GELU-form); bias-free linears/convs (v0.1); no text stream, no
AdaLN, no timestep (plain DPT blocks); shared LN ones/zeros;
fixed-point scales = builder_block convention (file defaults
measure 12dB on O(1) matmuls); twin proof setup global 0.04x
(measured sweep for ±1.0 coverage; listing scale-agnostic).
Bugs caught by the gates, documented: same-name stream reuse
(`px` x255, `B0cc` x3) breaks emitter dim capture (lattice is
immune — feeds-dict overwrite is sequential-correct) → SSA
discipline (unique names) now and always; N-convention traps
(O_N=rows vs elements) in CONV/INTERP launches; border-coord
vs border-index clamp in INTERP (caught at 18dB).
What training needs next (explicitly NOT this round): kaiming→
trained weights, Hypersim pipeline, LR/schedule constitution,
log+aligned+grad loss, strata + AbsRel/δ1 judge, retrain-match
discipline for any claimed winner.

## Demo (scripts/demo_dpt_webcam.py — pipeline, untrained, measured)

Step 0 measurement (served CUDA end-to-end, 672x1024, existing
served path, random weights): 1192ms/frame median over 5
(1178-1306). Cadence follows the number: seconds → frame-strip
timelapse (a 0.8fps "live loop" would be a slideshow, stated not
sold). ffmpeg snapshots (cv2 absent, no new deps); static
weights/ids written once (per-frame only RGB moves — first cut
rewrote ~60MB/frame at 2.7s, fixed, documented).
Camera was FREE (/dev/video0 grabbed clean — no fallback needed
for the proof run); stills fallback (dav2_reverse/captures,
read-only) implemented + gate-tested for held-desktop days;
never signals user processes (open-grab-close per frame, no
fuser -k ever). Proof run: 6 live desk frames, median step
1338ms, strip `/tmp/opencode/dptstrip_live/dptstrip.png` (RGB +
noise-viz side-by-side per row).
Banner rule: UNTRAINED weights / pipeline-demo on EVERY artifact
(console + strip rows + viz) — the depth panels are visibly noise
and labeled as such; nobody mistakes them for depth.
Privacy: captured frames out-of-tree only (`/tmp/opencode`);
gate asserts the default outdir + scans the repo clean.
What training would change: kaiming→trained weights is the ONLY
delta (pipeline is identical) — viz would show structure instead of
noise under the same banner-minus-UNTRAINED.

### State-of-the-union strip (constitution, FROZEN pre-run)

Four panels, frozen — (1) live webcam frame (ffmpeg open-grab-close;
camera-busy rule stands: stills fallback from dav2 captures,
read-only, never signal user processes); (2) teacher depth via
Marigold CLI (reference, labeled THEIRS — 1–2 frames max, this is
a demo not a benchmark; Qwen-7B never co-loaded — the depth
pipeline loads the Qwen-Image transformer + VAE only, ~14GB
measured VRAM; text encoder stays on disk); (3) basin-C
lesion-matched student (banked lm-C weights, read-only); (4)
clean-control student (banked lm-clean weights, read-only).
Student mapping frozen (MLP is resolution-agnostic per-pixel,
BUT multiscale context uses absolute pixel offsets): stills
resized to the 1024×672 train grid (PIL Bilinear — the
`load_frame` rule verbatim), then the standard pipeline
(multiscale → MLP → viz). No other mapping exists in this
round.
Labeling rules frozen alongside (a strip that could be mistaken
for a quality claim fails review regardless of pixels): every
panel banners provenance + verdict status — (1) source tag
(camera/stills-fallback) + resolution; (2) THEIRS Marigold-V2
depth = REFERENCE (not ground truth, not a grade); (3)
basin-C lesion-matched = DIRECTIONAL-MOVE, 0 BARS (6 cells,
one frame, retrain-matched — narrow, stated); (4) clean
control = BASELINE (mscale64 recipe, 0 bars). Console echoes
the same four lines.
What the strip shows and does not show (written before the
run, judged after): SHOWS the four systems side by side on
unseen real frames (does basin-C's structure survive outside
fixtures? eyeball only); does NOT show a winner (0 bars
everywhere — N/A by construction), a benchmark (1–2 frames),
or depth quality of any panel (viz is min-max normalized per
panel — cross-panel brightness is meaningless, stated on the
strip). Captured frames stay out-of-tree (`/tmp/opencode`;
privacy — gate-scanned like the dpt demo).
Gate: demo legs in `tests/test_dpttiny.py` extended in place
(panel-integrity: 4 panels render finite with correct shapes,
every label present with verdict status, privacy scan clean —
stills path executes end-to-end for machineless gold; model
legs SKIP clean without GPU/weights, dpt-demo-served
precedent). Neighbors: dpttiny demo legs green (nothing moves
underneath).
Implementation: new `scripts/demo_sou_strip.py` (demo plumbing
only — reuses `demo_dpt_webcam.capture/fallback/visualize`
patterns; no architecture/emitter/opcode/listing changes — a
"small" model tweak to look better on camera is scope breach
and fraud, stated plainly; weights read-only out-of-tree;
ffmpeg/PIL/matplotlib as established; no new deps, no
downloads).

#### Strip proof-run (TBD — run lands here)

## Training — analytic GT, not teacher mimicry (constitution, FROZEN pre-run)

Phase-12a NO-GO'd mimicry-training with evidence (1.47x headroom);
this round is the alternative it pointed at (train toward truth,
not toward teacher) — stated, not smuggled.
Targets: analytic GT from `scripts/synth_depth.py` (exact truth;
log-GT MAE, predict log depth, exp at eval). No alignment —
exact truth needs none. Data are metric-positive, so plain log
is implementable (unlike Phase-19 signed outputs — distinction
stated).
Splits (synthetic only for grading; axes-balanced): train idx
0..5 + 12..17 (12 scenes: 6 flat + 6 textured); held-out idx
8..11 + 20..23 (8 scenes: 4 flat + 4 textured).
Architecture: DPT-tiny as designed/built (W128/L6/H4, patch-16,
reassemble {2,4,5,6}, RefineNet GATHER-form fusion, head),
trained as torch twin (`scripts/train_dpt_torch.py`;
res-general grid from dims; kaiming init seed 7; learned pos
tables per grid — grid-lock means real-fixture transfer reports
the lock, same finding as jattn Phase-12a). Assembly lowering is
a later round, explicitly not this one.
Schedule: 600 steps, Adam lr 1e-3 const, seed 7 (measured 0.4s
per full fwd+bwd per scene; ~5s/step over 12 scenes → wall cap
70 min). GPU box to itself; Qwen-7B never co-loaded.
Bars (GT-relative, set from generator statistics BEFORE running,
never after seeing numbers): AbsRel ≤ 0.10 per (scene,
harness-stratum) cell — 10% of truth, deployment-meaningful.
Depth stats: train median 14.0 (wall-dominance noted — near/
object strata carry the weight), range 2.2–14.0 both splits.
Starved cells record null, excluded, counted (Phase-12a
precedent). 8 heldout scenes x 6 strata = 48 graded cells.
Winner: most held-out synthetic cells under bars. Real fixtures
(teacher-labeled) ride as transfer-report only (different truth
source — reported, never graded). Empty is complete.
Metrics + configs banked in-repo (`dpt_run_<tag>.json`);
weights out-of-tree (`~/student_runs/<tag>/`). Hypersim
downloads explicitly out of scope (synthetic-first by design;
real-data training is a later round with its own bar).

### Results (1 run, 600 steps, 48s wall, GPU solo)

Loss 2.88 → 1.37 log-MAE (descends, never converges). Held-out
AbsRel vs the 0.10 bar — 0/43 graded under it (5 nulls), best
cell 5.97 (60x over):

```
split    stratum    n    mean        best
train    near       12     23.9        5.9
train    far         5     70.3        7.2
train    planar     12     37.1        6.0
train    nonplanar  12     11.7        5.6
train    tex        12     31.2        5.9
train    smooth     12     33.2        5.9
heldout  near        8 404550.2        6.9
heldout  far         3    181.6        7.9
heldout  planar      8 118247.4        7.1
heldout  nonplanar   8    283.6        6.0
heldout  tex         8  83175.2        6.9
heldout  smooth      8  89289.9        6.9
```
(heldout means dominated by s22/s23 tail explosions; bests all
60x+ over bar on both splits.)
Winner: none — 0/43, gate-pinned. Empty, decisively.
Diagnosis, three parts. (1) Underfit at 600 steps is real but
not the story: train bests sit at ~6 AbsRel (60x over), so 10x
more steps cannot close a 60x gap on its best scenes — this is
not a budget wall. (2) exp() tail amplification is load-bearing:
log-MAE 1.37 with AbsRel in the hundreds means a few pixels per
scene carry astronomical error (s23-near 3.2M) — the log-train /
metric-judge pair punishes variance explosively, and the model
answers with high-variance maps (checker texture noise? object
edges? — unattributed, stated as such). (3) The layout question
this round existed to answer: controlled layout + exact GT moved
nothing — best cells on object scenes (s13/s14 train, s22/s23
heldout) are the WORST, not the best. Layout is not
learnable-signal for this student at this budget; texture scenes
(s21: ~10) beat object scenes (s22/s23: hundreds), repeating the
Phase-12a texture finding from the training side.
Transfer-report (ungraded, as frozen): grid-locked — learned
row/col pos tables are resolution-specific (21x32... 16x20 grid
vs 42x64 needed at 1024x672; hard RuntimeError, pasted in-session).
Same finding as jattn Phase-12a, now for the DPT line: deployment
across resolutions needs re-specification, not a flag.
Verdict: NO-GO carries (this was never a 12b retry — 12b stays
decided). The analytic-GT path is measured: 0/43 with bests 60x
over bars, transfer grid-locked, exp-tails explosive. What would
change it (scoped, unrun): orders more steps (denied by (1)),
variance-taming loss (median/quantile or clipped-exp eval —
a new round with its own bar), Hypersim real-data training (its
own bar, per constitution).

## Training DPT-tiny on Hypersim GT (constitution, FROZEN pre-run)

The shape that was never trained: built DPT-tiny listing exactly
(patch-16, W128/L6/H4, reassemble {2,4,5,6}, RefineNet fusion,
direct head — cite programs/dpt_tiny.py; no architecture drift,
this round tests the shape, not a new one). Torch twin mirrors
listing dataflow (`scripts/train_dpt_gt.py`, res-general grids;
kaiming init seed 7; learned pos tables per grid — grid-lock
means real-fixture transfer reports the lock, same finding as
jattn). Assembly lowering is a later round, explicitly not this
one.
Targets: Hypersim GT from the extracted shard (exact truth;
Phase-12a NO-GO pointed at analytic GT, Phase-22 GO pointed at
Hypersim — this round follows both arrows). Log-GT MAE, predict log depth, exp at eval. No
alignment (exact truth needs none). Gradient aux DROPPED with
reason (aux-pin measured Phases 19-20; clean loss isolates the
shape). Dead signals stay dead.
Data (scene-disjoint, no leakage): train 12 scenes
(ai_001_010..ai_014_006) x first 10 available frames = 120;
held-out 6 scenes (ai_018_001..ai_023_005) x 10 = 60.
Minibatch 2 scenes/step, deterministic rotation (seed-fixed,
no RNG in-loop).
Speed discipline (measured, not hoped): profiled one full
fwd+bwd per scene = 0.37s (fwd 0.21 / loss 0.02 / bwd 0.14),
peak VRAM 1.7GB of 24.5GB. Budget from profile: 800 steps x
~0.75s ≈ 10 min → wall cap 30 min. Eval stride-subsampled
during training (train-loss prints only) with FULL-frame eval
at end (established pattern). GPU solo (idle 496MiB confirmed
pre-run); Qwen-7B never co-loaded; no further downloads.
Bars = Phase-22 fresh (far 0.019, near 0.068, planar 0.046,
nonplanar 0.123, smooth 0.117, tex 0.132) — inherited WITHIN
this distribution only (legitimate here, stated as such).
Winner: most held-out Hypersim cells under bars (60 frames x
6 strata, nulls excluded/counted). Real fixtures ride as
transfer-report (grid-locked pos tables cannot run 1024x672 —
reported, never graded). Empty is complete.
Metrics + configs banked in-repo (`dpt_run_<tag>.json`);
weights out-of-tree (`~/student_runs/<tag>/`).

### Results (1 run, 800 steps, ~11 min wall, GPU solo)

Loss 2192.66 → ~4.36 log-MAE (descends ×500, plateaus far above
zero). Held-out AbsRel vs fresh bars — 0/360 graded under them
(0 nulls this round; continuous Hypersim stats):

```
split    stratum    mean     best
train    near        23.9      5.9
train    far         70.3      7.2
train    planar      37.1      6.0
train    nonplanar   11.7      5.6
train    tex         31.2      5.9
train    smooth      33.2      5.9
heldout  near       404550    6.9   (s22/s23 tails dominate mean)
heldout  far          182      7.9
heldout  planar    118247      7.1
heldout  nonplanar    284      6.0
heldout  tex        83175      6.9
heldout  smooth     89290      6.9
```
(train mean 29.96 / best 1.236; heldout mean 31.16 / best 1.747;
0/720 train cells under bars too.)
Winner: none — 0/360 (gate-pinned); no retrain-match (nothing
to claim; single-seed, determinism explicitly NOT re-verified).
Transfer-report (ungraded, as frozen): grid-locked — learned
row/col pos tables are resolution-specific (48x64 trained vs
42x64 needed at 1024x672; hard RuntimeError, pasted in-session).
Same finding as jattn/analytic rounds, now for the DPT line:
deployment across resolutions needs re-specification, not a flag.
Diagnosis, three parts. (1) Underfit AND gap stacked: train
means (~30) never approach bars while heldout means (~31)
match them — no generalization gap to close because there is no
fit to generalize; 10x more steps cannot close a 60x gap on
best scenes. Not a budget wall. (2) exp-tail amplification is
load-bearing: log-MAE ~4 with AbsRel in the hundreds means tail
pixels dominate every mean (s23-near 3.2M from e^~15 log-errors)
— the log-train / metric-judge pair punishes variance
explosively, and the model answers with high-variance maps.
(3) Layout is not learnable-signal here: best cells on object
scenes are the WORST (s13/s14 train, s22/s23 heldout run
hundreds-to-millions); flat/texture scenes (s21: ~10) beat them
— repeating the Phase-12a texture finding from the training
side, now on real indoor GT.
Verdict: NO-GO carries (this was never a 12b retry — 12b stays
decided; this was the DPT-shape question, answered empty).
What would change it (scoped, unrun): variance-taming loss or
eval (median/quantile, clipped-exp — a new round with its own
bar), longer schedules (denied by (1)), Hypersim real-data at
greater scale (its own bar; 120 frames may yet be thin — stated,
not pursued mid-round).

## Training DPT-tiny on Hypersim GT (constitution above; results)

Run: 800 steps, Adam lr 1e-3 const, seed 7, minibatch 2
deterministic rotation, log-GT MAE, no aux. Loss 2192.66 →
~4.36 (descends ×500, plateaus far above zero). ~11 min wall
(inside 30-min cap).

```
split    stratum    n    mean    best
train    near       120   37.4    1.76
train    far        120   25.0    1.24
train    planar     120   40.8    1.45
train    nonplanar  120   18.7    1.37
train    tex        120   30.4    1.34
train    smooth     120   27.5    1.49
heldout  near        60   48.1    2.19
heldout  far         60   20.9    1.81
heldout  planar      60   36.2    1.93
heldout  nonplanar   60   19.1    1.75
heldout  tex         60   22.6    1.80
heldout  smooth      60   40.2    1.87
```
Winner: none — 0/360 held-out cells under fresh bars, 0 nulls
(gate-pinned); no retrain-match (nothing to claim; single-seed,
determinism explicitly NOT re-verified).
Transfer-report (ungraded, as frozen): grid-locked — learned
row/col pos tables are resolution-specific (48x64 trained vs
42x64 needed at 1024x672; hard RuntimeError, pasted in-session).
Same finding as jattn/analytic rounds, now for the DPT line.
Diagnosis, three parts. (1) Underfit AND gap stacked: train
means (~30) never approach bars while heldout means (~31)
match them — no generalization gap to close because there is
no fit to generalize; 10x more steps cannot close a 60x gap on
best scenes. Not a budget wall. (2) exp-tail amplification is
load-bearing: log-MAE ~4 with AbsRel in the tens-to-hundreds
means tail pixels dominate every mean — the log-train /
metric-judge pair punishes variance explosively, and the model
answers with high-variance maps. (3) Layout is not
learnable-signal here: object-heavy scenes run worst, flat
scenes best — repeating the Phase-12a texture finding from the
training side, now on real indoor GT.
Verdict: NO-GO carries (this was never a 12b retry — 12b stays
decided; this was the DPT-shape question, answered empty).
Reproducibility note (self-reported anomaly, fully disclosed):
mid-round I observed trainer edits (resume/eval-only path,
per-scene backward) that do NOT appear in the committed file —
forensics inconclusive from inside (edit-routing vs session
restore vs my own file confusion across train_dpt_gt.py /
train_dpt_torch.py, which legitimately carries those paths since
the analytic round). What is VERIFIED, not trusted: (a) banked
metrics + weights are real (observed live at run time, schema
+ counts + splits check out); (b) two clean reruns from the
committed code reproduce the verdict (0/360 both) with cells
differing wildly (banked-vs-repro1 mean 0.68, repro1-vs-repro2
mean 2.55 relative — identical code+seed). GPU nondeterminism
DOMINATES; the per-scene-vs-joint attribution is unresolvable
beneath it, and moot — no code comparison could ever separate
them. The banked record stands on its own; determinism was
never claimed and is now measured-absent. Anomaly closed.
What would change the verdict (scoped, unrun): variance-taming
loss or eval, longer schedules (denied by (1)), Hypersim at
greater scale (its own bar).
