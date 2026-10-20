# Backlog — queued work, in order

## 2. DPT-tiny served-step profile, burn-down vs 33ms (measurement round)

Constitution (written BEFORE any measurement — violations bounce;
scoping-round discipline: implementation only by this signature, no
impl beyond what is declared here):

- Method (FROZEN): per-op CUDA event timing via EXISTING served
  machinery — `compile_program(..., time_ops=True)` wraps each op body
  with `cudaEventRecord/sync/elapsed` and prints `op MN ms` to stderr
  in program order (1237 ops → 1237 lines). Reason: it measures the
  actual served kernels (right backend/shapes/launches), not a proxy
  that would need a scaling fudge (fudge voids comparability); nsys is
  not installed; precedent `serve_decode.py` (`QWEN_TIME_OPS=1`).
  Perturbation stated: one event-sync per op (~10µs × 1237 ≈ 12ms on a
  ~1192ms step, ~1%) — the reconcile leg bounds it.
- NO listing/emitter/opcode changes. `time_ops` already exists, so NO
  profiler extensions are needed; stderr parsing lives in test-side
  analysis only. Any emitter edit or new profiler flag voids this
  constitution (no-impl tripwire).
- Served total re-measured — the 1192ms demo-round number is NEVER
  inherited blind. Fresh workdir `/tmp/dptprof/` (never `/tmp/dec7b`,
  never touching `/tmp/dpttiny_cu` the demo gate reads); two fresh
  binaries from the same listing revision (plain + timed); 5 frames
  each, median; deterministic zero-RGB 672×1024 input (pipeline timing
  is input-shape-bound, not content-bound); random kaiming weights
  (`test_dpttiny.gen_weights`, seed 0 — training state unchanged, this
  profiles the pipeline not the model); GPU solo for served runs
  (idle + VRAM stated); Qwen-7B never co-loaded.
- Honest unit (DECLARED): raw = per-op stderr lines (complete iff 1237
  parsed); table = by-mnemonic aggregates (same kernel pattern per
  mnemonic) with n_ops/total-ms/share — every op counted, provenance
  per row (`time_ops` event). Mixing methods mid-round voids
  comparability — this section is the freeze point.
- Target: 33ms frame (1192/33 ≈ 36× must shrink; "40×" is shorthand —
  the verdict uses the re-measured ratio). Winner rule: N/A —
  deliverable is the table + burn-down + verdict.

### 2a. Burn-down (lands here with evidence)

## 1. Geometric code profiler (BUILT on `exp/geo-profiler` — evidence below)

**Why:** every speed round hand-rolled its own profiling (trace-`sec`
attribution, intercept fits, N-sweeps, `/tmp/profrep` workdirs,
`bench/matmul_decode.cu`). Unify into one tool; aim it at the
real-time goal.

**Shape (two halves, no emitter changes):**
- Static (no execution): census maps + advisor pass-counts →
  predicted passes/bytes per op.
- Dynamic (one traced run): measured wall per op, dispatch-vs-compute
  split, bytes per stream → per-op bottleneck verdict
  (traffic/dispatch/compute-bound).
- `--budget` flag: milliseconds-in, per-op over/under out (e.g. 33ms
  frame budget → the real-time burn-down list).

**Bar:** gated on self-consistency (static prediction vs measured
within a band). CPU-first. Small, wraps existing trace + census.

**Trigger:** MET (training verdict landed as e974eaf, empty 0/360) — prompt next.

### 1a. Built with evidence (`chain/geo_profiler.py`, `tests/test_geo_profiler.py`)

Placement: `chain/geo_profiler.py` — alongside the `census.py`/`fusion.py`
it wraps, sharing their import surface and the `python3 tests/test_x.py`
gate pattern. No emitter changes, no new dependencies, CPU-only.
Static half predicts TRAFFIC (passes → bytes linearly), not wall: compute
varies by op class (stated v1 scope). Dynamic half is one traced run after
one untraced warmup (GC paused during the trace). Floor/roof come from the
run itself (no magic constants): dispatch iff wall ≤ 2×floor and slow
mover; traffic iff bw ≥ 0.30×roof (0.30 sits in the measured gap:
max-compute 0.17× roof, min-traffic 0.46× roof); else compute.
`bench/matmul_decode.cu` stays the served-path GPU reference — cited, not
executed (no served-path profile needed CPU-first anything here).
Demo sizes: flagship 24×32 as suggested; demo 24×32 REPLACES suggested
12×16 with reason (at 12×16 a scheduling outlier flipped one verdict
1/10 runs; at 24×32 verdicts are 10/10 stable, statics size-independent).

Measured example — flagship 24×32, `--budget 33` (gate: WMAPE 0.173 ≤ 0.40):

```text
op           class       P  predB     B  wall_ms  bw_MBps verdict fused
SRGB_DECODE  elementwise  2  10805  11520    0.048    240.3 traffic  -
LUMA         reduce       2  10805  15360    0.035    443.2 traffic  -
SQRT         elementwise  2  10805  10752    0.067    159.7 traffic  -
SPLAT_BLUR   stencil      3  16207  13824   11.737      1.2 compute  -
SUB          elementwise  3  16207  13824    0.189     73.1 compute  -
BETA_V5      elementwise  3  16207  13824    1.388     10.0 compute  Y
MUL          elementwise  3  16207  13824    0.033    423.6 traffic  Y
ADD          elementwise  3  16207  13824    0.161     85.7 compute  Y
SQUARE       elementwise  2  10805   9216    0.196     47.0 compute  Y
GAIN         broadcast    4  21610  29184    0.400     73.0 compute  Y
SRGB_ENCODE  elementwise  2  10805  11520    0.030    381.1 traffic  Y
totals: passes=29 saved=10 (1 chains) bytes=156672 wall=14.284ms
consistency: WMAPE=0.173 band=0.40 OK
budget 33ms burn-down head: SPLAT_BLUR +8.323ms over (the frame-buster);
next worst BETA_V5 already -2.026ms under (total over -18.716ms).
```

Demo 24×32: 43 ops, passes=109 saved=28 (4 chains), bytes=466800,
wall≈3ms, WMAPE=0.066 OK; verdicts 27 traffic (memcpy movers: MUL/SLICE/
CONCAT) + 16 compute (lattice ADD/SUB), zero dispatch; 33ms budget head
is a SLICE at -0.586ms (whole listing fits the frame 10× over).
Gate `tests/test_geo_profiler.py` (CPU, <1s): band ×2 listings, budget
over/under/sorted/sums, determinism (canonical + verdicts), synthetic
verdict edges, fail-loud legs, CLI smoke (`--budget` required).
Neighbors green: `test_census.py`, `test_fusion_advisor.py`.
Bugs caught while building (mechanisms): (1) demo 12×16 verdict flipped
1/10 runs — µs-scale wall outlier crossed the 0.30✕roof boundary
(bw/roof 0.29 vs 0.30); fixed by profiling demo at 24×32 (absolute jitter
shrinks relative to walls; 10/10 stable since). (2) burn-down ROW ORDER
is wall-derived — two near-tied overs swapped places between runs, so
`canonical()` compares budget rows (op,line)-sorted while in-run
worst-first order is gated separately. Gap named: dispatch-bound never
fires on these CPU listings (every op moves ≥140 elems; Python overhead
never dominates) — the branch is covered by synthetic unit legs only,
live dispatch verdicts await a served-path (GPU) profile.
