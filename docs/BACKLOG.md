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

Method executed exactly per constitution (no deviations, no emitter
changes): fresh `/tmp/dptprof/{plain,timed}/` binaries from the same
1237-op listing revision (`programs/dpt_tiny.py` builder), kaiming
weights seed 0 (`test_dpttiny.gen_weights` — training unchanged),
deterministic zero-RGB 672×1024 input, 5 fresh-process frames each,
GPU solo (no compute procs at launch; 670MiB/24564MiB base).

Served total RE-MEASURED (supersedes — 1192ms confirmed, not inherited):
plain walls [1195, 1219, 1220, 1243, 1396] (frame 0 = CUDA-init cold),
median **1220ms** vs demo-round 1192ms (+2.3% — same ballpark, CONFIRMED).
Timed binary median 1223ms → event-sync perturbation +2.8ms (+0.2%,
well under the ~12ms estimate). Qwen never co-loaded (no Qwen
machinery on this box path at all).

Per-op table (timed stderr `op MN ms`, 1237/1237 lines parsed every
frame; per-position median-of-5, aggregated by mnemonic):

```text
mnemonic       n    ms  share%   provenance
MATMUL       235 190.2   27.7   time_ops events, median-of-5
CONCAT       273 149.2   21.7   time_ops events, median-of-5
ADD          152 132.4   19.3   time_ops events, median-of-5
GATHER       407 128.6   18.7   time_ops events, median-of-5
SOFTMAX_WIDE  24  44.3    6.4   time_ops events, median-of-5
SILU          14  11.8    1.7   time_ops events, median-of-5
MUL           12   9.5    1.4   time_ops events, median-of-5
SLICE         72   5.8    0.8   time_ops events, median-of-5
TBETA          6   5.2    0.8   time_ops events, median-of-5
BETA           6   4.7    0.7   time_ops events, median-of-5
LAYERNORM     12   3.3    0.5   time_ops events, median-of-5
TRANSPOSE     24   2.1    0.3   time_ops events, median-of-5
per-op sum  1237 687.2   56.2%  (of 1223ms timed wall)
process harness — 535.5   43.8%  by subtraction, split below
served total  —  1222.6 100.0%  re-measured median
```

Harness-bucket split (measured where possible): file I/O 16ms
(`cat` all 489 inputs), CUDA-init 2ms (`/tmp` probe binary, 3 runs
1.8–1.9ms); remainder ≈ 517ms = managed alloc + first-touch
migration + cuBLAS setup + stores, RECURRING per fresh process
(one-shot argv binary re-migrates GBs every frame — artifact of the
harness, not the model). Reconcile: 687.2 + 535.5 = 1222.7 ≈ 1223
(exact to rounding); gap share 43.8% ≤ 50% band (gated).

Verdict — what must shrink 1220/33 = **37×**, in order:
0. Process harness 535ms → resident serving (precedent: serve_decode
   resident mode amortizes argv I/O + re-migration to ~0 per frame).
   No math change; removes ~44% first.
1. MATMUL 190ms (235 GEMMs: QKV/out ×6 blocks, MLP 128↔512, 1×1
   projs) — 27.7% of kernel time.
2. CONCAT 149ms (273 concats — fusion interleave traffic) — 21.7%.
3. ADD 132ms (152 residual/bias adds) — 19.3%.
4. GATHER 129ms (407 gathers — patchify/conv taps, upsample
   broadcasts) — 18.7%.
5. SOFTMAX_WIDE 44ms (24 full 2688-wide rows) — 6.4%.
6. Tail 43ms (SILU/MUL/SLICE/BETA/TBETA/LAYERNORM/TRANSPOSE) — 6.3%.
After harness removal, compute 687ms must shrink 687/33 ≈ 21×.
Top-4 (MATMUL/CONCAT/ADD/GATHER) = 88% of kernel time — any plan
that does not start there is not reading this table.
Gate: `tests/test_geo_profiler.py` served legs (banked numbers —
no GPU in-gate): reconcile exact, gap ≤ 50%, provenance per row,
burn-down arithmetic (shares = 100, over = wall − 33, sorted).
Neighbors: `test_dpttiny.py` build + demo legs green (nothing moves
underneath — zero repo-code changes this round).

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
