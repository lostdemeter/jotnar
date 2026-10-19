# PATTERN_GAPS — priced, unbuilt (student gap-examination round)

Scoping-only: design + parity-gate shape + size estimate + dependency
order per item. Zero implementation, zero new patterns, zero training.
A price card without a falsifier is incomplete — every item names one.
Smaller-than-priced findings are the report (see Sequences).

Extension recipe (all Track-A builds plug into it): registry row +
shape rule + one pattern per backend (`chain/backends.py`;
`README.md:212-215`). Scaffolding pointers verified pre-pricing;
nothing below was built in this round (tripwire-gated).

Conventions: size S = days (pattern + gates, scaffolding exists),
M = 1–3 weeks (design or audit + patterns + workload), L = month+
(protocol design). Falsifier = the bar the implementation must
clear or the pricing was wrong.

## Track A — DiT-minimum patterns

### A1 — LAYERNORM (C/CUDA) — BUILT (pattern-a1 round, evidence below)

- Unlocks: teacher-faithful DiT blocks (norm1/norm2 as built, not
  RMSNorm-substituted); gpt2-class listings return.
- Existed: drilled-only — `tests/test_asm.py:80-106` (0-diff vs
  legacy, 40dB vs torch, eps tripwire), thin wrapper
  `chain/asm_ops.py:326-343` mirroring eps_rms doctrine; parked
  (`README.md:181`, shares RMSNORM scaffolding).
- Built: shared `infer_shapes` branch (same-kind x(...,D)+w(D,)+b(D,));
  C F64 row loop (mean+var+affine, CONFIG eps_ln, RMSNORM doctrine);
  CUDA `k_layernorm_f32` (block-per-row cooperative mean+var, fixed
  mapping → bit-exact rerun); F32 only (fp16 call sites fail at
  nvcc loudly, never silently). Non-FPU untouched (still NoPattern).
- Measured (`tests/test_layernorm.py`, ALL OK): lattice 65.8dB,
  C 315.0dB, CUDA 144.3dB vs torch (40dB bar); CUDA bit-exact
  rerun; missing-eps/width/kind all fail loud.
- Coverage line corrected with evidence: C was stale at 20
  (any-F convention from `4581651`: true value was 23 pre-build,
  now 24); non-FPU 20→21 (pre-existing drift, no non-FPU change
  made here); CUDA 25→26 exact.
- Parity-gate shape: as priced (per-backend parity + rerun legs +
  text-form listing leg) — landed in `tests/test_layernorm.py`.
- Size: was S — scaffolding carried as priced.
  Dependencies: none (confirmed).
- Falsifier: did NOT fire (all backends clear 40dB by 25dB+).

### A2 — GELU (C/CUDA)

- Unlocks: DiT MLPs as built (no SILU substitution).
- Exists: drilled + single-op listing (`tests/test_asm.py:741-773`,
  impl `chain/asm_ops.py:346-353`); parked, no workload
  (`README.md:178`); CUDA `k_elemU` notes room for gelu ids
  (`chain/emit_cuda.py:194-201`).
- Open (priced in, not resolved): which approximation the teacher
  needs — erf vs tanh pin during build (audit the DiT config/code
  first; wrong variant = silent numerics drift).
- Parity-gate shape: per-backend parity vs torch variant-pinned
  (40dB) + edges leg (saturate/dip/zero/linear, cf. existing
  drill) + MLP listing leg.
- Size: S. Dependencies: none.
- Falsifier: variant audit showing the teacher needs neither erf
  nor tanh within 40dB (then SILU-substitution stands and A2 dies).
- MEASURED — FIRED (`tests/test_gelu.py`, ALL OK): teacher DiT MLPs
  use diffusers ApproximateGELU = x·sigmoid(1.702·x) (audited in
  `transformer_qwenimage.py:628,638` → `activations.py:149-169`).
  erf-vs-sig1702 38.6dB, tanh-vs-sig1702 38.4dB in [-4,4] (both
  below 40dB where pre-activations live; 41.5dB on [-8,8] with
  0.02 maxabs does not save them). A2 KILLED — no erf/tanh
  patterns built; row stays parked (still true).
- Smaller-than-priced note: the teacher form composes from tabled
  ops at 314.6dB — SILU(MUL(x,1.702))/1.702 via BETA+MUL+SILU
  (all F). DiT MLP fidelity needs NO new pattern.

### A3 — CONV past audit-first (C/CUDA)

- Unlocks: VAE-grade decoders, local vision lowering, densewin
  expressed as CONV; the VAE branch of the student program.
- Exists: lattice drill (`tests/test_asm.py:541-547`, wiring only)
  + C bit-exact vs numpy (`tests/test_c_conv.py:70-114`, impl
  `c_chain/holo_conv.c`); emission `NoPattern` on all backends
  (`chain/emit_c.py:472-473`, no case in `emit_c.py:529-966` or
  `emit_cuda.py:475-943`).
- Audit-first (the price driver): `c_chain/holo_conv.h:1-13` —
  edge replicate (NOT zero-pad, differs from phi-core phi_conv),
  order-free int64 accumulate. Landing emission needs the
  edge-reconciliation audit vs phi_conv FIRST, then one pattern
  per backend + workload. If the audit diverges (new semantics),
  price grows — audit-first means the card holds either way.
- Parity-gate shape: audit record (edge-case matrix both
  conventions) + per-backend parity + gradient/step/impulse/
  flat/corner/noise workload (cf. existing C suite) + one vision
  listing leg.
- Size: M. Dependencies: phi-core access for the audit; nothing
  in-tree.
- Falsifier: audit concluding zero-pad equivalence (then the audit
  was oversized and A3 reprices to S).

### A4 — fused encoder (joint-bidirectional) attention kernel

- Unlocks: student-scale joint attention as ONE kernel (not
  decode-GQA shape); memory/perf for the nominated architecture —
  capability-adjacent, NOT new math (BMMV + SOFTMAX_WIDE already
  cover it; unfused `BATCH_MATMUL+SOFTMAX` composition exists in
  `stdlib/attention.asm:7-18`).
- Exists: fused single-output precedents (ATTN decode-GQA
  `chain/asm_ops.py:635-669` + `chain/emit_cuda.py:320-376`
  `k_attn_gqa_f32`; ROPE; SWIGLU) with bit-exact rerun gates
  (`tests/test_attn.py:153-159`, `tests/test_rope.py:107-138`).
- Parity-gate shape: lattice-vs-torch joint attention 1e-9 (cf.
  `tests/test_attn.py:63-104`) + mask-exactness + fail-loud
  geometry + CUDA parity + `attn-cu-deterministic` twin leg.
- Size: M (one kernel × backends + gates; math proven, kernel new).
  Dependencies: emitter table; unfused composition (exists).
- Falsifier: unfused composition matching fused within 10% wall on
  student-scale seqs (then A4 is perf theater — defer, don't build).
- MEASURED — FIRED (`tests/test_fused_defer.py`, ALL OK; walls in
  `samples/marigold/fused_measure.json`): unfused composite vs
  flash-SDPA at student scale — 676 tokens 0.039ms vs 0.123ms
  (0.31x), 2692 tokens 0.275ms vs 0.552ms (0.50x), RTX 3090Ti solo.
  Unfused is FASTER (small-seq launch overhead dominates flash) —
  A4 DEFERS with evidence. Composition agreement leg green;
  no fused encoder kernel exists (tripwire).

### A5 — multi-output kernels (fusion interior taps)

- Unlocks: fusion of chains with interior consumers (MLP+norm
  fusion, QKV-tap fusion) — memory traffic, not new math.
- Exists: advisor explicitly parks it (`docs/FUSION.md:11-16,28-30`,
  `chain/fusion.py:26-30,136-144` interior-escape skip); only
  single multi-out ops (TOPK 2→2, SPLAT_BLUR 1→2); builder-level
  QKV fusion stamps `fused_qkv` (`docs/VELOCITY.md:4394-4408`).
- Parity-gate shape: advisor interior-tap leg (chain with tapped
  middle proposed + materialized) + `fuse_exec` byte-identical leg
  (cf. `test_fusion_codegen.py:89-91`) + `fx-cuda-block` tripwire
  twin + one fused-interior workload leg.
- Size: M (advisor surgery + exec + 1+ kernels + stamp).
  Dependencies: fusion advisor + fuse_exec (exist); A4-class
  kernels as consumers (order below).
- Falsifier: advisor finding zero interior-tap chains in DiT-block
  listings (then nothing to fuse — A5 dies for lack of demand).

### A6 — halo exchange messages

- Unlocks: tiled vision past single-device memory (1024px+
  SPLAT_BLUR-class workloads sharded across devices).
- Exists: nothing but the refusal — `chain/batch.py:210-214`
  (`exchange` refused loud, "needs a seam protocol"), drop/refuse
  policies + whole-`H` pipeline ferry (`docs/PROTOCOL.md:36-186`,
  M-lanes bit-identical precedent `docs/PROTOCOL.md:301-313`).
- Parity-gate shape: seam-protocol spec + exchange-vs-drop
  equivalence leg (tiled == untiled bit-exact, cf.
  `tests/test_diagonal2.py:75-127` seam legs) + M-lane-style
  75/75 bit-identical multi-process leg + contamination-confinement
  leg.
- Size: L (protocol design is the bulk; code is small).
  Dependencies: PROTOCOL M-lane precedent; A5-class tiled kernels
  as the demand side.
- Falsifier: single-device memory fitting all student scales
  through the nominated architectures (then A6 is premature —
  defer to first OOM with a seam design ready).
- Determinism: UNKNOWN pending design — exchange ORDER across
  devices must be fixed in the protocol (allocation-state
  summation variance is documented: `chain/emit_cuda.py:1198-1203`);
  the card requires the protocol to fix order or A6 stays unopened.

### Track-A sequence (recommended)

A1 → A2 → A4 → A5 → A3 → A6.
Why this order: smallest-first (A1/A2 are days, scaffolding
shared/drilled); capability-before-perf (A4's kernel matters more
than A5's traffic); student-verdict-early (A1+A2+A4 = first
faithful-DiT student attempt — the Phase-7 blueprint as built,
finally testable); A3 independent but big and VAE-branch (runs
anytime after A2, placed fifth so the DiT verdict lands first);
A6 last (scale-out only; single-device students never need it —
its own falsifier may defer it indefinitely).
Smaller-than-priced findings (the report): A1 is thinnest
(wrapper + shared scaffolding — proposed FIRST build); A4 math is
already covered (kernel is perf+shape, deferrable per its
falsifier); A3 may reprice to S if the audit shows equivalence.

### Track-A student verdict (what it unlocks)

A1+A2+A4 build the faithful-DiT student: teacher blocks as built
(LayerNorm, GELU, joint attention) at student scale. Verdict it
yields: does teacher-faithful beat the mixer wall (0/156
held-out)? A YES reopens architecture empirically; a NO (with
A5/A3 behind it) closes the faithful branch and leaves only Track
B or the textured-domain restriction. What stays unknown after
pricing: GELU variant pin, fused-attention wall numbers (measure
post-build), CONV audit outcome.

## Track B — adaptive control flow

Substrate facts (priced from): no JUMP/WHILE/runtime dispatch
exists (`docs/LANGUAGE.md:251-252,335-336`,
`chain/asm.py:759-772` — WHILE/data-dependent termination +
dynamic shapes refused); CALL is textual macro only
(`chain/asm.py:62-292`, seam `chain/emit_c.py:1023-1064`); SELECT
is verdict-gated data-select on fixed geometry
(`tests/test_asm.py:255-283`, runtime count guard
`chain/emit_c.py:881-899`); GT is lattice+nonFPU only (C/CUDA
NoPattern, filed). Determinism baseline: everything bit-exact
today because nothing branches at runtime.

### B1 — CALL lowering (runtime dispatch)

- Unlocks: reusable block subroutines in listings (xf_block CALL
  is textual today); subroutine calls in emitted C/CUDA.
- Design space (priced, not chosen): static call-graph lowering
  (expand-at-build, fixed order — deterministic by construction,
  cf. macro expansion `tests/test_asm.py:569-570` CALL==inline)
  vs dynamic dispatch table (smaller binaries, needs determinism
  story — UNKNOWN, marked).
- Parity-gate shape: CALL==inline exact leg (exists at lattice:
  extend per backend) + recursion/nesting refusal twins +
  `sanitize_cnames` collision leg (exists: extend).
- Size: S (static) / M (dynamic + determinism proof).
  Dependencies: assembler expansion + builder call/def (exist).
- Falsifier: listings never reusing a block twice (then CALL is
  sugar — textual macros suffice, B1 dies).
- Determinism: static = deterministic by construction (fixed
  expansion order); dynamic = UNKNOWN (dispatch order +
  allocation state must be fixed in design or B1-dynamic stays
  unopened).

### B2 — data-dependent routing (dynamic SELECT/GT)

- Unlocks: per-input paths (mixture routing, top-k data masks,
  conditional depth) — the Phase-16 demand in pieces.
- Exists: SELECT on fixed geometry + GT lattice-only
  (`tests/test_emit_nonfpu.py:339-392` bit-identical GT+SELECT);
  TOPK k is build-time const (`docs/LANGUAGE.md:193`).
- Needs: GT C/CUDA patterns (filed gap) + dynamic-shape
  semantics (explicitly unscoped today) + routing-shape proofs
  (state-shape discipline exists: `tests/test_asm.py:349-355`).
- Parity-gate shape: GT per-backend parity (exact integer
  compares) + dynamic-SELECT shape legs + refusal twins
  (float mask, unknown arity — cf. existing) + one routed
  workload leg.
- Size: M. Dependencies: B1-static (call sites for routed arms);
  GT gap filed.
- Falsifier: no routed workload beating static SELECT on fixed
  geometry (then data-dependence buys nothing — B2 dies).
- Determinism: SELECT/GT ops exact (`np.where`/integer compares —
  deterministic); routing that changes EXECUTION ORDER per input
  (skip arms, reorder) = UNKNOWN (order variance is the documented
  nondeterminism source: `chain/emit_cuda.py:1198-1203`); the card
  requires order-fixed routing or B2 stays unopened.

### B3 — per-input adaptivity machinery (bounded iteration + state)

- Unlocks: the Phase-16 measured demand (normalize-then-read,
  early-exit, adaptive depth) — adaptivity with a price tag.
- Exists: `repeat()` fixed geometry (`tests/test_asm.py:335`),
  estimator hull (`tests/test_asm.py:388-415`, `chain/ranges.py`);
  WHILE refused.
- Needs: B1 + B2 + bounded-iteration semantics (max-iters const,
  fixed state shapes — the `repeat()` precedent generalized) +
  state-shape proofs per arm.
- Parity-gate shape: bounded-iteration determinism leg (rerun
  bit-exact at max iters and early-stop — both) + state-shape
  legs + one adaptive workload leg + data-dependent-termination
  refusal twin (unbounded stays refused).
- Size: M (needs B1+B2 landed first; alone S).
  Dependencies: B1, B2.
- Falsifier: adaptive workload matching fixed-budget quality at
  equal-or-higher wall (then adaptivity is overhead — B3 dies).
- Determinism: bounded iteration at fixed geometry =
  deterministic (repeat precedent); data-dependent TERMINATION
  (stop when converged) = UNKNOWN and stays refused — B3 opens
  max-iters-bounded adaptivity only.

### Track-B sequence (recommended)

B1-static → B2 → B3 (strict dependency order; B1-dynamic deferred
to its determinism proof, which may never come). Why: each card's
needs list the previous; no parallelism available. B1-static is
days and unblocks listings hygiene regardless of adaptivity.

### Track-B student verdict (what it unlocks)

B1+B2+B3 build the adaptive student: per-input routing +
bounded-iteration refinement — the Phase-16 demand with a price
tag. Verdict it yields: does adaptivity close near-field
(Phase-9 brief strata) where static budgets flatlined? A YES
reopens the student empirically with control flow priced in; a NO
closes adaptivity and leaves the textured-domain restriction as
the last standing door. What stays unknown after pricing:
dynamic-dispatch determinism, order-fixed routing designs,
unbounded termination (refused, not unknown — never open it).

## DiT verdict (pattern-a2a4 round product — no student built)

With A1 built, A2 killed, A4 deferred: the teacher-faithful path
is OPEN with zero further patterns. LayerNorm landed (C/CUDA,
deterministic); the teacher's GELU form composes from tabled ops
(no pattern needed); joint attention runs as unfused composition
(faster than fused at student scale — no kernel needed). Every DiT
block op is now landed-or-composed. What still blocks: A3 CONV
(VAE path only — DiT blocks need no conv), A5 taps (perf),
the VAE itself (decoder = A3 territory). Verdict, not a build:
a faithful-DiT student attempt is unblocked at the pattern level —
whether it passes bars is an experiment for a declared round, and
the mixer/attention wall (0/156 + 0/36s) prices its odds low.
Retired unknowns: GELU pin (sig-1.702, composable), fused-attention
walls (0.31x/0.50x unfused-wins — A4 falsifier tested).

## What stays unknown after pricing (both tracks)

GELU erf-vs-tanh pin; fused-attention wall numbers; CONV audit
outcome (equivalence or new semantics); halo exchange order
(UNKNOWN); dynamic-dispatch determinism (UNKNOWN); unbounded
termination (refused); routed-workload demand proof (B2's
falsifier untested); interior-tap demand proof (A5's falsifier
untested); unfused-vs-fused wall gap at student scale (A4's
falsifier untested). Unknowns are design debt, not build debt —
each names the experiment that retires it.
(RETIRED this round: GELU pin → sig-1.702 composable, no pattern;
A4 falsifier → tested, unfused wins 0.31x/0.50x, deferred.)
