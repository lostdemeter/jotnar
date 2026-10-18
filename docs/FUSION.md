# FUSION.md — advisory fusion reports (`chain/fusion.py`)

Given a listing text, report fusible elementwise chains. Advisory
only: no rewrites, no emitter changes, no new opcodes, no stamp
changes. Anything ambiguous is reported unanalyzable with reason,
never guessed. Deterministic (program order throughout, no
randomness); gated in `tests/test_fusion_advisor.py`.

## Legality rules (per link; else boundary or refusal)

1. Single consumer: producer stream fan-out == 1 (exactly this
   link). Fan-out ≥ 2 blocks (surfaced in `blocked`). Fan-out == 1
   also means intermediates can't escape as outputs (a consumed
   interior member is never a sink); an explicit `outputs` set
   additionally blocks chains whose interior it taps (served
   binaries tap intermediates — fusing through a tap needs
   multi-output kernels, parked).
2. Pointwise class: both ops in FUSIBLE (census elementwise +
   GAIN, a pointwise map with broadcast reads — the fused kernel
   performs the same reads the program already does). Reductions
   (MATMUL epilogue is second, not this), stencils, gathers, moves,
   selects, recur: boundaries. Unknown-class ops: unanalyzable +
   boundary, never guessed.
3. Shapes/layouts agree: carried by the program's own runtime
   contract (shape mismatch fails loud at run time, independently
   gated), which pointwise maps preserve structurally; plus an
   explicit check where static layouts are KNOWN (both binary
   operands declared with layouts that differ → unanalyzable).
   STATE-touching candidates: unanalyzable (order-sensitive).

Chains are paths (greedy longest-first, program-order tiebreak,
non-overlapping on ops, minimum length 2 — singletons fuse
nothing). Fan-in side inputs stay live (path-model limitation: the
luma head reports its longest path, not the full multi-input
region; DAG regions are parked codegen).

## Profitability model

Each eliminated internal edge saves one write + one read = 2 memory
passes (stream sizes unknown statically: passes, not bytes).
Per chain: 2×(ops−1). Reported per chain and summed; never bytes,
never wall time.

## Measured reports

- flagship (11 ops): 1 chain [BETA_V5,MUL,ADD,SQUARE,GAIN,
  SRGB_ENCODE], 6 ops, 10 passes, sink boundary. Blocked LIN:2
  Y:2 D:2 A:3. Singletons ([SRGB_DECODE],[SQRT],[SUB]) correctly
  skipped. No MATMUL inside; nothing unanalyzable.
- demo_img 12x16 (43 ops): 4 chains ([MUL,ADD,ADD] t0-line,
  [SUB,MUL,ADD] t4-line, [ADDx8,MUL] s-line, [SUB,MUL,ADD]
  detail-line), 18 ops, 28 passes. Blocked gray:13 blur:2 crop:3.
  t1/t3 overlap-skipped by the greedy tiebreak (documented).
  Nothing unanalyzable.
- decode program, serve_decode.build_decode(16) (619 ops): ZERO
  chains — correct, not a bug: the fused-opcode ISA ate everything
  (SWIGLU/ATTN/ROPE absorb; MATMUL/RMSNORM/SCATTER/GATHER/ARGMAX
  bound), independently cross-checked (0 fusible→fusible fanout-1
  edges exist). Nothing unanalyzable, no STATE.
- escape delta: flagship with outputs={OUT,AE} reports zero chains
  vs one normally — the no-escape rule firing, proven by delta.

## Codegen — one kernel per chain (`chain/fuse_exec.py`, lattice)

`run_fused(text, chain_lines, payload, ...)` mirrors `chain/asm.run()`
exactly -- same assemble, same op functions, same order, same checks
-- with one change: chain-interior ops write a side env (reads via
ChainMap, so even hypothetical feeds-reading ops behave), chain-final
ops (consumed outside, or sinks) materialize to feeds. Same functions
+ same order = byte-identical by construction (gated anyway, never
trusted). Generic over chains: zero per-chain code; the consumed
lines MUST equal the advisor's current report (the gate asserts
identity -- hand-wiring without the advisor link is refused by
construction here). Wall savings on lattice are feeds-dict traffic
only (numpy allocates per-op temporaries in both arms).

Measured (flagship 6-op chain, input_example.png):
unfused 260-272ms vs fused 255-268ms -- tied, as predicted. The 10
saved passes are memory-traffic the interpreter never paid (array
REFS, not copies); numpy temporaries dominate both arms equally.
Placement honesty applies: the lattice leg proves mechanics +
exactness, never speed.

### Stamp keys (specified, unbuilt)

No fused binaries exist, so no stamp is written (dead contract
refused). When fused emission lands, the key is `"fused":
[{ops, lines, out}]` alongside the existing serve keys; any binary
whose fused set differs from the key fails loud on the existing
stamp-mismatch path. The `fx-cuda-block` leg below is the tripwire
that forces this work the day coverage lands.

### CUDA: blocked, with evidence (stop finding)

The T-kind flagship chain cannot reach CUDA, twice over: (1) T-kind
inputs are refused by doctrine (`cuda: T input -- decode at
data-prep`); (2) BETA_V5/GAIN/SRGB_DECODE have zero emitted patterns
on every backend (README opcode table: —/—/—), and SQUARE is T-only
on non-FPU. The measured first refusal is `NoPattern: C backend: no
pattern for SRGB_DECODE (line ...:12 ...)`. Landing the CUDA leg
needs T-kind CUDA support everywhere (loads/stores/malloc per
stream kind -- the whole backend assumes F) plus 3+ new kernels with
their own parity gates: a backend-extension project, not chain
fusion. `fx-cuda-block` pins the refusal; it fails the day anything
compiles, demanding the real parity + determinism-0.0 gate.

### What the next chain needs (demo_img's 28 passes)

Four more chains, same machinery (run_fused takes any line set;
multi-chain union works naturally). Nothing new to build except
gate legs per chain. MATMUL-epilogue stays second (backend change,
priced separately) -- second chain overall is fine, epilogue is not
this task.

### Dispatch hunt — profile first (no winner: table IS the finding)

Method: wall = dispatch + k·elements fit over image sizes (intercept =
dispatch), per-op trace attribution, N-lane tiny-chain sweep, advisor
counts. No real listing is dispatch-dominated — Step 2 fusion did not
run (fusing for sport refused); the CUDA question below is answered
from code, not speculation.

```
listing            ops   wall        dispatch   share   where it goes
flagship 288x352   11   256.6ms     13.2ms      5.1%   SPLAT_BLUR 74%,
                                                       BETA_V5 14%
demo_img 352x288   43   105.1ms      3.0ms      2.8%   ADDs 45%,
                                                       SUB/SLICE ~10% ea
decode layer (1)   ~22   n/a lattice  n/a       n/a    lattice execution
                                                       abandoned: RMSNORM
                                                       needs triples;
                                                       matmul_int at 3584
                                                       wide is CPU-prohib-
                                                       itive (see below)
decode program    619    27ms served  ~0 launch  ~0%   113 GEMMs + small
                                                       kernels; 0 advisor
                                                       chains (nothing to
                                                       fuse regardless)
tiny chains N-lane  3   0.44ms/lane  ~all       ~100%  dispatch by
                                                       construction
```

Lattice decode-layer execution was abandoned honestly: the listing
needs triple-kind norm inputs and integer matmuls at 3584-wide are
CPU-prohibitive; the served path (CUDA) is where decode lives, and
its numbers stand (27ms, graphs on). Decode row above uses advisor
counts (619 ops, 0 chains) + the standing served wall.

### CUDA dispatch accounting (read, not guessed)

`chain/emit_cuda.py` graph capture (warmup + capture + instantiate,
:1387-1392) reduces the per-step served loop to a single
`cudaGraphLaunch` (:1459); kernels are stream-parameterized for
capture (:1073, `, 0, capStream`); the multi-launch `step_loop`
exists only in non-graph mode. Tripwire-gated
(`fx-graph-amortizes`, pure-Python compile, no GPU): 2-op listing
emits 2 launch sites without graphs vs 0 launches + 1 replay launch
with graphs. So per-op LAUNCH cost on the served path is already
zero; what remains per op is kernel EXECUTION (global-memory passes
-- reducible only by merged kernels = emitter work, out of scope)
plus cublas API cost (also captured once at build, not per step).
fuse()-style materialization-skipping has no CUDA analog in the
current emitter (every op is a separate kernel writing global
memory). The win was already eaten, structurally, before fusion
arrived.

### What earns the next round

A listing that is BOTH dispatch-dominated AND advisor-positive AND
CUDA-runnable. None of the three measured is all three (flagship:
compute-bound + T-only; demo_img: compute-bound; decode: zero
chains + already-graphed). Candidate class: small-kernel F-listings
with long elementwise chains on the served path -- none in-tree
today. If one appears: fuse at the kernel level (merged passes),
not the dispatch level (already zero).

### Eyes open remainder: emitted-kernel codegen (still parked)

- Fused-elementwise emitter pattern: one kernel per reported chain
  (loop over the stream, inline the op sequence); the report's
  (ops, lines, boundary) is the codegen worklist. Boundaries tell
  where values materialize (sinks) vs where the next kernel takes
  over (feeds-nonfusible).
- Determinism/stamp tripwires: fused kernels must preserve op order
  and rounding (integer paths bit-exact, float eps — same leveling
  as backends); any fused binary needs its own build stamp or the
  stamp check can't distinguish it (cf. protocol stamp provenance).
- Why MATMUL-epilogue is second: producer-fusion into reductions
  (bias-add, scale) is the highest-value shape but changes the
  GEMM call itself (cublasLt epilogue vs separate kernel) — a
  backend change, not a listing change. Elementwise chains first
  (pure win, no backend risk); epilogue after, priced separately.
  Codegen temptation = stop-and-report (that finding is the report).
