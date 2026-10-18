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

## Eyes open: what codegen would need next (NOT built)

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
