# Backlog — queued work, in order

## 1. Geometric code profiler (NEXT UP — starts after the DPT-tiny training verdict)

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

**Trigger:** worker prompt goes out once the training round
(`student/train-dpttiny`) merges and the box is free.
