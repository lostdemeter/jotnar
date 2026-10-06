# Geo Qwen2-7B performance (measured 2026-10-05, smax=16, ntok=8)

Prompt (14 toks): "The capital of France is Paris. It is the most
populous city in". Box: RTX 3090 Ti 24GB, 61GiB RAM, user-local
toolchain (nvcc 13.4; see scripts/cuda_env.sh). Reproduce:
`source scripts/cuda_env.sh && python3 scripts/bench_perf.py --s 16 --ntok 8`.

## Walls

| side | prefill / build | decode/token |
|---|---|---|
| HF (bf16, cached) | 278ms prefill (S=14) | 24.3ms (min 21.8) |
| geo one-shot (fp16, full-recompute) | nvcc 133s clean / 475s profile (one-off) | wall 11.6s (min 7.2s), GPU-busy 3.3s |
| geo serve (fp16, weights-load-once) | same build, 6s resident load | 145ms/step, parity 4.84 (== one-shot) |
| geo serve --no-sync (stream ordering) | same | 100ms/step, bit-exact vs synced |
| geo serve --no-sync --graph (1 replay launch) | same | 88-91ms/step, parity 4.84 (== all) |
| geo serve --bmmv --graph (head-batched attn) | same | 76-83ms/step, parity 4.84 (== all) |
| geo serve --bmmv --graph + block-parallel reductions | same | 44-46ms/step, parity 4.84 (== all) |
| geo decode --serve (KV caches in files) | same build class, 0.5s prefill | ~90-100ms/tok, prefill parity 1.08 vs recompute |

## S-scaling (recompute vs decode, bmmv+graph+nosync both)

| S | recompute/step | decode/tok | decode prefill parity vs recomp |
|---|---|---|---|
| 16 | 76-83ms | ~90-100ms | 1.08, same argmax |
| 32 | ~111ms | ~104ms | 1.03, fork-physics argmax |
| 64 | ~132ms steady (first-run warmup climbs; transient, not scaling) | ~108ms | 0.92, same argmax |
| 128 | ~241ms | ~124ms (prefill 142ms/tok) | 0.71, same argmax |
| 512 (recompute pathological: 14s/step VRAM) | decode ~144ms | 0.8, same argmax (after pos fix; was 10.04) |

Crossover between 32 and 64, widening: recompute climbs
(~linear-plus: MLP activation traffic 2x per doubling + attention
compute), decode stays ~flat (constant 1-row compute; relay grows
KBs). Both fluent at 128 (fork after "tourist attractions.").
| geo decode --serve (KV caches in files) | same build class, 0.5s prefill | ~90-100ms/tok, prefill parity 1.08 vs recompute |

Decode is wall-neutral at S=16 by physics (same 15GB weight traffic;
attention FLOPs are negligible this short) -- its value is S-scaling
and true autoregressive serving, not S=16 pace. Prefill-vs-HF parity
5.15, first-pick rank 2 (tests/test_decode.py).

Gap to close: ~3.3x per token vs HF. Weight floor ~15-30ms; rest is
cublas-call + small-kernel time inside the replay.

## Per-op profile (cudaEvents, one step, 9944 ops)

| op | ms | share | count |
|---|---|---|---|
| MATMUL | 3095.0 | 93.7% | 197 |
| BATCH_MATMUL | 52.1 | 1.6% | 1568 |
| RMSNORM | 33.6 | 1.0% | 57 |
| SLICE | 25.2 | 0.8% | 2352 |
| CONCAT | 19.3 | 0.6% | 756 |
| ROTARY | 16.9 | 0.5% | 1568 |
| SOFTMAX_WIDE | 10.2 | 0.3% | 784 |
| TSHIFT | 8.0 | 0.2% | 784 |
| SILU / TRANSPOSE / ARGMAX / CONVERT | ~28 | 0.9% | — |

## Reading (three gaps, ordered)

1. **Reload gap: CLOSED by serve mode.** `live=[...]` loop binaries
   (chain/emit_cuda.compile_cuda + chain/serve.py ServedExe) read
   frozen inputs once (gated: bank poisoned mid-run, outputs
   unmoved) and serve steps at 145ms warm (100ms with --no-sync).
   tests/test_serve.py 10/10 ALL OK (match, frozen-proof,
   determinism, matmul no-accum, nosync bit-exact).
2. **Sync gap: CLOSED by stream ordering.** Per-op
   cudaDeviceSynchronize cost ~45ms/step; same-stream ordering is
   correct (stores sync before host reads) and bit-exact
   (serve-nosync-equiv). Default stays per-op sync (debug
   friendly); --no-sync for runs.
3. **Launch gap: CLOSED by graphs, mostly.** One replay launch
   per step (capture warmup + instantiate once): 100ms -> 88-91ms,
   parity 4.84 identical (QWEN_GRAPH=1 gate). Only ~10%: launches
   were never dominant; remaining is cublas-call + small-kernel
   time inside the replay over the ~15-30ms weight floor.
   tests/test_serve.py graph legs (match, track, deterministic).
   KV cache explicitly DEFERRED: at S<=32 both sides are
   weight-traffic-bound, so the cache saves FLOPs but ~no wall --
   it earns its keep at S-scaling (attention O(S^2)), after graphs.

## Correctness found en route (beta=1 accumulation)

The loop exposed a real emitter bug: all four cuBLAS call sites
passed beta=1, so outputs accumulated (C = A@B + C_old) instead of
overwriting. One-shot survived on fresh zero-pages; step 1 vs
step 0 drifted 17.9 with identical inputs. Fixed to beta=0
(kZero) at all four sites; regression gate serve-mm-no-accum
(0.00e+00) in tests/test_serve.py -- the bigram gate couldn't
catch it (no matmul in it).

## Consequences

- Graphs-before-persistence would optimize the 6% while the 94%+ is
  reload+recompute. Order: persistent process, then KV cache, then
  re-profile, then graphs iff launches dominate what remains.
- Cached-decode target: MLP 1 row x28 + trivial attention should
  land ~50-200ms/token (HF does 24ms with the same math).
- nvcc profile builds cost 3.6x clean (event code x10k ops):
  profile rarely, keep clean binaries for runs (QWEN_REUSE_BIN=1).
