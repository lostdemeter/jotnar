# Jotnar — where we stand (2026-10-22)

## What it is

Jotnar is a geometric assembly language: 63 mnemonics over three data
kinds (F float, T lattice-triples, I integer) plus a compiler
(`chain/`: assembler, builder, census, three emitters — C,
CUDA/cuBLAS, integer-only C) that lowers real programs to real
hardware. Two production workloads compile through it: Qwen2-7B-Instruct
decode at 27ms/token (vs 24ms HF reference) and an image pipeline
(unsharp + flagship). Programs also serve over real sockets across
heterogeneous nodes (protocol M1–M6, calculator/brain split proven to
8 saturated lanes), and batch-fuse through a gated contract
(`chain/batch.py`, byte-exact throughout).

## Performance standing (measured, gated)

- Decode: 27ms/token resident vs 24ms HF. The gap is 3ms, and every
  basis point of it is accounted for: ~15ms is weight traffic
  (14.3GB/token at ~750GB/s effective), ~12ms across 113 GEMM calls.
- Three hunt rounds, all merged green, all negative:
  1. MATMUL characterization — GEMM is 92–96% of every step; per-call
     kernels at ceiling; epilogue fusion the only unpriced idea.
  2. Dispatch fusion (diagonals) — fused vs serial ties twice
     (260/255ms flagship, 2.5/2.5s unsharp); CUDA graphs already
     amortize every launch on the served path.
  3. Representation survey (`docs/REPRESENTATIONS.md`) — 11 DEAD with
     kill evidence, 3 PARKED; S=128 killed the last open question
     (ATTN middle band 0.43%, KV growth already shipped as resident).
- Conclusion: the remaining wall shrinks exactly two ways — move fewer
  bytes per weight (**quantization**, blocked by standing order: no
  weight changes) or do more work per byte moved (**epilogue fusion**,
  an unmeasured backend change). Both are outside current scope, by
  decision, not by oversight.

## The language is close to done

The ISA has been stable at 63 mnemonics with the `asm-lang-coverage`
gate enforcing registry/table/docs agreement. Recent additions earned
their place by the same rule (registry row + shape rule + one pattern
per backend + gate): TOPK, SCATTER, ATTN, ROPE, SWIGLU, IADD/ISUB/
IMUL/IMAD, GATHER3, IDIV/IMOD, F2I/I2F. Failure is
loud everywhere (`NoPattern`, arity/shape/kind checks); determinism is
load-bearing and holding (workspace-pinned cuBLAS, 0.0000 legs).

What "done" means here: the language expresses everything we ask of
it (7B transformer decode, image pipelines, remote serving, batch
fusion), refuses what it can't do, and proves both directions in
gates. Open language work is inventoried, not lurking: per-opcode
batch surface (halo exchange), listing-level BATCH syntax, guard-as-op
for served fusion, LAYERNORM/CONV backend patterns, MATMUL-epilogue
support. Each is parked behind a workload that pays for it — none has
one today.

## Parked, explicitly

- Quantization (standing order: no weight changes).
- Backend extensions (T-kind CUDA, cublasLt epilogues, new kernels).
- Cross-program fusion / shader-merge (Step 2, parked behind demand).
- DAV2-Small lowering to assembly (Qwen-scale arc, sequenced after need).
- Yarnball / ball tier (lives on `research`, never merges).

## Process note

Every claim above is gated in-tree and every negative result merged
with its evidence. The tripwires (`fx-graph-amortizes`,
`seam-blur-halo`, `rep-*` bands, the PARKED falsifiers) re-prove the
closures every run — if the ground moves, a gate goes red and says so.
