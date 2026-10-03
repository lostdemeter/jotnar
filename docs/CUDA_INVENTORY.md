# CUDA Inventory vs the 47-op ISA (v0.1, 2026-10-03, branch reorg/compiler-api)

Pipe proven: nvcc builds, RTX 3090 Ti (cc 8.6, 24GB) runs
(`/tmp/cuda_smoke/vec.cu`: 1M-float vadd, managed memory, correct).

## What exists (phi-core c_core, all triple fixed-point)

| kernel | approx ISA match | verdict |
|---|---|---|
| k_leaky | PRELU (sign-select + slope) | ADAPT-AUDIT (same family, not identical) |
| k_addfix | ADD at common scale | ADAPT-AUDIT (mirrors _bin2 add) |
| k_tmulsc | scalar tmul (GAIN-ish) | ADAPT-AUDIT |
| k_copych | CONCAT, HWC channel-slice only | ADAPT-AUDIT (layout-narrow) |
| k_nearest2 | INTERP nearest-x2 special case | ADAPT-AUDIT |
| phi_conv_cuda | CONV | ADAPT-AUDIT |
| phi_conv_fcat | ? (fused-cat, unaudited) | AUDIT first |
| CPU: arith/bridge/geo/fconv | tmul/binops/pool/sigmoid/nearest | CPU-only; informs ports, not reusable on GPU |

**Bottom line: zero kernels for AI programs as-is.** No matmul, no
softmax, no norm, no attention movement anywhere in phi-core (the
numpy_ops.py versions are CPU triple code).

## Doctrine conflict + ownership split (decided by principle)

c_core's `trap` target fails the build on float/double. Fastest
execution on a 3090 Ti means FP32 (+TF32/FP16 later), cuBLAS matmul,
tensor cores -- irreconcilable with integer-only. So:

- **CUDA-F lives in JOTNAR** (this repo, `chain/emit_cuda.py` when
  real): FP32 kernels + cuBLAS, eps-agreement class (L1 where
  conditioned, L2 decisions -- same leveled story as the C backend).
- **phi-core stays integer-pure** (trap intact, nothing asked of it).
- CUDA-T (bit-exact triple ports) is possible later but is NOT the
  fastest path; parked, not planned.

## 47-mnemonic build table (CUDA-F)

Lib = cuBLAS (no hand kernel). Template = ONE parameterized
pointwise kernel, per-op instantiation (~12 ops, one source).
Hand = small dedicated kernel. Meta = compile-time only (strides/
views, zero emitted code). Audit = verify existing kernel, adopt or
rewrite.

| mnemonic | tier | plan | notes |
|---|---|---|---|
| MATMUL | 2 | Lib cuBLAS Sgemm | ikj hand-loop dies here |
| BATCH_MATMUL | 2 | Lib cuBLAS SgemmStridedBatched | |
| RMSNORM | 2 | Hand (row reduction + scale) | LAYERNORM same family |
| LAYERNORM | 2 | Hand (shares RMSNORM scaffolding) | |
| SOFTMAX_WIDE | 2 | Hand (online softmax, per-row) | TSHIFT fuses as prologue option |
| SOFTMAX | 2 | Hand (same kernel, legacy path) | saturating behavior kept by gating, not by code fork |
| TSHIFT | 2 | Fuse-into-softmax OR trivial Hand | row-max only |
| ARGMAX | 2 | Hand (argmax reduction, ties-first) | |
| ROTARY | 2 | Hand (pairwise sincos, theta consts) | |
| GATHER | 2 | Hand (row gather; coalesced when sorted) | |
| TRANSPOSE | 1 | Hand (shared-mem tile; small sizes: naive ok first) | |
| SLICE | 1 | Hand (strided copy) | |
| CONCAT | 1 | Hand (two copies; k_copych audit may cover HWC) | |
| SELECT | 1 | Hand (predicated copy) or Template+mask | |
| BETA/TBETA | 2 | Fill (cudaMemset-like or fused into consumer) | constants from CONFIG |
| ADD/SUB/MUL/DIV | 0 | Template | |
| SQUARE/SQRT | 0 | Template (sqrtf) | |
| SILU/GELU/SIGMOID | 0 | Template (expf/tanhf) | |
| PRELU | 0 | Template (k_leaky audit may donate logic) | |
| CLIP/RESCALE | 0 | Template | |
| STATIC | 1 | Hand-small (formant classify; rare) | |
| GAIN | 0 | Hand-small (needs yenh plane) | |
| POOLAVG | 2 | Hand (reduction, easy) | |
| RESHAPE2/3/PERMUTE3 | 1 | Meta (stride descriptors, no code) | |
| SCAN | 2 | Hand (recurrence/prefix; mamba path) | after AI set |
| DECONV | 2 | Hand or cudnn-path (decide at build) | after AI set |
| INTERP | 2 | Hand (k_nearest2 covers x2-only) | after AI set |
| CONV | 2 | Audit phi_conv_cuda, else Hand/cudnn | after AI set |
| WARP/GAUSS/ISO_BLUR | 3 | Hand-custom | vision backlog |
| SPLAT_BLUR/MIXDYAD/BETA_V5 | 3 | Hand-custom | vision backlog |
| SRGB_DECODE/ENCODE/LUMA | 3 | Template (powf/madd) | trivial when needed |

Count to a compiling AI set: ~10 hand kernels + 1 template + cuBLAS.
Everything else is backlog with a named slot.

## Memory + interface fit

- v1: cudaMallocManaged (proven in smoke), weights uploaded in
  prologue, results downloaded in epilogue -- matches the Backend
  interface (prologue/pattern/epilogue) with zero redesign.
- Finishing touches (not v1): explicit H2D/D2H + streams, FP16/TF32,
  fused attn kernel, cublasLt, graphs for fixed-shape programs.

## Gates (per kernel, in order)

1. vs C backend float-float: TIGHT eps (~1e-6 rel; same math, order
   documented where reductions differ).
2. vs lattice: leveled agreement (L1 value-eps where conditioned,
   L2 argmax-exact) -- inherits the v0.3 finding, including the
   saturation carve-out.
3. Perf: per-kernel ns/element + roofline note vs cuBLAS baseline
   where applicable (matmul MUST beat-or-tie naive; else keep naive).

## First cut proposed (next work)

Template + RMSNORM + SOFTMAX_WIDE + ARGMAX + cuBLAS-wiring against
the bigram/headt programs: five pieces that re-prove every existing
gate on a second backend. Then movement kernels, then rotary/gather,
then audit-adapts.
