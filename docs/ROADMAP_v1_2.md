# v1.2: the substrate release (one listing, three substrates)

v1.1 proved blocks mix across magnitude regimes (65dB) and lower to C
bit-exact (matmul). v1.2 answers the execution question: the model
listings run end-to-end on CUDA, on CPU, and on non-FPU CPU — same
listing, same values, parity-gated. This is the release that makes the
"bespoke AIs" vision executable: shaped programs that run anywhere the
substrate exists.

Ground truth (measured 2026-09-30, not assumed): phi-core `c_core` has
CUDA only for conv (`conv.cu`) + elementwise kernels (`ops.cu`: leaky,
addfix, tmulsc, copych, nearest2). NO matmul/softmax/norm CUDA anywhere.
Our `c_chain` has CPU conv + splat + matmul. So v1.2 writes kernels
family-by-family; there is no existing CUDA path to "just wire up".

## Spine (in order)

### 1. Matmul CUDA kernel (first, everything flows through it)

Integer-only `k_matmul` (per-product tmul + to_fixed@m_acc in registers,
int64 block accumulate, from_fixed out), FPU-free by construction.
m_acc as a kernel parameter (the v1.1 scale path reaches CUDA too).
Gate: 3-way file-exchange parity numpy/C/CUDA bit-exact on the
test_matmul_c.py case set (small/batched/broadcast/big-m/zeros).
 doctrine risk, stated upfront: reduction ORDER across threads must not
 change integer sums (in-bounds int64 addition commutes — gate it; if an
 order effect appears, the measured-rows doctrine applies, not silence).

### 2. Flagship path on CUDA (mostly wiring + parity)

conv.cu exists; the rest (sqrt/amplitude/gain/chroma) is elementwise over
existing kernel shapes. Deliverable: flagship listing output on CUDA vs
numpy parity ≥40dB (image basis, peak=1.0 — same bar as every parity gate).

### 3. Emission driver + non-FPU audit (the release gate)

One mechanism selecting the substrate per run (shape: CONFIG key, default
numpy — undecided, decide at build time), running BOTH model listings
(flagship + xf_block) on all three substrates. Non-FPU CPU is proven the
way it always was: generalized trap (no float/double in ANY emitted C/CUDA
source — the LIB-011 convention: say FPU-free) + parity on the CPU path.
Gate: substrate matrix green — 2 listings × 3 substrates, every cell with
a number (bit-exact) or a basis (dB), no silent CPU-fallback.

## Acceptance gates (all must hold)

1. 3-way matmul parity bit-exact (5 cases incl. big-m + broadcast).
2. Flagship CUDA parity ≥40dB vs numpy.
3. Emission driver runs both listings × three substrates, all cells gated.
4. Trap clean over all C AND CUDA sources.
5. VELOCITY.md continues.

## Out of scope (named, not forgotten)

Performance parity (correctness first, priced — threading/OpenMP + kernel
fusion headroom stay backlog), full 42-op CUDA coverage at once (matmul →
norms/activations → rest, family by family), multi-GPU, autotuning,
per-DEF `@scale` (v1.1 end-state, still the design), phi-core merges
(owner action).

## Declared 2026-09-30 (all five gates hold)

1. 3-way matmul parity bit-exact (5 cases incl. big-m + broadcast;
   test_matmul_cuda.py, #LIB-034).
2. Flagship CUDA parity bit-exact — stronger than the 40dB bar
   (synth/real/big-m YENH rows; test_flagship_cuda.py, #LIB-035..037).
3. Emission driver + matrix: flagship numpy/C/CUDA exact, xf numpy
   standing cell + loud C/CUDA refusals, bogus names refused, dispatch
   proven via run_log (test_emit.py, #LIB-038).
4. Trap clean over all C AND CUDA sources (`*.cu` covered since gate 1).
5. VELOCITY.md continues (v1.1/v1.2 build entries with honest clock
   status + process fix — no fabricated times).
23/23 suites green. v1.2 is done; v1.3 seeds: per-DEF `@scale`, xf C/CUDA
lowerings (matrix refusals flip to numbers), SIGX-32bit comment upstream.
