# Backend capability matrix (2026-10-06; verified against code)

Three emitting backends + the lattice reference. Every cell is either
a tested pattern or a loud refusal (`NoPattern`) -- nothing silently
missing. Execution modes (live/graph/sync_each) are CUDA-only; other
backends refuse them loudly (compile_program, not silently dropped).

## Value models

| | lattice | C | CUDA | non-FPU |
|---|---|---|---|---|
| F streams | triples @ m_acc/m_cov | float64 (libc+libm) | FP32 compute (FP16 storage opt-in) | refused (decode first) |
| T streams | native triples | decoded F64 at data-prep | NoPattern (decode at data-prep) | native integer ops |
| I streams | int64 | int64 | int64 | int64 |

## Op coverage (48 mnemonics)

- C 18, CUDA 19, non-FPU 19 (see README opcode table for the rows).
- Deltas that matter: BMMV is patterned on C + CUDA (+lattice),
  refused on non-FPU (float views); CONVERT exists only inside the
  CUDA backend (compiler-invented F16->F32 upcast, not a mnemonic);
  GATHER over F16 tables is CUDA-only (k_gather_f16).

## Execution modes

| mode | C | CUDA | non-FPU |
|---|---|---|---|
| one-shot binary | yes | yes | yes |
| live=[...] persistent loop | loud refusal | yes (chain/serve.py) | loud refusal |
| graph capture | loud refusal | yes (needs live) | loud refusal |
| sync_each=False | loud refusal | yes (stores still sync) | loud refusal |
| time_ops profile | yes (clock) | yes (cudaEvents; refuses graph) | n/a |

## Agreement classes (leveled, per path)

- Integer paths (I/T, GATHER/ARGMAX/SELECT-masks): BIT-EXACT.
- Float paths, same precision: EPSILON (C-f64 vs CUDA-f32 ~1e-4-class;
  width spot-check tests/test_cross_width.py).
- Integer-vs-float (lattice vs C/CUDA): EPSILON where conditioned.
- 7B scale: CUDA-vs-HF parity bound + fork-aware gates (the only
  scale run; C/non-FPU honesty rests on small exactness + the width
  spot-check -- transitivity stated, not wished).

## Scale story per backend

- CUDA: full 7B (serve/graph/decode, S<=512 measured).
- C: toy + width spot-check (depth-1 full width); whole-7B in
  float64 never run (slow by construction, not refused).
- non-FPU: toy integer programs; 7B shapes never attempted
  (float weights need quantizing first -- unbuilt, stated).
