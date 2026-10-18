# REPRESENTATIONS.md — cost-by-op map with kill criteria

Survey, not a plan: every row measured, every verdict has a falsifier
or killing evidence, unevaluated stays PARKED. Quantization appears
as PARKED-with-reason only (standing order: no weight changes).
Gate: `tests/test_rep_survey.py` pins the numbers underneath these
opinions (fast rows re-measured with bands; decode rows SKIP without
their profile workdirs -- setup: `QWEN_TIME_OPS=1 QWEN_WORKDIR=
/tmp/profrep scripts/serve_decode.py --smax 16 --sync --no-graph ...`
and the same with 128 into /tmp/profrep128).

## Cost table (measured, this box)

```
decode GEMM (served, S=16):  96.2% in-situ (2764/2874ms, 1130 calls),
  ~540-810 GB/s effective steady-state (bench/matmul_decode.cu,
  first call ~40ms autotune amortized), 14.3GB weight traffic/token
decode GEMM (served, S=128): 92.3% (2828/3064ms); wall 51ms/tok
GQA middle band (fused ATTN): 0.16% S=16 (4.7ms), 0.43% S=128
  (13.1ms) -- S2 growth insufficient to matter
KV traffic (relay): GATHER 0.9%->2.6%, SCATTER 0.3%->1.6% (S=16->128)
ARGMAX greedy tail: 1.2-1.5% (152k-vocab pick every greedy step)
RMSNORM: 0.29% in-situ; lattice 284ns/el vs 46ns/el pointwise (~6x,
  two passes: norm + normalize)
CONVERT casts: 0.2% -- floor
flagship lattice 288x352: SPLAT_BLUR 74% (191/257ms), BETA_V5 14%
demo_img lattice 352x288: ADDs 44-71% (share varies run to run;
  gate bands >35%), SUB/SLICE ~10% each; dispatch intercept 3.0ms
LUT-bound lattice ops: SILU/SIGMOID/SOFTMAX_WIDE sub-us/el
  (already LUTs)
```

## Verdicts (candidate, verdict, falsifier-or-kill)

- quantize-weights: PARKED (standing order) -- falsifier: parity
  <=4.85 + traffic drop at same wall.
- gemm-epilogue (cublasLt): PARKED (backend change, unmeasured) --
  falsifier: epilogue GEMM wall < separate at parity.
- sparse-S16 / sparse-S128: DEAD -- kill: ATTN 0.16% / 0.43%
  (band-map skips attention readout; GEMM bytes unchanged either way).
- rmsnorm-1pass: DEAD -- kill: 0.29% in-situ (2x win = ~0.5ms real).
- softmax-dense: DEAD -- kill: no standalone dense softmax exists
  (inside fused ATTN).
- scatter-resident / gather-upcast: DEAD (shipped patterns) --
  kill: resident banked 27-vs-30ms already; remainders 0.3-2.6% floor.
- argmax-tail: DEAD -- kill: required every greedy step; the file
  read it replaced (greedy-logits-skip) already shipped.
- splat-separable: PARKED (unevaluated) -- falsifier: separable
  >=40dB vs splat + wall win. No production path demands it.
- lut-finer: DEAD -- kill: 0.17-0.25ms/4k rows, already LUTs.
- dispatch-fusion-speed: DEAD -- kill: tied walls twice (260/255ms
  flagship, 2.5/2.5s unsharp).
- convert-elim / slice-zero-copy: DEAD -- kill: 0.2% / views already
  zero-copy where possible.

## Top-3 next arcs

None are LIVE -- that sentence is the honest top finding. The map's
only PARKED-with-a-pulse rows are quantize-weights (blocked by
standing order, not physics), gemm-epilogue (backend change,
unmeasured), and splat-separable (no production demand). If orders
change, each names its falsifier above (bar, gate shape, and size
follow the falsifier: parity bar + traffic delta for quant; wall-at-
parity for epilogue; 40dB + wall for separable -- each a 1-gate arc).
