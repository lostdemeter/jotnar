# LLM design (formal draft v0.1, from 3-row anatomy)

A from-scratch LLM in assembly, with every design decision priced by the
laws this program has measured. Nothing here is taste; each row cites the
gate that earns it. Status: DRAFT (anatomy complete 2026-10-01; nothing
built yet).

## Comparative anatomy (measured, not summarized)

| aspect | Qwen2-0.5B | SmolLM2-135M | GPT-2 | design pick |
|---|---|---|---|---|
| norm | RMSNorm, no bias | RMSNorm, no bias | LayerNorm + bias | **RMSNorm, no bias** (no mean-sub, no bias streams; 84dB proven) |
| activation | SwiGLU (SILU gate) | SwiGLU | GELU plain (tanh) | **SwiGLU** (2/3 families; SILU proven; GELU divergence 4.7e-4 noted) |
| positions | RoPE 1e6 | RoPE 1e5 | learned-add | **RoPE** (formula tables need NO training; learned emb needs fitting we refuse to pretend) |
| heads | GQA 14/2 | GQA 9/3 | MHA 12 | **single-head start** (listings today), MHA as growth (repeats) |
| biases | QKV biased (!) | none | everywhere | **none** (tiled-ADD pattern banked if ever needed) |
| scales | single-m fine | FORCED two-scale | big-m + biases | **per-block m** (goldilocks is definitional) |
| eps | 1e-6 | 1e-5 | 1e-5 | **true-eps per scale** (eps law; never counts) |
| scores | 963 (out) | ~8 (out) | 7 (out) | **in-contract only** until T-transform merges |
| sampling | — | — | — | **ARGMAX greedy** (exists); temperature/top-k = host boundary |

## The three laws (design constraints, all gated)

1. **Goldilocks scales**: cover tightly (m_of + standard margin). Bigger
   starves counts through isqrt/divide (measured non-monotonic both
   ways); smaller folds. Per-block M-dicts, never single-m.
2. **True eps**: eps_c(m) = eps*2^36/U_m^2 (SmolLM2: 47->84dB on fix).
3. **Fan-in floor**: parity_floor ~= 20log(|out| / (q_enc * ||terms||_2)).
   Budget dB per matmul from (K, magnitudes) BEFORE running: narrow FFN
   or small weights buy floor directly. Cancellation penalty rides on top
   where structured sums cancel (GPT-2 DOWN: predicted 46.6dB peakmax,
   measured 47.9 -- confirmed within 1.3dB).

## Construction plan (no training anywhere)

- Tokenizer: tiny, own data (wikitext counts for coverage; Echion
  batteries for eval shapes). Counts, not gradients.
- Embedding: random lattice points at calibrated magnitude (decode range
  checked vs m_cov BEFORE first run -- the SmolLM lesson: measure first).
- Weights: calibrated magnitudes (std chosen by the fan-in budget, not
  copied), frozen; spectrum checked (decaying, not flat-random).
- Positional: RoPE tables (frozen, formula).
- Growth order: single block, single head, S<=8 -> parity vs torch mirror
  (mirror built alongside, never after) -> heads -> depth, each with the
  three laws applied beforehand (predict the dB, then measure).
- Verification: perplexity on held-out counts? battery pass rates
  (Echion expects)? Both stated before building (demo-2 pattern).

## Open (stated, not missing)

Gated-vs-plain MLP (expressivity need unmeasured -- start SwiGLU, ablate
later); depth schedule (one block proven means N blocks composed, but
error COMPOUNDS: per-block budgets must sum -- first depth run decides);
sampling beyond greedy (host boundary until demanded); the +0.2dB
systematic in blind prediction (noted, unexplained, watching).
