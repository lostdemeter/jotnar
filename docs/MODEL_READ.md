# MODEL READ: Qwen2-0.5B layer-0 down_proj (first demonstration)

What a "read" of a model looks like with Jotnar instruments: a directional
readout table (112/896 singular directions sampled every 8th) over 8 real
tokens, plus three searches answered from it. Method: chain/read.py;
gates: test_read.py (instrument correctness on toy). This file is the
demonstration, not the gate.

## Setup

- Matrix: `down_proj` layer 0 (896x4864), trained weights, spectrum
  decaying 17x (3.54..0.21).
- Input: 8 tokens `The/capital/of/France/is/Paris/,/and`, real embeddings,
  H via full float64 attention boundary (test_realw pattern).
- Rows: ablate direction i, record global dB + per-token delta vector.
- Cost: 112 listing runs, 116s.

## Q1: dead shelves (removable storage)

2/112 sampled dirs above 55dB: **864 (55.1dB), 872 (58.1dB)** — both tail
(sval ~0.6). Extrapolated crudely, the tail holds a handful of free
shelves: capacity where writes land with minimal disturbance.

## Q2: who moves France / Paris?

Top-5 directions by token delta:

| token | top dirs (dir: dB) |
|---|---|
| France(3) | 0: 24.9, 8: 28.0, 16: 30.2, 72: 32.9, 24: 34.0 |
| Paris(5) | 0: 23.8, 96: 31.6, 72: 31.9, 8: 32.1, 328: 33.4 |

Shared giants (0, 8, 72) move both; 16/24 lean France, 96/328 lean Paris.
Direction 0 (top singular) is the giant everywhere (22-25dB).

## Q3: broadcasters vs specialists

Most uniform: 872 (4.9dB spread — dead, uniform because it moves nothing),
824, 160, 344. Most selective: **328 -> Paris (25.5dB)**, 80/24 ->
France, 32 -> capital (28.7dB). Selectivity itself spreads ~29dB.

## Caveats (load-bearing, read before citing)

1. Sampled 1/8: 112 of 896 directions. Rankings among neighbors are
   unresolved; giants and shelves are robust.
2. Fingerprints are POSITIONAL, not semantic: "moves France" means moves
   position 3's output on THIS prompt. Whether fingerprints stabilize
   across contexts is the labeling loop's question -- this read is its
   input, not its answer.
3. One prompt, 8 tokens, layer 0 only. Generality unclaimed.
4. Correlation note (LIB-046 comment): dir-dB tracks sval partly
   algebraically (removed energy). The spread and the ranking are the
   content; the correlation is the sanity.

## Cross-context stability (second prompt, same 112 dirs)

Prompt 2 (different domain, subword tokens):
`Quant/um/computers/manipulate/q/ubits/using/super`. Same readout,
112 runs, 115s. Comparison (measured rows, no bars yet per LIB-015):

| measure | geo | quantum | stability |
|---|---|---|---|
| global range | [22.1, 58.1] | [22.0, 54.9] | same scale |
| giant dir0 | 22.1 | 22.0 | identical to 0.1dB |
| global-dB rank Spearman | — | — | **0.55** |
| top-10 causal overlap | — | — | **6/10** |
| selectivity rank Spearman | — | — | **0.25** |
| dead shelves (>55) | 864, 872 | none (max 54.9) | near-miss, not overlap |

Reading: HOW MUCH a direction matters is partly context-stable (giants
stay giants; 0.55 rank correlation across domains); WHICH token it lands
on is contextual (0.25 — expected: fingerprints are positional, and the
positions hold different tokens). This mirrors the fixed-weights /
varying-inputs separation from the route trials: stable WHAT (weight
property) + contextual WHERE (input property). The labeling loop's input
just got its second column: stable magnitudes to track across contexts,
contextual fingerprints to match within them.
