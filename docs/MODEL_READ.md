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

## The factorization (the fundamental connection, 2026-10-01)

Ablation delta factorizes: removing s.u.vT moves output by s.(X.u)(+)v
(exact identity in reals, gated as read-factor-form). So global dB =
-20log(s) - 10log(mean_t alignment^2) + C, where alignment = |MID[t].U_i|.
Define residual = dB + 20log(s): it should EQUAL minus-log-alignment with
nothing left over. Measured on the 112-readout: **corr 0.998**. The store
model (Wx = sum of key-value stores) is quantitatively exact and content
== key alignment, computable STATICALLY with zero runs. Consequences:
(a) HOW MUCH any direction matters is predictable without running
(anything left over after energy+alignment is the only true surprise
left in magnitudes); (b) match-step of the loop (does d follow P?) can
run on alignments (~ms) instead of ablations (~100s); (c) selectivity
spread is moderately predicted the same way (corr 0.64 -- dB nonlinearity
+ quanta, stated). Labels ARE alignment profiles: operational, geometric,
no semantics required.

## Labeling loop, run one: (dir24, France) VERIFIED (2026-10-01)

Hypothesis (from Q2 above): dir24 carries France-content. New prompt with
France at position 5 (was 3): `My/friends/from/school/visited/France/
yesterday/morning` (loop1.py, chain/qwen_mirror.py promoted helper).
- MATCH: silence dir24 -> France-pos rank **1/8** at 33.9dB (band: top-3
  + <=45dB). Fingerprint followed the content, not the position.
- VERIFY: write dir24's output direction (Vt[24]) keyed at France ->
  France rewritten at **-12.7dB**, gap **24.4dB** (band: gap > 6dB).
Both bands held with margin. First verified label: S21 goes PREDICTED ->
SINGLE (second instance wanted). The loop's first full turn took one
script and ~6 listing runs.

## Labeling loops two and three: CONFIRMED (2026-10-01)

Generalized script (loop_turn.py: dir/token/prompt/pos/A as argv; replay
of loop one bit-identical at 33.9/-12.7, proving the generalization safe).
Same bands (match top-3 + <=45dB, verify gap >6dB):

| loop | label | match | verify |
|---|---|---|---|
| 2 (dir32, capital, Berlin-pos3) | rank 1/8, 31.7dB | -12.2dB, gap 32.7 | CONFIRM |
| 3 (dir328, Paris, Yesterday-pos1) | rank 1/8, 33.4dB | -15.1dB, gap 25.9 | CONFIRM |

3/3 matches rank ONE (bands asked top-3); 3/3 writes negative-dB with
24-33dB gaps. S21: CONFIRMED (three instances). Caught en route: the
generalization shipped `s[24]` unparameterized (wrong-magnitude ablation)
-- found by reading, fixed, replay-verified identical. Generalization
without replay proof is how mirrors drift (cf #LIB-014 postscript).

## Using it: blind prediction (2026-10-01)

Calibrate C on the 112 measured dirs (C=29.347), predict three NEVER-RUN
directions from statics alone -- locked before running:

| dir | predicted | measured | err |
|---|---|---|---|
| 100 | 39.3dB | 39.5dB | +0.2 |
| 300 | 43.6dB | 43.8dB | +0.2 |
| 700 | 43.2dB | 43.5dB | +0.3 |

Edit-with-preview works: any intervention's delta is knowable before it
runs (calibrate once per matrix/regime, predict freely). The consistent
+0.2-0.3 bias is noted, not explained (quantization floor candidate --
one line, not a theory). Predictor standardized as calibrate_C/predict_db
(chain/read.py, logic gated exact as read-predictor; lattice accuracy
measured at ~0.2dB). Uses unlocked: cheap full readouts (statics + a
calibration handful instead of full sweeps), principled pruning (cut by
predicted share with stated dB cost), implant aiming (pick output
directions by predicted effect).

## Full 896-direction predicted map (2026-10-01, zero runs)

docs/readout_down_896.csv: all 896 dirs predicted from statics (SVD once
+ one MID matmul + C=29.347). Range [21.9, 61.5], giant dir0 at 21.9.
Predicted dead (>55dB): 27 dirs, mostly tail (797+) plus mid-spectrum 226
(big direction pointing at nothing -- low alignment despite energy).
Top-10 predicted vs measured-top-10 overlap: 3/10 (0/8/16 shared) --
values predict at +/-0.3dB but neighbor-ranks reshuffle within +-2dB
(measurement noise + quanta among near-ties). Rule: cut/prune by VALUE
threshold, never by rank. Blind trio now gated standing in test_realw.py
(realw-predict, worst err 0.22dB in-suite).

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

## Multi-context shelf map (third prompt: code)

Prompt 3 (`def/fibonacci/(n/):/return/n/if/n`, code domain): range
[23.0, 54.4], giant dir0 at 23.0 (invariant to ~1dB across all three).
Pairwise global Spearman: 0.55 / 0.46 / 0.47 (consistent moderate).
Shelf map at 55dB (chain/read.py shelf_map):

| context | dead (>55) |
|---|---|
| geo | 864, 872 |
| quantum | none (max 54.9) |
| code | none (max 54.4) |
| **intersection (safe everywhere)** | **empty** |
| union (dead somewhere) | 864, 872 |

The empty intersection is a finding, not a failure: at 55dB NOTHING is
dead on all three contexts (both near-misses sit just under bar). Shelves
are threshold-fragile -- storage claims need multi-context reads (the fix
LIB-051 prescribed, now built), and "free shelves" at 55dB do not exist
for this matrix. The tail (864/872) remains the closest thing to free
storage. Design consequence, recorded: writes should TARGET the union
with per-context disturbance budgets, not assume a safe intersection.
