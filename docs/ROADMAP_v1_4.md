# v1.4: attention-side stores (selection meets geometry)

v1.3 proved CRUD on matmul-consumed stores (Wx = sum of key-value gains,
previewed to +/-0.3dB). Attention consumes stores DIFFERENTLY: values
selected by softmax weights (dynamic gains) instead of summed with static
ones. v1.4 extends the store model across that seam: same declaration
(keys/values/gains as data), new selection-side semantics, S08's first
gated instance as the prize.

## Spine (in order)

### 1. QKV-projection storebanks (linear half, existing machinery)

in_proj/out_proj MATMULs as SVD storebanks with parity gates (down_proj
pattern verbatim: decompose, bank-form listing, prune-confirm). No new
math; brings the projections under CRUD. Proves the easy half and prices
the hard half by contrast.

### 2. Selection-side decomposition (S08 first instance)

Per (head, key-position): key, value row, DYNAMIC gain (the attention
weight). Gates: recomposition from (P, V) triples bit-near-exact vs the
torch path; per-position gain attribution (which keys the output came
from) with a stated bar. S08 (attention as selection) goes
PREDICTED-ish -> SINGLE here or stays flagged with a dated reason.

### 3. DDColor cross-attn layer in-assembly (conditional)

IFF folded real scores fit the softmax contract (measure first!): the
layer as a listing consuming frozen query stores, parity-gated. If scores
blow past (Qwen hit 964x), this gate waits on the T-transform and the
release ships 1+2+4 instead -- decided by measurement, not hope.

### 4. T-transform research spike (time-boxed, stated-hard)

Full-range attention without saturation. Success = a design with a parity
gate; honorable failure = a dated negative with the obstruction stated
more precisely than "scores get big".

## Acceptance gates (all must hold or be dated)

1. Projection storebanks parity + prune-confirm (both forms).
2. Selection decomposition with gain attribution (or dated negative).
3. DDColor layer listing IF contract holds (else T-transform waits).
4. T-transform spike run (design or dated negative).
5. VELOCITY.md continues, CLOCKED.

## Out of scope (named, not forgotten)

Full DDColor-in-assembly end-to-end (needs gate 3 AND packaging);
multi-head batching elegance (correctness first); listing-text pruning
(masked sums); per-DEF @scale; phi-core merges (owner action).

## Declared 2026-10-01 (38/38 suites green; all five gates hold)

1. Projection storebanks: 6/6 at 59-63dB + prune harmless 76.1dB with
   tax + floor corrections (test_projbank.py, #LIB-070).
2. Selection-side decomposition: P-invariance bit-exact, recompose
   82.9dB, ordering holds (test_select.py, #LIB-071; S08 SINGLE).
3. DDColor layer: WAITS by measurement (scoremax 429x, dd_score.py).
4. T-transform: DESIGN, better than spike brief (docs/T_TRANSFORM.md:
   fold obstruction, 9-layer ranges, one-assert fix + composition;
   #LIB-072). Staged phi-core branch: offered, owner action.
5. VELOCITY.md: clocked throughout (no missed starts this release).
v1.4 is done: attention consumes stores with the linear half banked,
the selection half instanced, and full range specified.
