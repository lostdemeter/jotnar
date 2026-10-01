# CRUD for weights: goal, plan, and research program

**Goal:** weights as assembly data. Read, update, delete, and create model
weights through the assembly language -- every operation with a predicted
dB cost confirmed by measurement, every artifact gated. A matmul is a sum
over directional stores (key `v`, value `u`, gain `s`); stores are declared
as data, consumed by structure, edited as text.

**Ceilings (stated up front, not discovered late):** bit-lossless
multi-hop is impossible in principle -- lattice quantum taxes every
materialization (~0.2% per hop). The honest ceiling is near-lossless
(60-90dB single-hop, quantum-tax bounds beyond). CREATE with intended
semantics needs labels at volume (3 verified today); CREATE with
functional aim works now.

## Operations (each: predict cost -> execute -> confirm)

- **READ:** direction_readout + predictor (DONE: +/-0.3dB blind).
- **DELETE:** prune stores by predicted share (THIS ROUND: all 27
  predicted-dead dirs in one edit, combined cost predicted statically
  from linearity, confirmed in one run).
- **UPDATE:** retune a store's gain / rewrite a store's content
  (implant machinery exists; gain-retune ungated -- research).
- **CREATE:** new store from a verified label at a freed shelf
  (needs label + shelf + aim; all three exist separately -- compose).
- **STORE:** versioned weight-store format with roundtrip parity
  (pieces exist; unbuilt -- research engineering).

## Language shape (v1.4 horizon, designed not built)

- `STOREBANK` sections: `(key, value, gain)` triples as declared data.
- Assembler-side SVD (expand-time, host -- like IMPORT/macros today):
  no in-lattice SVD, ever.
- Store-apply from existing ops (gather + tmul + accumulate family);
  `implant_apply` is already the rank-1 case, running.
- Predictor as the assembler's cost model: cut/store-X previews dB
  before emitting. Pruning and retuning become listing edits.

## Research questions (in order, each falsifiable)

1. Combined-cost linearity: does the 27-dir removal land inside the
   statically predicted band? (THIS ROUND -- below.)
2. Gain retune: does scaling one store's s move output by the predicted
   dB with everything else fixed? (Predict from the law; run once.)
3. Label-to-store: does a store built from a verified label behave as
   the label says on a fresh prompt? (CREATE demo; needs loop volume.)
4. Shelf reuse: does writing new content at freed 864/872 disturb less
   than writing at live directions? (The shelves' purpose, tested.)
5. Cross-layer: do the laws hold past layer 0? (One matrix, one prompt.)

## Non-goals

Bit-exact multi-hop parity (physics says no); full-matrix synthesis from
intent (frontier, needs labels x many); in-lattice SVD (wrong layer --
assembler-side by design).
