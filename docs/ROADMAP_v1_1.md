# v1.1: the composition release (mixing blocks safely)

v1.0 proved one block in one scale regime (84dB) and that strangers can
extend the language (~3 min/drill). v1.1 answers the next question: can
TWO blocks with DIFFERENT dynamic ranges share one listing — and can a
listing leave numpy for C? The vision (bespoke AIs by mixing geometric
shapes; architecture replacement à la the transformer→SSM swap) needs
both: mixing needs per-block scales, emission needs C lowerings.

## Spine (in order — each unlocks the next)

### 1. Per-block scales (load-bearing)

Single-m drowns small blocks: one stage can span 1000x dynamic range
(phi-core handoff §4: 21.9dB single-m → 33dB per-block; our own Gate 1
went 84dB → 14dB outside its envelope). Deliverables:
- Per-block M-dict format (extends chain/M.json, which stays frozen for
  the holo path — new regimes ADD dicts, never move the old one).
- Declaration syntax (CONFIG overrides first; DEF-header `@scale`
  declarations per the handoff as the verifier end-state).
- Seam rule: RESCALE at block boundaries (the mnemonic already exists —
  the spike must prove the mechanism, not invent it).
- Saturation audit: fail loud if the kept region rails (>0.95·2^18
  fraction assert, the llama pattern).
- Gate: mixed-scale listing parity ≥40dB vs torch (same bar, same basis).

### 2. Matmul C lowering (first transformer-path C)

c_core covers arith/bridge/conv/geo — every transformer op is numpy-only.
Pattern exists (c_chain: file-exchange bit-exact + `make trap` clean);
matmul goes first because everything flows through it. Gate: file-exchange
bit-exact on 4+ cases incl. B-broadcast + trap clean. Unlocks the
"listings emit to C" claim for the transformer path; CUDA follows the
conv.cu precedent, explicitly NOT in v1.1.

### 3. S19 bet (time-boxed_parse, don't build)

Is diffusion sampling just temporal IIR with a=1.0 (INVENTORY S19 open
question)? Recognition-first porting: parse, don't implement. Success =
a second S19 instance entered in INVENTORY (promotes IIR from trick to
structure) or a dated negative (the question stays open honestly).

## Absorbed (v1.1 backlog F1–F4 — queued, not forgotten)

F1 portable setup (`env -i` verified stranger rerun); F2 mask example in
tutorial; F3 SELECT float-coercion decision (refuse vs document, with gate
+ full-suite green — strictness is earned); F4 `rgb_array` nit.

## Acceptance gates (all must hold)

1. Mixed-scale listing parity ≥40dB vs torch (peak=1.0, fixture reported,
   in-contract tripwire — the Gate 1 pattern).
2. Matmul C file-exchange bit-exact (4+ cases) + `make trap` clean.
3. F1–F4 closed or re-evidenced with a dated reason.
4. VELOCITY.md continues (every item wall-timed).

## Out of scope (named, not forgotten)

Full CUDA emission, WHILE + dynamic shapes, per-tile routing, rank
arithmetic beyond v1 layouts, phi-core merges (owner action).

## Declared 2026-09-30 (all four gates hold)

1. Mixed-scale listing parity: 65.1dB vs 24.0dB frozen (test_xf_block.py
   block-mixed-*, #LIB-031).
2. Matmul C bit-exact: 5/5 exchange cases + trap clean (test_matmul_c.py,
   wired into test_c.py, #LIB-032).
3. F1–F4 closed with evidence (docs/V1_1_BACKLOG.md; SELECT refusal +
   env -i stranger rerun).
4. VELOCITY.md continued (GELU entry; drill cadence established).
19/19 suites green. v1.1 is done; the deferred list above seeds v1.2.
