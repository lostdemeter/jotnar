# Siphon install through the yarn ball (Qwen2-7B, 2026-10-07)

First accurate modify through the generic structure: one ball
(address x content x dose, ledgered) installs Germany -> Paris at
top-1 with controls holding, on the torch mirror AND in a 9866-op
geometric listing. Recipe: L27 pair-contrast content x L2
lexical-early address, null stores for non-targets (hold by
construction). Scripts: `research/siphon_ball.py` (mirror),
`research/addr_sep.py` (listing-design probe), `scripts/siphon_geo.py`
(listing). Ball data: `chain/engram.py:yarnball_bank()`; structure:
`stdlib/yarnball.asm`.

## Setup

- Model: Qwen2-7B-Instruct (snapshot f2826a00, float32 mirror;
  FP16/CUDA geo backend). Prompts: "The capital of X is" for
  Germany (target), Italy, Japan (controls).
- Content: L27 prompt-end contrast France-minus-Germany, unit
  (orientation load-bearing per siphon section 6: the negation
  anti-installs). Address: L2 country-token residual, unit, per
  country, live-mined every run (no caches: stale-direction species).
- Ball: 3 stores [Germany: contrast value | Italy: null |
  Japan: null], keys x key_scale 8.0. Ledger tiers assoc/null/null
  (`/tmp/siphon_geo_ledger.json`; dose row below is the zero-dose
  control's file -- gain-1 dose was 536.6 = 1.0 x |x27|, build log).
- Mirror apply = yarnball_apply math (softmax over bank routes at the
  country row; exact add at L27 prompt-end). Listing apply = the DEF
  body inlined at L27 output (CALL lowering gap, docs/GAPS.md),
  explicit country-row routing (addr_sep below forbids self-gating).

## Mirror ladder (gain in residual-magnitude units)

| gain | Germany | Italy | Japan |
|---|---|---|---|
| 0 (unsteered) | Berlin | Rome | blank |
| 1 | **Paris r1 INSTALL** | Rome hold (Paris r51) | blank hold (Paris r93) |
| 2 | **Paris r1 INSTALL** | Rome hold (Paris r30) | blank hold (Paris r63) |
| 4 | Paris-surface r3 | Rome hold (Paris r12) | blank hold (Paris r30) |
| 8 | Paris-surface r3 | Rome hold (Paris r4) | blank hold (Paris r9) |
| 16 | Paris-surface r3 | **MOVED (->Paris r1)** | blank hold (Paris r2) |

Every cell retrieves its own store (argmax P == own index).
Operating point: gain 1 (English-form top, all hold, all retrieve).
Dose discipline: target installs at 1-2, holds through 8, Italy leaks
at 16 (partial-weight x large-dose species, same as the native ladder).
Note the surface shift with dose (English rank-1 at 1-2, native-form
rank-3 at 4+): content identical, the LENS_7B filter rhyme.

## Address separability (listing-design probe)

All prompts length 5, country token at row 3, prompt-end at row 4
(one binary serves the battery). L2 row match vs bank keys:

| row | Germany prompt | Italy prompt | Japan prompt |
|---|---|---|---|
| 0-2 (shared) | tied ~0.3 | tied ~0.3 | tied ~0.3 |
| 3 (country) | **own 1.0** vs <=0.64 | **own 1.0** vs <=0.63 | **own 1.0** vs <=0.64 |
| 4 (end) | tied ~0.42 (Japan 0.425 ahead) | tied ~0.41 | tied ~0.43 |

Consequence recorded BEFORE building: per-row self-gating would
misroute at the read row; the listing SLICEs row 3 and routes
explicitly. Tiling the value to all rows is airtight (post-L27 ops
are all row-wise; only row 4 is read).

## Listing verdict (gain 1, 9866 ops, CUDA/FP16)

| prompt | geo zero-dose base | geo ball | grade |
|---|---|---|---|
| Germany | blank (Paris r45) | **' Paris' (Paris r1)** | INSTALL (effect 45 -> 1) |
| Italy | blank (Paris r75) | blank (Paris r75) | HOLD (bit-identical) |
| Japan | blank (Paris r137) | blank (Paris r137) | HOLD (bit-identical) |

Null-store routing leaks exactly zero in-listing (control ranks
unchanged to the integer).

## Caveats (load-bearing, read before citing)

1. **Fork baseline.** Geo FP16 base already blanks Germany (mirror:
   Berlin) and Italy (mirror: Rome) -- near-tie fork per the
   test_qwen7b_gen doctrine. Holds grade vs the geo zero-dose base
   (`--base-json`), never vs mirror. First grading caught this live:
   mirror-graded verdict misread Italy as broken; the control proved
   it pre-existing (tripwire species: cross-baseline grading).
2. One fact, one template family. Country@3 is baked as SLICE
   literals; non-conforming prompts would misaddress silently (loud
   refusal designed, not built). Multi-fact teacher-side bank unbuilt.
3. Retrieval in-listing is implied-by-effect (target got the value,
   controls bit-identical), not read from the YBP stream (mirror side
   reads it explicitly: all RETR). Dumping YBP geo-side is backlog.
4. Dose mapping mirror->listing (Vc row = d x gain x |x27|) is host
   arithmetic outside both listings -- dose discipline across
   substrates, priced, not derived.
