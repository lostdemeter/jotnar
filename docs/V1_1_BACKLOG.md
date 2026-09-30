# v1.1 backlog — stranger-test friction log (v1.0 Gate 6)

ALL FOUR CLOSED (evidence below). Kept as the record of what the
stranger test found and what each item cost.
Each item: friction observed, repro/evidence, fix applied.

## F1. Tutorial setup is not portable [CLOSED]

Was: §1/§3 hardcoded absolute `sys.path`. Now: LANGUAGE.md §1 documents
the layout (phi-core sibling, run from root, PYTHONPATH alternative) and
states no env vars are read. Verified: test_stranger.py green under
`env -i PATH=...` (no hidden env deps).

## F2. No bool-payload example in the tutorial [CLOSED]

Was: mask construction inferred from §2+§4. Now: §3 shows the mask line
(`np.array([True, True, False, False])`) with the int/float rule pointer.

## F3. SELECT silently coerces float masks [CLOSED — refused]

Was: `np.full(4, 0.5)` mask ran (0.5->True). Decision: REFUSE (fail loud),
not document — float truthiness is the surprising half; bool/int keep the
numpy nonzero rule per the `I:*` kind. Gates: `asm-select-floatmask`
(refusal) + `asm-select-intmask` (int 0/1 == bool, exact). Full suite
green unchanged (strictness earned). MIXDYAD coerces the same way —
noted, untouched (out of scope, no gate).

## F4 (nit). Orientation snippet references undefined `rgb_array` [CLOSED]

Was: bare `rgb_array`. Now: annotated inline (U8 HWC array, e.g. opened image).

## Non-frictions (positive log — keep these properties)

- Bare-payload mistake: `multi-IN program needs dict payload, got ...` —
  exact and actionable.
- Typo'd mnemonic: `unknown mnemonic: ADDD (known: [...])` — names the
  culprit and offers the vocabulary.
