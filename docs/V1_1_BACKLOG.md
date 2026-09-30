# v1.1 backlog — stranger-test friction log (v1.0 Gate 6)

The stranger test (test_stranger.py) passed first try from docs alone, so
v1.0 is NOT blocked. Everything below is queued work, not forgotten work.
Each item: friction observed, repro/evidence, proposed fix.

## F1. Tutorial setup is not portable (biggest friction)

The §1/§3 snippets hardcode `sys.path.insert(0, "/home/thorin/...")`.
A fresh instance on another machine cannot copy-paste them. The exercise
ran only because the paths happened to exist here.
Fix: relative setup instruction (run from repo root + phi-core checkout
note, or PYTHONPATH), verified by re-running the stranger test with a
clean environment (`env -i` + documented vars only).

## F2. No bool-payload example in the tutorial

§2 says streams include bool arrays and §4 says SELECT takes a bool mask,
but the tutorial only ever encodes triples. The stranger must INFER
`np.array([True, ...])` as a payload. The inference worked (green first
try), but one line in §3 would remove the guess:
`m = np.array([True, True, False, False])`.
Fix: add the line + one SELECT sentence to the tutorial.

## F3. SELECT silently coerces float masks (unannotated streams)

Repro: `m = np.full(4, 0.5)` (float64) as the SELECT mask on a bare
`IN m` (UNKNOWN layout) RUNS — `float 0.5` becomes `True` via
`ascontiguousarray(bool)` inside `op_select`. The layout-kind cross-check
skips UNKNOWN streams (gradual typing), so nothing fails loud. Truthiness
coercion of 0.5->True is the surprising half (0.0->False reads naturally).
Fix (needs its own gate + full-suite green first — strictness is earned):
refuse non-bool masks in `op_select`, or document the coercion in
LANGUAGE.md §4 SELECT. Decide in v1.1.

## F4 (nit). Orientation snippet references undefined `rgb_array`

§1 shows `ASM.run_text(text, REGISTRY, rgb_array, sigs=SIGS)` without
saying what `rgb_array` is (a U8 HWC array — presumably an opened image).
Harmless (tutorial step 1 covers payloads), but one clause would close it.

## Non-frictions (positive log — keep these properties)

- Bare-payload mistake: `multi-IN program needs dict payload, got ...` —
  exact and actionable.
- Typo'd mnemonic: `unknown mnemonic: ADDD (known: [...])` — names the
  culprit and offers the vocabulary.
