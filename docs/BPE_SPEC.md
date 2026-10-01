# BPE spec (behavior bridge): subword tokenizer, counted not trained

Status: SPEC (measured trigger: word-2k bought coverage 0.826 but cost
top1 0.406->0.252 / ppl 47->390 -- mass shatters across finer ids;
subwords SHARE mass via recurring pieces, the concentration word-2k lacks).

## Doctrine fit

- Freezing is offline (counts in, merges out, frozen like scales);
  runtime is exact integer moves on id sequences (CONCAT/SLICE/GATHER
  family -- the exact-moves census class, no statistics inside).
- Count-first: pairs counted on frozen corpus, greedy most-frequent
  merges, zero gradients anywhere (construction only, per design).
- Host boundary first (like temperature/top-k): encoder/decoder as
  audited host code with frozen files; listing form later, parity-gated.

## Freeze (offline, `scripts/bpe_freeze.py`)

1. Corpus: grokipedia HTML sentences first (same 938/235 split, seed 0
   -- comparability with word-512/2k), wikitext second.
2. Base: lowercase `[a-z0-9']` chars + word-boundary marker + UNK-proof
   byte fallback (OOV ~0 by construction vs 39% word-512).
3. Loop: count adjacent pairs, merge most frequent, re-count, until
   N merges (start N=2000 -> vocab ~2.1k incl. base). Deterministic
   (sorted traversal, frequency-ranked, ties alphabetical).
4. Artifacts (frozen, provenance): `data/bpe_vocab.json` (id->piece),
   `data/bpe_merges.json` (ordered pair list), `data/bpe_manifest.json`
   (corpus shas, N, coverage, counts sha) -- S17-style sidecars.

## Runtime (host, exact)

- `encode(s)`: split to base chars, apply merges greedily in frozen
  order (first-applicable, longest-match by order rank). Pure integer
  sequence ops, deterministic bytes.
- `decode(ids)`: concat pieces, strip boundary markers. Exact inverse.
- Listing form (backlog): merge cascade as GATHER/CONCAT composition;
  host-vs-listing parity bit-exact when it lands.

## Gates (all must hold)

1. Roundtrip exact: `decode(encode(s)) == s` on all train+test (100%,
   not sampled -- own scale admits proof, per the 513-rows precedent).
   MEASURED 2026-10-01: 0 mismatches.
2. Coverage: subword OOV rate <1% on grokipedia test + reported (not
   barred) on wikitext-2 sample + qa batteries.
   MEASURED 2026-10-01: 0/14399 grokipedia; 1.71 pieces/token.
3. Behavior converts at the RIGHT context order (amended 2026-10-01:
   bigram-over-pieces is capability-mismatched -- pieces stretch a
   word over ~1.7 steps, so a 1-step model loses word memory;
   measured piece-bigram top1 0.15 / word-rollout 0.035 vs word-512
   0.406, recorded as mismatch evidence, NOT failure). Bar moves to:
   transformer-on-pieces (S=8 window, non-random weights) word-top1
   vs word-512 bigram on the SAME split. Counts-injected or fitted
   weights are the prerequisite construction, separate spec.
4. Determinism: same input bytes twice bit-identical (no hidden state).
5. Mass check: piece-frequency Zipf reported (concentration visible:
   top pieces recur across words -- the mechanism word-2k lacked).
   MEASURED 2026-10-01: top-200 pieces 54% of mass.

## Out of scope (named)

- Listing-form encoder (backlog with parity bar); wikitext-scale merge
  counts (measure grokipedia first); case sensitivity (lowercase first,
  stated); counts-injected transformer weights (sibling bridge, separate
  spec if BPE pieces feed embeddings).
