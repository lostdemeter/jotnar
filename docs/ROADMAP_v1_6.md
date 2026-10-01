# v1.6: an LLM from scratch, constructed (templates x relations x stores)

v1.5 proved stores work (read/write/delete/predict) on borrowed weights.
v1.6 builds a limited LLM experience with OWN data: no trained weights,
no HF model -- constructed content (templates, relations, fingerprints)
in assembly listings over ENGRAM stores, evaluated by inherited batteries.
"Constructing"/"building" (Echion's word), never "training".

## Reusable survey (measured 2026-10-01, not assumed)

**From Echion_Revisited** (parallel program, theory-first, 106-test suite):
- EVAL: qa_battery (41 q/expect) + v2 (32 + shape/note) -- behavioral
  gates with expected outputs, ready to run.
- CONTENT: voice_pairs (8 active/passive twins), edge_store (relation
  graph), grokipedia HTML (Alexander/Cleopatra/Mark Antony factual),
  relation_probe (relation vectors: consistency + transfer, S12).
- MECHANISMS: 10 templates + mining/search/promotion (sentence geometry),
  MI-profile fingerprints (GloVe-free cells), bootstrap (teacher->student
  walks), store.py (versioned JSONL + digests + refusal -- a STORAGE
  precedent with manifest discipline!), intgate (no-float CI gate).
**From HF cache**: wikitext (7.5M + 301M) for counts; SmolLM2-135M,
GPT-2 family, mamba-130m for comparative anatomy (describe first).
**From Jotnar**: assoc_mem (recall machinery), ENGRAM stores + predictor,
labeling loop, edit receipts, emission driver.

## Spine (in order)

### 1. Comparative anatomy (recognition-first)

One block each in assembly terms (listings or documented deltas):
SmolLM2 (smallest/fastest loops), GPT-2 (literature bridge -- most probing
work targets it), mamba (non-transformer pole; scan_step already staged),
Echion templates (the 4th column: no weights at all). Output: invariant
list = "how LLMs are supposed to work" in our language + a variance table
(norms? activations? GQA? biases? templates vs weights?).

### 2. Data freeze (construct from...)

wikitext counts (bigrams? skip-grams? -- measured sparsity decides) +
Echion batteries/pairs/edges/grokipedia as content+eval. Freeze format per
S17 (hash-key sidecars) or echion-store/1 (manifests + digests) -- decide
by writing both small and comparing (no armchair).

### 3. Construction (the model)

Template frames x relation edges over ENGRAM stores, retrieval via
assoc_mem machinery, fingerprints as verification. Small vocabulary,
narrow domain first (the grokipedia trio? voices?). Behavioral gates from
qa_battery shapes (expect fields!), not vibes.

### 4. Experience + loop (close it)

Generate, measure (perplexity? battery pass rate? -- stated metric),
then run OUR loop on OUR creature: read its stores, label, implant,
verify. The first model we fully understand because we built it that way.

## Acceptance gates (all must hold)

1. Anatomy: >=3 models described + invariant/variance tables.
2. Data frozen with provenance (either format, justified).
3. Constructed model passes stated behavioral gates (batteries).
4. Loop turn on own creature (read/label/implant/verify green).
5. VELOCITY.md continues, CLOCKED.

## Out of scope (named, not forgotten)

Scale (tiny vocab, narrow domain -- deliberately); training machinery
(counting + fitting only, no backprop program); fluency competitions
(coherence judged by stated batteries, nothing more); phi-core merges
(owner action); T-transform follow-through (branch pending review).
