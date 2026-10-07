# Siphon addressing: content without an address is noise (2026-10-06)

Question: why do strong steering directions install globally instead
of editing? Answer, measured: the implant is content without an
address -- like a positional encoding with the position stripped.
The fix is two channels (lexical address x late content), composed.

## 1. Broadcast implants cap at region overwrite (falsified clean)

L27 France-direction, gains 0.25-1.5, three direction forms, Italy
control: rank 1 from a=0.6 with margin to 1.31, but Germany, Italy,
AND Japan all land on Paris. L26 stalls at rank 3. Decoder-form
overshoots to 150k-rank junk past a=0.5. No gain/layer/form
separates (research/clean_screen.py, research/far_control.py).
Single-direction additive steering cannot edit; it broadcasts.

## 2. The reader obscures the content (norm map, not argmax)

Steered hidden is most direction-aligned with GERMAN tokens
(cosine Paris rank 46022) while argmax picks Paris -- decided by
row magnitude (Paris rows ~12x mean norm), not content. Blanks on
top elsewhere are high-norm frequent fillers outshouting
better-aligned content. Grade installs by the dot reader (where
content lives), not argmax alone (research/reader_content.py).

## 3. Conditional application restores specificity (confirmed)

Gate on the country token's early residual (layer 2, still lexical)
instead of full-state cosine (same-template states are
near-parallel: 0.61/0.54 unusable; lexical-early separates, ^6
sharpening decides 1.0 vs 0.06):

| target | gate | top | paris-rank | control |
|---|---|---|---|---|
| Germany | 1.000 | 巴黎 | 3 | hold (moved, correctly) |
| Italy | 0.061 | Rome | 4 | hold |
| Japan | 0.066 | ______ | 7 | hold (unchanged) |

(research/lex_gate.py; shared mirror research/qwen_torch.py --
third inline copy refused.)

## 4. The siphon product shape

Keyed conditional writes: address (early/lexical key match) x
content (late direction) x dose. This is exactly the assoc/bankhn
store form (keyed retrieval + conditional application) with mined
late directions as content -- and it is also why argmax-only
evaluation understates it. Next: port the gated install to a
builder listing (SELECT-gated steering vector: frozen key match +
steering add), proving the product is a program.

immaterializes as a figure: left, the flow with the problem
located (extract/key green, INSTALL amber, argmax red, dot green);
right, the measured margin curve with the empty clean region.
See `gallery/siphon.png` (built by `scripts/fig_siphon.py`).

## 5. Dose discipline: the controls hold (2026-10-06, corrected)

An earlier "margin fragility" verdict (random flips Italy too) was
VOID: the dose file never rewrote, so every control secretly ran at
Germany's dose 220. With true per-prompt doses, through listings:
Germany dose 220 -> Paris rank 1; Italy dose 13.5 -> blank top,
Paris rank 45; Japan dose 14.4 -> blank top, Paris 99; random
direction dose 13.5 -> blank top, Paris 45 (both paths agree now).
Lex-gating + dose discipline = clean install (target exact,
neighbors hold). The bar stands (margin above local noise), but
the evidence for endemic fragility is withdrawn -- razor picks
here hold against whisper breezes 7/8.
Tripwire, third instance of the species: IDENTICAL gates/values
across supposedly-different runs mean cached inputs (this time
prompt-dependent files skipped by exists-checks). Live inputs
always rewrite; the served binaries can't tell.

## 6. Probe-indexing bug + direction recipe + stale binaries (2026-10-06, dissected)

Three layered causes behind one confusing week, separated:
(a) My validation probes indexed `traj[0][L]` (layer-0's L-th
SCALAR) instead of `traj[L]` (layer-L row) -- silent broadcast,
no error, wrong numbers everywhere I wrote it. The siphon
result scripts index tuples (`tfr[0][0][27]`, `tG[2][gpos]`)
and were correct throughout; only ad-hoc probes lied. Species:
probe-vs-pipeline indexing -- verify shapes in probe output,
always.
(b) Stale direction/dose files across runs in one workdir
(random dsteer overwrote genuine; dose files skipped by
exists-checks). Fix: live inputs always rewrite; fixture reuse
explicit (QWEN_REMINE); source-hash stamps on binaries.
(c) Direction recipe AND orientation: PAIR contrast installs
rank-1 through listings (proven); vs-mean reads rank-250 but that
measurement is confounded (scalar bug + orientation) -- properly
VOID, queued cleanly. Orientation is load-bearing: France-minus-
Germany installs (rank 1), its negation anti-installs (rank
19912, cosine -1.0 measured live). d = contrast-minus-country,
never the reverse. Demo uses pair + correct orientation.
