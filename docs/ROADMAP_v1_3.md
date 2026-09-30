# v1.3: the content release (what's in the weights)

v1.2 proved one listing runs on three substrates. Translation (transformer→SSM,
sigmoid transplant) proves behavior survives re-housing — without comprehension.
v1.3 asks the content question: what do the weights contain, in a form we can
work with? Shapes are characterized; content is not. This release makes content
claims falsifiable the same way shape claims are: with numbers and gates.

## Spine (in order)

### 1. Weight census (mechanical, thorough)

Per-block characterization: distributions, saturation fractions, dominant
channels/directions. Runs first on our own fitted params (CTRL.json, v5 gate
weights, M coverage — real numbers, small) to shake down the method, then on
≥1 real model family via phi-core handoff (they hold the staged ports; same
coordination pattern as the GELU drill). Gate: census report + a planted-
structure fixture (synthetic weights with a KNOWN dominant channel — the
census must recover it within tolerance; randomized control must not report
it — tripwire against wallpaper censuses).

### 2. Causal probe (the bet — one probe, fully gated)

Zero/swap a stream inside a listing, measure the output delta in dB with a
STATED threshold beforehand, plus a negative control (ablate an irrelevant
stream → near-zero delta). Primitives all exist (SELECT + parity); the pattern
is new: interventional interpretability with parity discipline. Candidate
first targets: the v5 gate's coherence term (flagship) or an MLP intermediate
(xf_block). Success = one probe with threshold met + control clean, entered
in INVENTORY as a method instance. Failure honestly reported also counts
(falsification is the paper-first discipline working).

### 3. INVENTORY content entries (recognition-first)

≥2 PREDICTED content-level structures with canonical-form sketches (feature
directions? circuits? cross-model content isomorphisms — S11's open second-
instance question lives here). Probe results fold back in (confirm, split,
or close). S08 (attention as selection) gets its content reading attempted
— flagged as the biggest unverified claim, stays flagged if unproven.

### 4. Labeling loop (stated-hard, deferred with reason)

"Stream X carries diagonal energy" → human-workable concepts needs a labeling
loop, and labeling isn't obviously geometric — no gate design exists yet.
Deferred explicitly (not missing): v1.3 builds everything UP TO the loop
(census + probes + comparison scaffolding) so the loop, when designed, has
instruments to plug into.

## Acceptance gates (all must hold)

1. Census method + planted-structure recovery gate + report on ≥1 real
   family (via handoff) or a dated reason it couldn't happen.
2. One causal probe, threshold stated beforehand, control clean.
3. ≥2 PREDICTED content entries with canonical sketches; probe folded in.
4. Labeling deferral recorded with reason (this section IS the record).
5. VELOCITY.md continues — CLOCKED this time (v1.2 process fix in force).

## Out of scope (named, not forgotten)

Full mechanistic theory; dictionary learning à la SAEs (related work, not
planned — our exact-lattice version would differ in kind, design first);
per-DEF `@scale`; xf C/CUDA lowerings (matrix refusals still pending);
phi-core merges (owner action).
