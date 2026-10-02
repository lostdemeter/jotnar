# v3.0: the usable-LLM release (definition of done)

v3.0 is the version a stranger can talk to, defined by ONE question:
"can it hold a coherent, fact-grounded exchange in its domain without
wrappers hiding the seams?" v2.0 proved we can build it; v3.0 proves
someone can use it. Every gate below is measured against today's
numbers (in parentheses); nothing is wished.

Out of scope for v3.0 (named, not forgotten): open-domain fluency
past wikitext-coil, dialogue state beyond prepend-context, learning
from interaction (frozen fits only -- doctrine holds until a
learning spec with gates exists), fluency competitions of any kind.

## Acceptance gates (all must hold)

1. **Word top1 >= 0.55** (today 0.472 depth-4). Path: D32 refit at
   w103/groki operating points, per-block M at width (vehicle
   proven), distinct-content banks. No gain tweaks (12-0 record:
   they hide blur).
2. **Raw glue <= 0.35, loops zero unblocked** (today 0.60 / nrep-3
   managed). Structural, not wrapper: content confidence from (1)
   must carry it; penalties stay as guardrails with their costs
   stated (gpen ~1 bigram, recorded).
3. **Piece generation quality: validity + grounding + flow, all
   demonstrated** (re-scoped 2026-10-02: word-top1 >= 0.20 moved to
   research horizon, below). Bars: zero inventions held
   (test_piece_bound green); retrieved OOV facts spoken verbatim
   (hybrid fact-prepend, demo_hybrid -- was seed-only surrender);
   loop-free + replay-identical generation (declared rules,
   verified); router mix reported per run (core/piece/fact parts).
   Rationale: operating-point moves exhausted (fit stands from ~16
   directions), fusion falsified both shapes, piece OOV aim 0/144,
   oracle union bounds the pair AT the old bar with no router
   margin -- the old bar is a research question, not a release
   gate (recorded in VELOCITY, research track owns it).
4. **Retrieval disambiguation closed**: probe misses (7/202 mined,
   ~11% w103m5 ambiguity classes) resolved by second-pass
   full-triple re-cue; v07 both sides retrieved; gate bars
   second-pass hit rate with first-pass numbers kept.
5. **w103 top1 >= 0.35 with ppl <= 150** (today 0.27/217). Path:
   w103-body HN banks v2 (rank-matched), D32 refit at w103 point,
   per-block scales at width. Transfer alone is proven insufficient.
6. **Full suite green + flagship demo.** Every existing gate holds;
   demo_flagship speaks from the v3.0 stack with fitness parts
   reported; audit re-run shows glue down structurally (not by
   penalty alone).

## Deliberately deferred (v3.x)

D64 revisit (parked with diagnosis; needs content-fit ideas, not
sweeps), BPE cascade listing (shrink-geometry design), interaction
learning (speculative until a gated spec exists), dialogue state,
scale past D64 (no evidence it helps yet).

## Research horizon (not release-blocking; measured rows, no bars)

R1. **Piece word-top1 0.20** (today 0.091 word / 0.058 piece).
Exhausted: refit, boundary scoring, banks to K256, temps, context,
priors (unbuilt, wrong direction), fusion (both shapes). Open:
distinct-content banks, content-gated sharpening (BOUNDED SPIKE
FIRED 2026-10-02: argmax net-zero BUT decision-mass moves
sub-threshold (truth +3.9 rank, 76% prob-up) -- coupling is
threshold-gated, not absent; navigation reframe (flat converges,
sharp discriminates); absolute-origin leak measured 65dB,
exonerated. PMI differential (logit minus log-unigram) INERT at
0.25 (identical-30, zero flips) then destructive -- trace verdict:
error is ignorance, not bias; decode-differentials closed,
redirect to content paths).
