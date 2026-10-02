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
   full-triple re-cue + family stage (202-bank 195 pinned +
   202/202; w103m5 9528 pinned + 0.9636, family 0.9960 with 43
   family-absent residual priced as next mechanism); v07 both
   sides retrieved (word path; hidden path named limitation);
   gate bars staged hit rates with first-pass numbers kept
   (tests/test_disambig.py 10/10).
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

R1. **Piece word-top1 0.20** (today 0.091 word / 0.058 piece;
routed 0.081 -- first significant aim movement, below). Exhausted
as solo levers: refit, boundary scoring, banks to K256, temps,
context, priors (unbuilt, wrong direction), fusion (both shapes).
Deepened:
all accuracy is ROUTING (30/81 glue vs 0/249 content -- content aim
absent, not weak); generation quality via hybrid+closure+collage
(validity/grounding/flow gated separately under Gate 3). Live
mechanism: ROUTED SPECIALIZATION (flat@starts + sharp@continuations,
twin holds 0.665 + top1 46/566 exact, McNemar p~0.002 --
tests/test_routed.py). Open: distinct-content banks (closed as
class), content-gated sharpening (falsified net-zero), spectral
flattening (gradient confirmed, inverted-U mapped, optimum at
natural spec).
