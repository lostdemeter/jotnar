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
3. **Piece word-top1 >= 0.20** (today 0.085) with zero inventions
   held (test_piece_bound stays green). Path: piece-D32 refit
   (baseline 1.048 twin recorded), boundary-aware scoring, bigger
   piece banks with twin constraint.
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
