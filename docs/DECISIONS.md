# DECISIONS (binding research-program choices, newest last)

Each entry: date, decision, rationale (measurements, not taste),
what reopens it. A decision is not a backlog item: backlog is
unordered work; decisions constrain direction.

## 001 (2026-10-08): training stays out of the program

**Decision:** no gradient-fitted weights anywhere in the construction
or install path. Counts-derived, frozen, and exact-native content
only; alignment growth via training is not pursued.

**Rationale:**
- Skew audit (d16/d32/d64 + counts/fit): max row frozen 6.99,
  frequency order preserved in every variant. Dim and fit move skew
  15.4->9.9 only. Whatever training would do, our available knobs
  provably don't approach it.
- Probe 2 (flat ablation): our glue lives in norms (dies flat);
  Qwen statics: teacher glue lives in alignment (d=3584 trained,
  flat ~2x + fluent). The gap is representational capacity, and
  capacity without training is numerology.
- Dual-head routing achieves every install goal with the prior
  intact (4/4 FLIP, identical split). Nothing on the current goal
  list requires training; adding gradients would trade the
  no-fitting construction principle (docs/LLM_DESIGN.md) for an
  unneeded capability.

**Reopens if:** (a) a goal appears that routing demonstrably cannot
serve (dual-head + negmine + maximin values exhausted with
mechanisms, not tiredness); (b) a gradient-free alignment-growth
method is proposed (counts-derived, frozen -- fitting the *method*,
not the weights, stays in-bounds for discussion).
