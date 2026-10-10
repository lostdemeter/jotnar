# Theory of the ball: feasibility via lifting, convergence via locality

The hail-mary pattern, stated once so it stops looking like luck:
whenever a single parametric family cannot satisfy install + holds
(infeasible region: 494/512 targets provably unbeatable by one
vector), we LIFT (add channels/heads/stores/nulls: direct-sum the
parameter space until feasible) and then CONVERGE (mine failures as
nulls until holds). Both halves have closed-form backing below; the
program's history is this loop running, not a string of miracles.

## 1. Objects

Ball B = {(k_i, v_i, s_i, tier_i)}, i = 1..K, with at least one null
(v = 0). Tiers in {exact, assoc, opt, null, base}. Ledger per store.
Apply (single head): C(x) = match(x, K) (dot, unit keys, scale s);
P(x) = softmax(C - max); y = P(x) @ V; out = head(x + y).
Dual-head: out = w(x)*head_flat(x+y) + (1-w(x))*head_skew(x+y) with
w from retrieval mass. All existing ops; no new primitives needed.

## 2. Locality theorem (why negmine converges; analytic, no fit)

Appending null f keyed at x_f multiplies every other store's weight
at x by exactly 1/(1 + exp(s_f(x) - logZ_{-f}(x))), where s_f(x) is
the null's match score and Z the rest-sum. Consequences:
(a) suppression is LOCAL: influence decays exponentially in
(s_max(x) - s_f(x)) -- a null bites hardest near its key and
vanishes elsewhere; (b) installs survive iff
min-distance(neg keys, install rows) >> 1/scale (temperature sets
the radius: higher key-scale sharpens every null); (c) the loop
terminates when the leak set is covered: nulls needed ~ covering
number of leaks at the temperature radius, NOT proportional to
battery size. Whack-a-mole cannot happen by construction: each null
provably perturbs distant installs only exponentially weakly.
Predictions (falsifiable): doubling key-scale halves null count
needed; leaks cluster near install keys (hard cases) or far (need
genuine new nulls -- distinguishes confusion from coverage).

## 3. Lifting (why splits work; Cover's theorem)

Content installation is linear separability (separate target column
from competitor columns along one direction). Cover (1965):
separability becomes likely as dimension lifts. Every green
structure in the logs is a lift: bank over vector (16 -> 16xK),
tiers over head (skewed + flat), channels over match (HN + E),
nulls over gate (K -> K+N rejection regions). Tuning (dose, scale,
temperature) moves WITHIN a family; splits EXPAND the family. The
master rule, formal: when infeasible, lift; when feasible but
leaking, mine nulls (Section 2 guarantees convergence). Never tune
past a proof of infeasibility (full-vocab scan) or a flat response
(ks 8/16/32 identical prior).

## 4. Windows as feasibility intervals

Operating points are intervals, not points: install needs dose in
[D_lo locking margins, D_hi below leak/saturation], per fact, per
layer, per substrate (gains are substrate-local: 170x mirror/geo).
Ladders measure interval endpoints; synthesis computes them;
empty interval = infeasible at this structure (lift, don't push).
Single-point windows (L20@1, NY@0.05) are generic, not edge cases.

## 5. Status of each claim

Analytic (proof, no run needed): 2 (softmax algebra), 4-definitions.
Measured: negmine convergence x4 banks (native 19/26-store,
teacher 6-store, dual-channel), lifting instances (dual-head,
dualbank, entity route, maximin->QP 6.8x), windows (5 ladders +
depth curve + detents).
Open (priced): null-count vs covering number (prediction untested);
ks-halving-nulls (untested); full-volume selectivity (30h,
declined); second-teacher reproduction (yarnball-2's first job).
