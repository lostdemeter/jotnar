# Hopfield capacity math — size the associative memory before building one

## Constitution (FROZEN pre-computation — violations bounce)

Formulas from Ramsauer et al., "Hopfield Networks is All You
Need" (arXiv:2008.02282, ICLR 2021 — via existing knowledge,
nothing fetched; theorem numbers as-recalled and FLAGGED where
uncertain, never faked — see honesty note):
- (R1) Single-update retrieval theorem (§ Retrieval): pattern
  x_i is stored and retrieved with one update if its separation
  Δ_i := x_i^T x_i − max_{j≠i} x_i^T x_j satisfies
  Δ_i ≥ 2/(βN) + (1/β)·ln(2(N−1)NβM²), with retrieval error
  exponentially small in βΔ_i. Frozen with β=1 (paper default;
  β-sensitivity noted as gap, not swept).
- (R2) Storage-capacity theorem (§ Storage Capacity): N random
  patterns on the sphere of radius M=K√(d−1) are stored with
  probability ≥ 1−p for N ≥ √p·c^((d−1)/4), where with
  a := (2/(d−1))·(1 + ln(2βK²p(d−1))), b := 2K²β/5:
  c := b / W_0(exp(a + ln b)) (W_0 = principal Lambert W).
  Frozen with unit-norm patterns (M=1 ⇒ K=1/√(d−1)) and
  failure prob p=0.001.
- (R0, context only) Classical Hopfield C ≈ 0.138N
  (Amit–Gutfreund–Sompolinsky 1985) — the comparison baseline,
  labeled as such, not a Ramsauer claim.
Honesty note (part of the freeze): equation-level pinning is
UNVERIFIED against the text — the load-bearing claims of this
round (exponential-in-d shape, c>1, modern-beats-classical
crossover, verdicts) depend on the bound's SHAPE, the paper's
least disputable headline, not on constant-exactness. A
±20% constant-robustness check rides in analysis (verdicts must
survive it). Lambert W implemented by hand (Newton iteration —
scipy NOT installed here and installing is forbidden);
checked against W(e)=1, W(1)=Ω, W(0)=0.
Pattern count frozen: K = 20 (5 observed attractor basins ×
safety factor 4 — covers basin splitting under new data/seeds;
verdicts verified log-insensitive to K=10/40 in analysis).
Error tolerance frozen: successful retrieval = error within
strata-bar scale (AbsRel ≤ 0.1). Bridge (APPROXIMATE, stated):
for unit patterns, dot-separation Δ ↔ L2 ball; the tolerance
enters via (R1): required separation Δ_req(N=K) is computed
and held against the 0.1-scale tolerance for order-of-magnitude
consistency — no precision claimed past one digit.
Capacity table frozen: dims {64, 128, 256, 512, 1024} × max
patterns = min(N_R(d) [exponential bound], N_S(d) [separation-
limited from (R1) + random-pattern geometry: dot∼N(0,1/d),
union bound over pairs at Δ_req, prob ≥ 1−p]). Tighter governs.
Verdict rule frozen: capacity C(d) ≥ 4K=80 → SUFFICIENT
(current failures are then dynamics, not capacity — stated as
such); 20 ≤ C < 80 → MARGINAL (named, not rounded up);
C < 20 → INSUFFICIENT (explains Phase-8/10 flatness
structurally — stated as such). The 64-dim question
(capacity-starved all along?) is answered by the d=64 row.

## Results (TBD — computation lands here)
