# Beta-field controller (step 2 of the intelligence sequence)

The enhancement stops wanting one thing everywhere. A per-pixel effective
beta with stated geometric reasons, fitted offline on synthetic
degradations, frozen like scales, executed as integers.

## Rule (analytic v1, everything else is fitting)

```
beta_eff = beta * (bucket == 4 ? iso_atten : 1.0)
```

- Oriented pixels (buckets 0..3: a detected edge with known direction,
  axis or diagonal since v1.1): full beta. Boosting along a known edge is
  safe by construction of the splat bank (smoothing never crosses it).
- Isotropic fallback (bucket 4: flat or uncertain): beta scaled
  by `iso_atten`. Flats have D~=0 so the scale is moot there; the live case
  is uncertain structure, where reduced boost is caution with a stated reason.
- `iso_atten` and the coherence threshold `coh_thr` are the controller's two
  parameters. Everything else stays frozen.

## Fitting (offline, calibration doctrine)

`scripts/fit_ctrl.py`: grid search over (iso_atten, coh_thr) on synthetic
pairs (bars, texture, diagonals, flats × clean/blurred+noisy), maximizing
sharpness-gain minus overshoot-penalty minus flat-response. Best params
frozen to `chain/CTRL.json`. Runtime never fits. The shipped default
(iso_atten=0.5, coh_thr=0.25) is the analytic init point; the fitter must
beat it to rewrite the file, and every rewrite re-runs the gates.

## Execution (integer)

`beta_field()`: exact select between the beta triple and the attenuated
beta triple on the integer bucket mask -- same shape as `prelu_int`
(#LIB-004, third use; helper promotion threshold reached, see #LIB-009).
Then `tmul(D, beta_eff_field)` as before. Oracle mirrors in float.

## Gates

- parity with controller on (>=40dB, both settings);
- rotation invariance on smooth edges (v1.1: the diagonal fallback retired,
  so the signature evolved from bar>diag to bar~=diag);
- `CTRL.json` determinism (same pairs -> same file);
- flat identity preserved.

## v2: coherence-modulated atten [BUILT, then extended]

Oriented pixels split by coherence magnitude (strong-coh full beta, weak-coh
mid beta) + iso atten: params (iso_atten, mid_atten, coh_hi), fitted
(iso 0.5, mid 0.4, hi 0.4) on pairs extended with a corner fixture (quadrant
junction: weak coherence + large detail -- the pixels mid_atten exists for;
without it the fitter couldn't see v2). coh_thr stays at its v1 fit.

## v3: fitted strong level [BUILT]

The decision table is complete: (iso_atten, mid_atten, strong_atten, coh_hi)
= (0.5, 0.4, 1.0, 0.4). The fit verified strong=1.0 rather than changing it
(0.75 loses sharpness with no overshoot gain to compensate) -- a fitted
confirmation of the analytic default, now stated in the file instead of
assumed in code. Schema completion writes (missing key on tie) are
content-neutral but required: the file must state every param the runtime
consumes (#LIB-012). Rotation symmetry kept throughout (no per-direction
params). Read the rule as a 2-layer decision net: threshold routing
(coh_thr, coh_hi) + 3-entry table, executed as integer selects.

## v4: continuous fields (sigmoid beta + relu blur) [BUILT this round]

v3's hard selects cost 28-33dB on uniform noise: a 1-LSB coherence wobble
flips a whole kernel at large local cost. v4 removes every discrete decision:
beta blends the three fitted levels by coherence through integer sigmoids
(`beta_field_soft`, k=30, LUT-based, file levels reused unchanged), and the
bank blends all four oriented outputs by relu-split tensor magnitudes
(`_soft_blend`, weights r/sum, true-flat guard only). Both paths reuse the
v3 fit -- no new params, no refit. Noise parity is BARRED (>=40dB, measured
40.03dB vs 32.76 pre-gate); real 50dB, structured 53dB. Showcase: v4 sharpens
harder than v3-hard (1234 vs 1092) while staying continuous -- partial weights
everywhere beat hard fallback to iso. The v3 suites keep gating the hard path.

## Backlog (learned v5) -- BUILT below

## v5: learned detail gate (the tiny net) [BUILT this round]

Rule: `beff = beff_v4 * scale`, `scale = 0.5 + sigmoid(w0 + w1*coh + w2*dhat)`,
`dhat = clip(|D|*4, 0, 1)`. One 1x1 layer on two rotation-invariant features
(coherence magnitude, detail magnitude -- no orientation enters, so rotation
symmetry holds by construction) plus sigmoid, all existing IR ops
(tmul/binop/sigmoid/clip/abs: composition only, no new C needed). Zero
weights give scale~=1 (sigmoid(0)=0.50023 to LUT precision), so the base IS
v4: the 27-fit grid ((w0,w1,w2) over [-2..0]x[0,2,4]x[4,6,8]) had to beat v4
to rewrite. It did: (0, 4, 8) at 4.992 vs 4.588 (+9%). w1 saturates from 4.0
on (identical scores at 6/8: sigmoid-saturated interior optimum); w2 climbs
to the grid edge but flattening (+0.05/+0.03: sigmoid asymptote bounds
further gains, stated not chased).

Noise doctrine is two-tier (#LIB-015): realistic grain (sigma 0.02, the
pairs' own degradation level) is BARRED parity (v5: 47.95dB); adversarial
white noise at full amplitude is MEASURED (37.10dB, distributed
gain-on-spread, share ~11x -- same ladder as v3/v4, no flips). A veto term
inside the fit objective was tried and reverted: nothing clears 40, so an
unreachable veto is pure drag that elected (0,0,4) and killed the coherence
term. Verify veto reachability before adding it.

Gates (`test_v5.py`): file validity + grid + beat + cache-parity + hash,
grain/real/structured barred, flat, rotation re-run (gap 0.04 -- the swap
fix restored it), detail-ordering (corner >> flat: the signature w2 exists
for). Showcase: v5 sharpens harder than v4 (1294 vs 1108) with less mean
change than iso (1.36 vs 1.83 LSB).

## Backlog (learned v6)

Per-pixel predicted sigmas / gain-clip range from a tiny geometric net
(1x1/3x3 convs, existing IR ops) on the same pairs/score/freeze/gates.
First candidates: fitted sigmoid temperature (today folded into weights),
trace-magnitude feature, or the corner-pair weight in the objective.
