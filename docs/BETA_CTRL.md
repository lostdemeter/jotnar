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

## v3: fitted strong level [BUILT this round]

The decision table is complete: (iso_atten, mid_atten, strong_atten, coh_hi)
= (0.5, 0.4, 1.0, 0.4). The fit verified strong=1.0 rather than changing it
(0.75 loses sharpness with no overshoot gain to compensate) -- a fitted
confirmation of the analytic default, now stated in the file instead of
assumed in code. Schema completion writes (missing key on tie) are
content-neutral but required: the file must state every param the runtime
consumes (#LIB-012). Rotation symmetry kept throughout (no per-direction
params). Read the rule as a 2-layer decision net: threshold routing
(coh_thr, coh_hi) + 3-entry table, executed as integer selects.

## Backlog (learned v3)

Per-pixel predicted fields (sigma bank weights, gain-clip range) from a tiny
geometric net (1x1/3x3 convs, existing IR ops) trained on the same pairs.
The v1/v2 scaffolding (pairs, score, freeze, gates) is reused unchanged; only
the rule body grows.
