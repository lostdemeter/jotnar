# `splat_blur` opcode spec (splats-lite, step 1 of the intelligence sequence)

Single fused op replacing the blind isotropic Gaussian in the holo chain.
v1.1 bank is 5-way (V/H/diag\\/diag//iso); diagonals resolve by Jxy sign
(#LIB-005 closed). Lowered compositionally AND fused on both substrates
(#LIB-006 closed, corrected pricing below).

## Signature

```
splat_blur: trip(H,W) x m_acc x m_cov -> trip(H,W) + diag
```

`diag` (Python-side only, never in IR): bucket map {0:V,1:H,2:\\,3:/,4:iso},
gate/diag fractions, coherence stats for audits.

## Contract (each sub-step names its existing op)

1. **Gradients** (`conv` ×2): Sobel-X/8, Sobel-Y/8 on amplitude A.
   kernels normalized offline so |g| <= 0.5 on [0,1] input -- tensor domain
   stays inside m_cov coverage (calibration asserts this; see #LIB-003).
2. **Tensor** (`tmul` ×3): Jxx=gx*gx, Jyy=gy*gy, Jxy=gx*gy. Exact.
3. **Tensor smooth** (`conv` ×3, small Gaussian): spatial regularization.
4. **Discriminant** (`tmul`+`binop`): disc=(Jxx-Jyy)^2+4*Jxy^2, all triples.
   Non-negative by construction; `sqrt_trip` forces s=+1 regardless (#LIB-002).
5. **Coherence** (`sqrt`+`tdiv_trip`+`clip`): aniso=sqrt(disc),
   trace=Jxx+Jyy, coh=aniso/(trace+eps) clipped [0,1]. Zero-guarded by tdiv.
6. **Orientation bucket** from the SMOOTHED tensor (never raw gradients:
   smoothing moves energy to edge flanks where raw direction is zero --
   measured 50% mis-bucketing with the raw rule). Integer compares in fixed
   domain, no atan, no float. v1 is 3-way (diagonals fall back to iso,
   which is the safe choice; v1.1 refines per #LIB-005):
   - gate shut, or |2Jxy| > |Jxx-Jyy| -> 2 (isotropic: flat or diagonal)
   - Jxx-Jyy > 0  -> 0 (V kernel, elongated along y: gradient across x)
   - else         -> 1 (H kernel, elongated along x)
   Implementation note: `s2 = Jxy+Jxy` already IS 2Jxy -- compare |s2| vs
   |diff| directly. Doubling again halves the oriented set (shipped once,
   gates caught it: agreement 0.80 vs 0.99).
7. **Coherence gate** (fixed compare): coh >= 0.25 -> oriented kernel,
   else isotropic (bucket 2). Threshold is a frozen constant flagged for
   the beta-controller (step 2 owns all thresholds eventually).
8. **Bank blur** (`conv` x5, v1.1): V/H/diag\\/diag// anisotropic
   (sigma_long=1.8 along edge, sigma_short=0.5 across, 5x5, offline-built
   frozen formulas: `rotated_kernel`). Each accumulates @ m_acc, rescales
   to m_cov -- same contract as `conv_trip`.
9. **Mux** (exact select, `select_mux` #LIB-009): per-pixel pick of the 5
   outputs by bucket index. v4 alternative: `_soft_blend` (relu-weighted
   average, continuous; diagonal mapping rD1->outs[3], rD2->outs[2] per the
   bucket rule -- a swap shipped once and parity stayed green while rotation
   failed, see #LIB-014 postscript).

## Edge semantics

Replicate (clamped indices), matching `conv_trip` / `holo_conv`. NOT zero-pad.

## Schedule (IR.md: recorded, currently value-identical under reorder)

oy-ox-ky-kx per blur; five blurs sequential; mux last. Fusion (single pass
with per-pixel kernel gather) is the C-lowering backlog -- parity gate travels.

## Gates

- orientation truth table on synthetic bars (H/V/diag) -- bucket accuracy;
- flat -> isotropic fallback (gate fraction ~0 on flat);
- halo metric: step-edge overshoot splat < iso;
- int-vs-float-oracle parity >=40dB on the blurred amplitude;
- file-exchange C parity when the fused lowering lands (backlog).

## Cost (honest)

v1 runs 3 bank convs + 3 tensor-smooth convs + 2 gradient convs = ~8x the
isotropic blur in the numpy reference. Fusion backlog itemizes the recovery
(single gathered pass). The intelligence-sequence judgment: correctness of
selection first, speed second, gates traveling throughout.

## Backlog (v1.2+)

- Fused single-pass C lowering with per-pixel kernel gather + file-exchange
  parity (same pattern as `holo_conv`). -- CLOSED this pass: `holo_bank_fused`
  + exchange flag, bit-exact both modes. What remains is threading/OpenMP.
- Coherence threshold + sigmas owned by the beta-controller (step 2).
  --iso_atten/coh_thr owned since step 2 shipped; sigmas stay frozen (v2?).
