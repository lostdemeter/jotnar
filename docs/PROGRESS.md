# Progress: from ratio hack to geometric instrument

## Thesis (what we're stressing)

The tool stays **geometric**: every operation is an explicit integer op on
lattice values (XOR+ADD, exponent halve, fixed accumulate, gathers) or a
frozen LUT built offline. Nothing statistical hides inside the runtime.
"Intelligence" enters only as *steering* -- which kernel, how much beta --
with stated geometric reasons, fitted offline on synthetic pairs, frozen
like scales, executed as integers. As we build, reusable patterns get
promoted into phi-core helpers (#LIB-009 `select_mux` already made it:
third use triggered promotion, numpy + C + shared vectors). The bet is
explicit: the more we use the library, the more the framework generates and
the less we hand-write. Reuse count is the promotion rule, not taste.

## Where we started

`holographic_enhancement` claimed `A=sqrt(I)` in THEORY.md and never took a
square root: ratio unsharp-mask in LAB + gamma, float everywhere, parabola +
epsilon + clamps as tuning constants. Not holographic, not geometric.

## What we built (with numbers)

1. **True amplitude** (`chain/holo_phi.py`): `A=sqrt(Y)` via exponent halve
   (maxerr 6.9e-04, gated), blur in amplitude domain, `tmul` square back.
   Parity 64-68dB vs same-math oracle (bar 40).
2. **Two scales** (`m_acc` products + `m_cov` values, `rescale_` the only
   changer). The gate caught a real saturation bug at 7.9dB; fixed to 68dB.
3. **C lowerings, bit-exact**: sqrt/mul/div/mux shared tables, `holo_conv`
   + full `splat_blur` file-exchange (compositional AND fused) -- exact
   equality, not dB. `make trap` clean (tripped once on the word "float"
   in a block comment; now convention: say FPU-free).
4. **Splats-lite** (`chain/splat.py`, `docs/SPLAT_OP.md`): structure tensor,
   coherence, 5-way oriented bank (truth tables 1.00 all orientations),
   rotation invariance gap 0.04 on smooth edges. Four bugs found by gates,
   none by inspection (underflow at m_cov floor, doubled diagonal factor,
   stale oracle line, double-sqrt fixture).
5. **Beta-field controller v1** (`chain/control.py`, fitted): 2 params,
   `iso_atten` 0.5->0.25 (caution wins the flat/noise tradeoff), `coh_thr`
   grid-flat (recorded honestly). Signature: oriented bar > diagonal
   fallback; retired honestly when v1.1 resolved diagonals.
6. **Fusion** (`#LIB-006`, corrected pricing): exact two-pass ceiling is
   10/6=1.67x, measured numpy 1.5x; the original "8x" miscounted the tensor
   path. Corrected in the notes, not deleted. Bit-identical both substrates.
7. **Showcase** (`showcase.py`): every mode + internals (A, As, D, buckets,
   coherence) as labeled sheets. Intelligence axis reads 1.83/1.40/0.95 LSB
   mean change: each step restrains itself more.

## v2/v3 (built): coherence-modulated decision table

Oriented pixels split by coherence magnitude + iso atten. Params
(iso 0.5, mid 0.4, strong 1.0, hi 0.4) fitted on pairs extended with a
corner fixture (without it the fitter couldn't see mid); the strong=1.0 fit
verified the analytic default rather than changing it. Whole-dict loader fix
+ cache-parity gate (#LIB-012: parity proves sameness of implementation, not
of configuration). See docs/BETA_CTRL.md.

## v4 (built): continuous fields kill the noise penalty

v3's hard selects cost 28-33dB on uniform noise (1-LSB wobble flips a whole
kernel). v4 blends instead: sigmoid-weighted beta levels + relu-weighted
bank outputs, no discrete decision anywhere, no new fitted params (v3 file
reused). Primitives are integer (`relu_trip` exact, `sigmoid_trip` LUT,
shared C vectors). Noise parity 40.03dB BARRED (was reported); real 50dB,
structured 53dB. A diagonal-weight swap shipped in `_soft_blend` mirrored in
both sides -- parity green, rotation 5x broken -- caught by the rotation
gate and fixed; rotation coverage now exists on all three blur paths.
#LIB-014 banked.

## v5 (this round): the tiny net

`beff = beff_v4 * (0.5 + sigmoid(w0 + w1*coh + w2*dhat))`: one 1x1 layer on
two rotation-invariant features, all existing IR ops (composition only, no
new C). Zeros ~= v4 (sigmoid(0)=0.50023 LUT), so the 27-fit grid had to beat
v4: (0, 4, 8) at 4.992 vs 4.588 (+9%). w1 plateaus from 4.0 (interior
optimum); w2 climbs to the edge but flattening (asymptote-bounded, stated).
Two-tier noise doctrine (#LIB-015): grain barred (47.95dB), white noise
measured (37.10dB, distributed); an unreachable veto in the objective was
tried, caught distorting, and reverted. Showcase: v5 1294 sharp at 1.36 LSB
-- harder than v4 (1108) with less change than iso (1.83).

## Step 3a (built): depth composition, L1

DAV2 relative depth as an offline prior (`chain/depthprior.py`, cached with
input-hash sidecar), median-split near/far modulating beta multiplicatively
(`--depth TAG`). Hot path stays integer. Gates (`test_depth.py`): cache
determinism, parity 45.68dB, near p99 0.063 > far 0.017 on f_012. L2
triple-direct handoff stays backlog; temporal IIR stays spec
(docs/COMPOSE.md).

## Library ideas banked (#LIB-001..015, docs/LIBRARY_NOTES.md)

Promoted: `select_mux`, `kernel_triples`. Closed this round: diagonals,
fusion (corrected), v4 continuity primitives, mirrored-bug rotation coverage,
unreachable-veto doctrine. Open: gradient-frontend helper
(2 users, threshold 3), trap comment-stripping (upstream suggestion),
continuous-blend stanza promotion (1 use, threshold 3 per #LIB-014),
soft-blend + abs/neg C lowerings (compositions only so far).
Closed entries keep their evidence; open ones name their trigger.

## What's next

Learned-v6 (sigmoid temperature, trace feature, corner-pair weight),
threading/OpenMP headroom in C, step 3 rest (temporal IIR + L2 depth).
