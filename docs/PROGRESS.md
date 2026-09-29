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

## v2 (this round)

Coherence-modulated atten: oriented pixels split by coherence magnitude
(strong-coh full beta, weak-coh mid beta) + iso atten. Params
(iso_atten, mid_atten, coh_hi), 27-fit grid on the existing pairs,
frozen to CTRL.json. See docs/BETA_CTRL.md.

## Library ideas banked (#LIB-001..011, docs/LIBRARY_NOTES.md)

Promoted: `select_mux`, `kernel_triples`. Open: gradient-frontend helper
(2 users, threshold 3), diagonal v1.1 (closed), fusion (closed, corrected),
`select` promotion (closed), trap comment-stripping (upstream suggestion).
Closed entries keep their evidence; open ones name their trigger.

## What's next

Learned-v3 (tiny geometric net on the same pairs/score/freeze/gates),
threading/OpenMP headroom in C, step 3 (depth + temporal -- repos checked
out locally, seam contracts in docs/COMPOSE.md).
