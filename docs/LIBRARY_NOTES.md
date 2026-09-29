# Library notes (splats-lite build) — ideas to improve phi-core, banked as found

Numbered, evidence-backed, with disposition. The rule: using the library
effectively means noticing where it pinches and writing it down with a
repro case, not working around it silently.

## #LIB-001: Sobel/gradient frontends repeat per model — candidate helper? [OPEN]

Structure-tensor needs gx, gy. Today: Sobel kernels hand-built in
`chain/splat.py` via `conv_trip` (works, gated). Third model to need
gradients (depth had its own, RIFE has warps, now holo). If model #4 needs
them too, promote a frozen `gradient_kernels()` + integer-compare bucketing
pattern into phi-core (or a documented recipe in IR.md). Evidence: this repo,
docs/SPLAT_OP.md step 1. Cost of waiting: one copy-paste. Cost of promoting
early: wrong abstraction. Threshold: 3 users.

## #LIB-002: `sqrt_trip` sign-forcing contract [DOCUMENTED HERE, propose IR.md note]

`sqrt_trip` forces s=+1, which makes sqrt(discriminant) safe even if lattice
rounding ever pushed a should-be-nonnegative quantity marginally negative in
fixed (the forced sign discards it; magnitude path via exponent halve is
unaffected). Discriminant here is non-negative by construction (sums of
squares via exact `tmul`), so the forcing is belt-and-braces. Proposal: one
line in IR.md under a future `sqrt` opcode row — "sign forced +1; input sign
ignored". No code change.

## #LIB-003: kernel normalization as scale discipline [RECIPE, this repo]

Keeping the tensor domain inside m_cov by normalizing Sobel/8 offline (max
|g|<=0.5, squares<=0.25) instead of adding a third scale. Calibration asserts
tensor maxima < coverage (`chain/calibrate.py`, tensor check). Generalizes to
a recipe: prefer offline kernel normalization over new scales when the dynamic
range need is bounded and known; new scales when it isn't. Proposal: IR.md
"Per-model remainder" calibration bullet gains one sentence.

## #LIB-004: exact-select mux pattern (mirrors `prelu_int`) [RECIPE]

Per-pixel kernel pick via `np.where` chains on integer bucket masks — exact by
construction, no arithmetic. Same shape as `rife_int.prelu_int` (sign-bit
select). Second independent use; if a third appears, consider a shared
`select_mux()` helper with the C lowering beside it. Evidence: `chain/splat.py`
mux + bucket-fraction audit of gate behavior.

## #LIB-005: diagonal orientation + Jxy-sign refinement [CLOSED this pass]

Bank 3->5 (`rotated_kernel` 0/90/±45). Rule: gate shut -> 4; |2Jxy|>|diff|
-> diagonal by sign(sq) (sq<0 -> 2, else 3, sq==0 -> 2); else axis by
sign(diff). Truth tables 1.00 on all four; rotation invariance holds on
smooth edges (gap 0.04); hard rasterized diagonals carry staircase-corner
energy (measured 2x, reported not barred). Sign proof: backslash-edge
normal is (1,-1) so Jxy<0.

## #LIB-006: bank-blur fusion [CLOSED this pass, with correction]

Original pricing ("8x") was wrong: the 8 convs are 2 gradient + 3 tensor-
smooth + 5 bank... i.e. TEN conv-equivalents, and only the 5 bank blurs fuse
(the tensor path must run first: buckets are its output -- chicken-and-egg
for single-pass). Exact two-pass ceiling is 10/6 = 1.67x; measured numpy
1.5x on full frames. Corrected here rather than deleted: mispricings are
evidence too. Fused form: per-pixel gathered bank pass (25 tap-visits/px,
one accumulation), bit-identical by integer commutativity, gated == both
sides (numpy test_splat fusion-exact, C holo_bank_fused + exchange bit).
The C value is absolute time and threading headroom, not the ratio.

## #LIB-010: splat file-exchange is compositional, not fused [CLOSED this pass]

c_chain/splat_exchange.c lowers splat_blur as a composition (8 holo_conv
calls + bridge fixed ops + holo_mux), mirroring numpy op-for-op INCLUDING
roundtrips (binop results re-encoded before compares -- the first draft
compared raw sums and would have diverged). Full-path bit-exact on 4 cases.
Fusion (#LIB-006) stays open as the speed step; correctness is now substrate-
independent.

## #LIB-011: the float trap doesn't strip block comments [NOTED]

`make trap` seds away `//` lines only. A `/* No float in C */` header tripped
it (this pass). Convention from here: say "FPU-free" in block comments, never
the f-word. Upstream suggestion for phi-core's trap: strip `/*...*/` too.

## #LIB-012: loaders must return the whole frozen dict [CLOSED, gated]

`load_ctrl()` initially returned only {iso_atten, coh_thr}; the v2 keys
(mid_atten, coh_hi) stayed in the file while both sides silently ran
different in-code defaults -- and parity STILL passed (weak pixels are rare
enough to hide it). Fix: cache the whole validated dict; gate asserts key
parity between file and cache (`ctrl-cache-parity`). Lesson: parity gates
prove sameness of implementation, not sameness of configuration -- config
plumbing needs its own gate.

## #LIB-007: thresholds want a controller (feeds step 2) [CLOSED by BETA_CTRL]

Frozen constants introduced: coherence gate 0.25, sigmas (1.0 / 1.8 / 0.5),
tensor-smooth sigma. The beta-field controller (docs/BETA_CTRL.md,
chain/control.py, scripts/fit_ctrl.py) now owns iso_atten + coh_thr, fitted
offline on synthetic pairs (iso_atten 0.5->0.25: caution wins on the
flat/noise tradeoff; coh_thr grid-flat, honestly recorded in CTRL.json).
Sigmas remain frozen (v2 owns).

## #LIB-008: gates caught three real bugs this build [CLOSED, evidence]

1. Test double-sqrted the int side + sampled flat pixels (test bug; fixed by
   single-sqrt + transition-band basis). Lesson: parity fixtures must state
   sqrt-count explicitly.
2. Stale oracle line overwrote the tensor-rule bucket with the raw rule
   (agreement 0.44). Lesson: oracle mirrors need diff-review against the spec,
   not just the code.
3. Doubled diagonal factor (`2*|s2|` where s2 already IS 2Jxy; agreement
   0.80, parity 31dB). Lesson: name intermediate quantities by their VALUE
   (`two_jxy`), not their construction.
4. Discriminant sum underflowed m_cov's fixed floor (9e-6 < ~1.6e-5; coh
   0.49 -> 0). Fix: sum exact-product squares at m_acc (floor ~1e-6, cap
   0.26 covers disc<=0.25), rescale_ out. A normalize-first variant lost
   (0.65): sums of exact products beat ratios of rounded quotients.
All four found by the gate suite, none by inspection. The suite earns its keep.

## #LIB-009: `select` helper promotion threshold REACHED [CLOSED this pass]

Exact-select on integer masks used 3x: `prelu_int` (prior art), splat bank
mux, `beta_field`. Promoted: `select_mux()` in chain/holo_phi.py (both call
sites refactored onto it, gates green), `holo_mux` in c_chain/holo_ops.h
(+ shared div/mux vectors in test_holo_c.c / test_core.py c-vectors-*).
Evidence: ALL OK both sides.

