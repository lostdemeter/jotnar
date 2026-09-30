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

## #LIB-013: offline priors compose at L1 via hashed cache files [RECIPE]

Depth arrives like calibration data: computed once (any framework), cached
under samples/depth/<tag>.npy beside an input-hash sidecar, consumed as
boundary floats. Collision raises instead of reusing. The pattern (compute
once, hash-key, consume hot as integers) generalizes to any future prior
(segmentation, noise maps). L2 (triple-direct, no re-encode) is the upgrade
path with its own parity discipline, not a rewrite.

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

## #LIB-014: continuity is a gateable property, via relu + sigmoid [CLOSED]

v4's thesis: hard selects turn 1-LSB quantization into whole-kernel flips
(28-33dB noise cost, measured). Fix is two continuous primitives, both
integer: `relu_trip` (fixed-domain max, exact) splits orientation weights
that sum-normalize by exact division; `sigmoid_trip` (EXPACT + LUT, mirrors
the rife integer sigmoid, shared vectors both substrates) blends beta levels.
No new fitted params (v3 file reused); the only remaining select is the
true-flat guard, invisible since D~=0 there. Noise parity 40.03dB barred
(was reported). Promotion signal: if a third continuous-blend use appears,
the (relu-split, normalize, weight) stanza becomes a helper like select_mux.

Postscript (v5 round): parity mirrors bugs it cannot see. `_soft_blend`
mapped diagonal weights to the SWAPPED kernels (rD1->outs[2] instead of
outs[3]) in BOTH numpy and oracle -- parity stayed green while rotation
failed 5x (0.0018 vs 0.0090). v4 shipped without rotation coverage on its
own blur path (test_v4 gated hard-mux rotation only). Fixed + covered:
rotation gates now exist on hard (test_ctrl), soft (test_v4, added late),
and v5 (test_v5) paths. Lesson: every blur/mux variant needs its own
rotation gate; cross-check orientation mappings against the bucket rule
(coherence_bucket's diag_pos), never against the mirror.

## #LIB-015: price unreachable bars in gates, not in objectives [DOCTRINE]
A max(0,40-npsnr) veto inside the v5 fit objective was tried and reverted:
nothing on the grid clears 40 (best 37.8 -- adaptive blends disagree with
themselves exactly where content is random: bank outputs spread wide on
white noise, so small weight disagreements cost decibels by gain). An
unreachable veto is pure drag: it elected (0,0,4), killing the coherence
term the rule exists for. Doctrine: verify a veto's reachability (one probe
run) before adding it to an objective. Bars that can't be met live in the
gate suite as measured rows with stated mechanisms (white noise: reported,
share ~11x, distributed) while reachable siblings stay barred (grain sigma
0.02: 47.95dB). The fitter prices tradeoffs; the suite prices honesty.

## #LIB-016: identical columns substitute; the gate must prove it live [CLOSED]

First cross-model substitution proven, not just used: holo `sigmoid_trip`
(numpy) vs RIFE `sigmoid_int` (torch/CUDA) agree bit-exactly on 2007 shared
triples, and a holo chain with its sigmoid swapped for RIFE's produces
bit-identical Y-enh triples (`test_substitute.py`). Two doctrine points:
(1) the chain-swap gate carries a tripwire asserting the substitute was
actually INVOKED (2x) -- a swap gate that passes without exercising the seam
is green wallpaper; (2) the negative control: replicate-pad vs zero-pad convs
MUST differ at borders and agree interiorly, characterizing the divergence
contract. A substitution framework that can't say no is just aliasing. This
is the L3 architectures claim made concrete: same column, same behavior.
## #LIB-017: side-channels carry features, consensus governs them [CLOSED]

First seam carrying features instead of pixels: flow corroborates tensor
orientation. Rules that transferred: boundary conversion (float at seam,
integers downstream -- same shape as depthprior), frozen analytic v1 with
fitting flagged, hashed-cache producer pattern reserved for Phase 2.
New doctrine -- CONSENSUS for directional side-channels: a directional prior
may CONFIRM or VETO a local claim, never originate one (aperture problem is
the general reason; panning-along-edge is the fixture). Corollaries, both
gated: zero-input reproduces blind bit-exactly (static mask takes the direct
path -- roundtrips must never tax pixels that owe them nothing); direction is
assigned before gating (gating first lets corroboration only ever remove).
The pan-e2e ratio gate (0.504 ~= FLOW_ATTEN exactly) is the template for
quantitative modulation gates: don't just assert "changed", assert the
stated factor.
