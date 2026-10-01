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
## #LIB-018: feedback needs the seam convention checked, not assumed [CLOSED]

Temporal warp port found a live convention split by reading, not by failing:
our seam is (dy,dx) while rife's warp_fixed reads [...,0] as X. Warp-parity
(73dB) confirms the mapping; a transposed mapping scored 11dB (measured
during development -- wrong-index bugs fail LOUD in warp, which is a mercy).
Lesson: when reusing a foreign op, gate the index convention explicitly
(subpixel random flow + parity does it); "same formula" is not "same mapping".
Refactor note: the still path split into luminance_detail/finish with 0-diff
proven by the existing parity suites (no mirror to drift, cf #LIB-014) --
the temporal path reuses both with mixed D. Single source, two callers.
## #LIB-019: meta-selection + boundary-decision pattern [CLOSED]

The router (chain/router.py) is the first structure selecting AMONG
configurations (still/temporal per frame) instead of pixels. Two patterns:
(1) decide in float at the seam, consume the verdict as an exact flag --
third use after the motion static mask and the depth median split (helper
promotion threshold REACHED for a `decide`-family helper: exact-flag
decisions from boundary floats; next lowering pass should promote it);
(2) gate meta-claims component-wise (flicker-wins AND sharpness-bounded, not
one blended score -- a single number would hide the tradeoff the router
exists to navigate). Switches are seamless by construction (still ignores
state; state always refreshes so memory never lies -- the store uses the
SAME detail the output used, threaded explicitly, gated by seamless-static).
Per-tile routing is the stated backlog; the pattern transfers unchanged.
## #LIB-020: prior modulation, two forms one family [CLOSED]

depth_mult (bool-mask select) and flow_scale (continuous affine blend) are
one structure: multiplicative caution from offline priors, exact identity
on the unaffected subset (ones triples: tmul identity is exp-add exact).
Promoted to chain/prior.py with select() + affine() sharing the contract
docstring; both call sites refactored 0-diff (depth/motion/parity suites
green). No new C (composition of already-lowered ops -- stated, not missing).
Family splits are normal (cf replicate/zero): the split is gated, not hidden.
phi-core staging follows the decide precedent (branch, not main).
## #LIB-021: paper-first falsification + parity-basis doctrine [CLOSED]

Stage 2 was designed as a denoiser on paper, ran, and measured WORSE than
input on flats (25.1 vs 26.3dB): A + beta*D keeps the noisy base, so NO boost
architecture can denoise (output-domain averaging or As-output would be
needed -- backlog, explicitly unclaimed). Renamed denoise->stabilize; the
surviving true claim (flicker -26% on static runs) gates green. Falsification
at full resolution is the paper-first discipline working -- report, don't tune.
Companion finding: parity belongs at the chain boundary (linear, like every
other gate); comparing sRGB bytes holds the chain accountable for gamma's
shadow expansion (0.002 linear reads 25 LSB in deep shadow). Perceptual
claims (flicker) stay in sRGB. Machine-metrics vs experience-metrics, stated.
## #LIB-022: static verification splits three ways (layouts now, ranges now, verifier later)

verify() (declared-layout replay, zero execution) + ranges.estimate() (hull
intervals vs M.json coverage) land the first two thirds of the static
verifier; full parse-time verification (scales per-op, dynamic geometry)
stays backlog. Lessons, all measured: (1) ASYMMETRIC doctrine -- saturation
flags on any exceedance, underflow only whole-range-below (tiny values are
often legitimate ~0; the flagship tripped the first draft of the symmetric
rule and was right); (2) coverage binds LATTICE values only -- checking U8
bytes against m_cov cap false-positived on SRGB output (fixed by scoping
checks to T:* layouts); (3) hulls cannot see precision loss inside spanning
ranges (tensor_disc pins the boundary -- the estimator's stated limit, not a
failure); (4) estimator rules must mirror op semantics exactly (SQUARE rule
forgot the op's own clip -- caught by its own flagship-clean gate).
## #LIB-023: ITERATE v1 = append-only growth (KV pattern); WHILE deferred [CLOSED]

CONCAT exact move (triples plane-wise + plain arrays; non-axis mismatch fails
loud) + repeat(grow=[...]) with append-only-along-axis-0 discipline, growth
logged per iteration, off-axis/shrink refused. Demo gate: 4-row KV cache
contents exact vs manual concat + growth log [(1,6)..(4,6)]. WHILE
(data-dependent termination) explicitly deferred with reason: termination
semantics + verdict integration need a design conversation, and every current
iteration need (diffusion N-steps, IIR warmup, cache fill) is bounded --
no demand, no build (the process working: stated, not missing).
## #LIB-024: trace mode (debugger) + what it can/can't catch [CLOSED]

run(trace=[...]) records per-op {line, op, in/out summaries, sec} without
perturbing values (bit-identical, gated); format_trace() renders the
instrument panel (SPLAT_BLUR dominates at 0.2s/0.26s -- matches fusion
pricing independently). Summaries are best-effort and never fail the run.
Honest limit, stated in the gate: summaries show SYMPTOMS (shapes, layouts,
means per line catch the broadcast/transpose classes in minutes); gates
prove CAUSES. Profiler falls out free (per-op wall time). Test-variable
hygiene note, learned the hard way twice this session: loop variables
(rgb) leaking across gate sections cause phantom shape mismatches -- fresh
names per section (ftext/frgb pattern).
## #LIB-025: debugging multilingual failures (two bugs, one typo) [CLOSED]

The procedure round produced three failures in one session, each a different
species: (1) a missing "w" open-mode typo that mimicked filesystem/host
failure across five red herrings (strace showed O_RDONLY -- READ the
syscall, not the traceback line); lesson: when adjacent identical syscalls
behave differently, diff the CALLS (flags/modes), not the environment.
(2) head-split assumed HEAD( with no space; CALL name(args) has one --
fixed by matching the CALL form FIRST with an explicit regex. (3) real
semantic gap: `#` namespace separator collided with comment syntax
(generated names truncated on re-parse) -- fixed by tracking stripped-ness
through expansion (flat triples carry the flag). Plus the found-then-earned
one: float arrays reaching triple-ops computed garbage silently -- now
refused twice (layout-kind cross-check centrally + _need_triples on the
arithmetic core), and the procedure tests run on triples. Debugging
multilingual stacks (text->expansion->parse->values): instrument the seam,
not the symptom.
## #LIB-026: shape rules (strictness as doctrine, positives prove it) [CLOSED]

Stream-geometry checks per mnemonic at run time (shapes are KNOWN there --
no annotations needed): elementwise identical shapes, matmul inner dims
(with the TRANSPOSE hint: most common cause, measured), warp spatial match,
argmax axis range, gather id bounds, SELECT branch+mask agreement. Strict on
purpose: silent numpy broadcasting hid real bugs. The positives matter more
than the negatives: every existing suite passes WITH checks active, proving
no legitimate broadcasting broke. Doctrine: strictness must be EARNED by a
green full-suite, never assumed safe.
## #LIB-027: composition contracts (DEF formals checked at CALL) [CLOSED]

Geometry discipline, second half: DEF formals may carry AS layouts; CALL
sites check actuals' DECLARED layouts (both known + concrete + unequal fails
naming CALL site AND DEF origin); UNKNOWN either side defers to
post-expansion verify + runtime (gradual across files, stated). $VAR
formals refused (no bindings at contract time). verify() reports the DEF
interface table (composition contracts visible in one place). IN pre-scan
supplies caller layouts (IMPORTed INs stay UNKNOWN -- documented limit, not
a hole: post-expansion machinery covers them). First use found nothing to
fix (all suites green pre-existing) -- discipline that changes nothing
existing is safe discipline.

## #LIB-028: extension drill (GELU, v1.0 Gate 5) [CLOSED]

First stranger-supplied structure (phi-core ASM_HANDOFF.md): GELU exact-
form via `gelu_erf_int` (EXPACT + PHI LUT, any input range). Thin wrapper,
0-diff; 4 gates (0-diff, 60dB vs torch, edge shapes, in-listing exact).
Drill wall time ~3 min, logged in docs/VELOCITY.md (the log IS the v1.0
extensibility answer). Lessons: (1) drill time measures handoff quality
as much as our velocity (sig + gate pattern + rejected tanh-alternative
supplied -- zero design decisions our side); (2) LUT-with-exact-asymptotes
is the cheap mnemonic kind (no scale contract, unlike SOFTMAX); (3) parity
numbers are fixture-dependent -- report fixture with number; (4) shoulder
imprecision documented-not-gated (x=10 -> 9.99; exact only beyond +/-16).
Next candidates from the same handoff: SCAN (fills ITERATE gap #4),
GRN joint exercise (reference on request).
## #LIB-029: stdlib search order (stdlib-first, explicit-paths-exempt) [CLOSED]

v1.0 Gates 3+4: `stdlib/` seed (attention.asm + mlp.asm, factored out of
xf_block, dogfood-proven bit-exact) + IMPORT search order in
`_resolve_import` (new kwargs on expand/parse/assemble/verify/run_text/
repeat, all defaulted — backward compatible, every suite green unchanged).
Doctrines: (1) bare names search stdlib FIRST so shared listings are
addressable by name from anywhere (a basedir shadow LOSES, gated —
name local files distinctly); (2) explicit paths (`./`, `sub/`) bypass
stdlib entirely (no shadowing surprises); (3) refactor-onto-stdlib must
be 0-diff by gate (`asm-dogfood-xf` embeds the pre-refactor body —
comparing against HEAD would go vacuous the commit it lands).
## #LIB-030: scale-context prototype (CONFIG override on MATMUL) [SPIKE]

v1.1 spine item 1, first cut: `_scales(config)` + MATMUL-only override
(other ops defaulted — blast radius one line). Measured: U(-2,2) matmul
rails at frozen m_acc (-10.4dB, max|q|/2^18 == 1.0 = the auditable rail)
and holds 55.6dB at m_of(8)=35492 (66dB swing = the per-block win).
Gates: explicit-frozen == default exact + override ≥40dB with default
<40dB on the same fixture (non-vacuous) + in-listing CONFIG exact + 3
bad-scale refusals. Structural finding: RESCALE alone cannot give
per-block regimes (downstream ops re-bridge at global m) — scale context
must thread into execution; DEF-header `@scale` is the end-state.
Next: BATCH_MATMUL + arithmetic core follow the same line, then a
two-regime listing with RESCALE at the seam (roadmap gate 1).
## #LIB-031: two-regime listing (roadmap gate 1 CLOSED) [CLOSED]

MATMUL-family + ADD/SUB threaded (same one-line pattern; RMSNORM/SOFTMAX/
SILU-family need none — normalizing, BIAS-structural, scaleless, each
stated). Demo (test_xf_block.py): attention-small (scores 0.39,
frozen-fine) + MLP-big (products over frozen m_acc) in ONE listing —
 frozen 24.0dB, CONFIG m_acc 35492 recovers 65.1dB (41dB swing, bar 40).
 Honest scopes, all stated: (1) the sweep proved single-scale-up CANNOT
 separate the walls (matmul and softmax saturation arrive together,
 both ~w²) — the split works because attention is CONTRACT-bound
 (scores ≤1.0, no m fixes it) while MLP is COVERAGE-bound (bigger m
 fixes it); (2) big-m doesn't hurt small values here (4.5e-4 both —
 error is lattice-encode-dominated), so one CONFIG key serves both
 blocks; per-DEF `@scale` stays the end-state for the drowning case
 (phi-core GRN evidence); (3) two syntax slips caught by the suites
 (dropped `"""`, joined lines) — the gates earn their keep again.

## #LIB-032: matmul C lowering (roadmap gate 2) [CLOSED]

 `holo_matmul` (+ exchange driver, Makefile target, test_matmul_c.py):
 per-product tmul + to_fixed@m_acc inline (holo_conv form), int64 acc
 with 2^62 fail-loud (mirrors _assert_bound), from_fixed via gated
 bridge.c, B-batch-1 broadcast, m_acc as PARAMETER (v1.1 scale path in
 C too). 5/5 file-exchange cases bit-exact first run (small, batched,
 broadcast, big-m 35492 on ±4 values, zeros edge). Row chunking
 (n_chunk=32) deliberately NOT mirrored (tiling only, same sums).
 Stated divergence (holo_conv precedent): from_fixed extremes —
 numpy asserts, C clamps; fixtures stay in-range.
 ## #LIB-034: matmul CUDA kernel, 3-way parity (v1.2 gate 1) [CLOSED]

 `k_matmul` (c_chain/matmul_cuda.cu): one thread per output, k loop with
 the exact holo_matmul product law, int64 register acc with device assert
 at 2^62, from_fixed on device via cuda_dev.cuh mirrors, LUTs as int
 globals (conv.cu precedent), m_acc parameterized. 5/5 exchange cases
 bit-exact vs numpy FIRST RUN (small/batched/broadcast/big-m/zeros) —
 transitively C == CUDA. Reduction-order risk from the roadmap CANNOT
 fire here: single-thread accumulate per output means no cross-thread
 reduction exists (stronger than commuted). Notes: (1) same in.bin format
 as the C driver — one harness gates both sides, no format fork;
 (2) `make cuda` is NOT in default check (GPU-less boxes stay green;
 missing nvcc SKIPs like the other needs-hardware gates);
 (3) `-arch=native` (build+run same box; comment records the cross-build
 override); (4) trap now scans *.cu too (FPU-free held, incl. the word
 "float" ban in kernel comments).
## #LIB-035: CUDA units vs C twins (v1.2 gate 2a) [CLOSED]

`k_conv_rep` + `k_mux` (c_chain/holo_units.cu) mirror holo_conv.c /
holo_mux line-for-line; the C layer is the reference for the CUDA port.
6/6 bit-exact (conv fig/flat/corner/noise vs numpy ref; mux vs promoted
select_mux in-range + documented clamp contract out-of-range).
Two slips, both mine: (1) nvcc is C++ — void* needs casts (xmalloc macro);
(2) writer interleaved per-stream planes while the driver reads planar
(conv passed, mux failed — format bugs fail LOUD, not silent).
Doctrine refined: one harness format per driver, writer and reader
reviewed as a pair; transitivity (CUDA==C via shared numpy ref) beats
 direct C-vs-CUDA byte compares. Wired into test_c.py (c-cuda-units).
 ## #LIB-037: flagship on CUDA, bit-exact (v1.2 gate 2 FULL) [CLOSED]

 flagship_cuda_exchange: host float edges (decode/luma/encode-A in,
 gain/chroma out — boundary by doctrine) + ~50 device launches covering
 splat_soft+v5 A->YENH (tensor, coh+buckets, soft blend, v5 gate, finish;
 composition on the host side, no new math). 3/3 rows BIT-EXACT vs
 enhance_luminance_int (synth 16×16, real 288×352 frame, big-m plumbing).
 Composition of unit-gated kernels + driver composition with no new math
 yields bit-exactness for free — the strongest form of gate 2 (bar was
 40dB). Supporting kernels (neg/wherez/rescale/bucket) covered
 compositionally; had the row failed, per-kernel bisection was the plan
 (unneeded). Wired into test_c.py (c-cuda-flagship).
 ## #LIB-038: emission driver + substrate matrix (v1.2 gate 3) [CLOSED]

 chain/substrate.py: whole-program emission per (program, substrate) —
 numpy executes text, C/CUDA run exchange binaries from ONE shared builder
 (same in.bin format both sides). flagship_c_exchange (full C flagship,
 orchestration mirroring CUDA launch-for-launch from proven C fns +
 driver-local statics) bit-exact FIRST RUN. Matrix (test_emit.py): 3/3
 flagship cells exact + dispatch proven via run_log (agreement alone
 would also match a silent fallback) + xf numpy standing cell + xf C/CUDA
 loud refusals (missing cells refuse, never fall back — refusals flip to
 numbers as lowerings land) + bogus names refused. Decisions: (1) substrate
 is a RUN concern, deliberately NOT a CONFIG key (CONFIG travels with the
 program; listings stay machine-independent); (2) test_flagship_cuda's
 builder moved into substrate (chain owns building, tests own gating —
 layering); (3) one Makefile vars edit silently didn't land (caught by
 read-back, fixed directly — verify writes, don't trust return codes).
 Wired into test_c.py (c-emit-matrix).
 ## #LIB-039: access-shape census (v1.3 gate 1, spatial not statistical)
 [CLOSED]

 chain/census.py: static producer/consumer/fan-out per stream + access
 class per op (elementwise/moves/gather/reduce/select/stencil/broadcast),
 zero execution. Real maps: flagship A×3 (SPLAT/SUB/ADD), xf XN×3 (QKV) —
 reuse made visible (the promotion rule's raw material, measured not
 argued). One miss on first probe: RESCALE unclassified (42nd mnemonic —
 the coverage gate exists precisely for this; table now 42/42).
 Doctrine: census answers HOW data moves (topology + geometry + access
 class); ranges.py answers what VALUES span — complementary instruments,
 neither subsumes the other.
 ## #LIB-040: causal probes via run overrides (v1.3 gate 2) [CLOSED]

 ASM.run/run_text take overrides={stream: value} (replaced post-compute,
 pre-consume; trace shows downstream truth; dead names fail loud).
 Results (flagship listing, test_probe.py): D-zero 38.46dB in predicted
 [20,45] (mean 1.35 LSB ~= the full enhancement -- the boost IS the detail
 path); COH-zero 44.23dB FALSIFIED a predicted [3,40] upper bound (0.46
 LSB -- coherence decides WHERE, iso/atten carry HOW MUCH); revised band
 [38,50] confirmed on held-out content (42.66dB) with D holding too.
 Two catches: (1) flipped-frame "held-out" proved nothing (pipeline is
 flip-equivariant -- identical decimals gave it away; different images
 only); (2) probe TABLE is the representation (one row per intervention,
 growing into the content map). Methods aren't structures, so no
 INVENTORY entry (stated) -- the pattern lives here + LANGUAGE.md §8.
 ## #LIB-042: full-sweep equivalence classes (deeper, not wider) [CLOSED]

 All 11 flagship streams probed: four EXACT classes -- {D,BEFF,BD} no-boost
 (38dB), {Y,A,AE,YENH} halved-image (15dB), {LIN,G} black (3.5dB),
 {COH} iso-fallback (44dB), {AS} max-boost (19dB, biggest honest change).
 Two classes PREDICTED from pipeline algebra, two DISCOVERED (LIN==G via
 GAIN's direct LIN read -- the census fan-out of LIN over LUMA+GAIN
 already recorded the subtlety; instruments corroborate). AS-zero is the
 most informative row: killing the blur reference = full A as detail.
 Standing rule: probe every stream of a listing before calling it
 understood (the sweep is the unit of "known").
 ## #LIB-041: information round — xf rows, S11 hunt, L4 entries [CLOSED]

 Probe table grows at STRUCTURE granularity (O/DOWN outputs, never
 mangled sub-wires — the assembly-view doctrine applied to probing):
 O-zero 36dB, DOWN-zero 43dB, both deterministic to 0.01dB, MEASURED not
 barred (toy weights; bands form later per LIB-015). S11 hunt (coh vs
 DAV2 depth-edges, shared f_012): pearson 0.27, top-decile overlap 0.29
 (~3x chance) — method works, not evidence; NOT a second instance
 (texture-without-depth and depth-without-texture both exist).
 INVENTORY gains L4 (content motifs) with S21/S22 PREDICTED + canonical
 sketches + matrix rows. Standing rule, earned twice now: probe INTERNAL
 names and you inherit expansion counters (attn_core#1.P renumbers when
 listings change) — probe STRUCTURE outputs (O, DOWN), which are stable
 by the composition-contracts discipline.
## #LIB-043: wider — census versions + inverted stabilize prediction [CLOSED]

 Census v1.1: per-stream VERSIONS (IN seeds read at v0, each OUT bumps;
 single-assign streams behave exactly as before — all prior gates green
 unchanged). Stabilize's dprev is the forcing case: seed v0 read by WARP,
 carried v1 read by MIXDYAD/BETA, producer SUB, fan-out 3, shadowed IN
 declared. Without versions the census conflated seed with carried memory.
 Probe side: predicted W-zero EXACT under static flow; measured identical
 outputs under BOTH flows instead. The listing header already contained
 the answer ("moving pixels trust the current frame bit-clean"): MIXDYAD
 mixes memory into STATIC pixels, passes moving pixels direct — the exact
 inverse. Corrected: static 27.34dB MEASURED (memory's real share on still
 content), motion EXACT (moving trust bit-clean, gated). Kept as the
 record, not edited away. Debugging took 2 minutes because the versioned
 census showed the read order directly.
## #LIB-044: first modification (iso variant, editable structures) [CLOSED]

 programs/flagship_iso.asm: SPLAT_BLUR+BETA_V5 replaced by ISO_BLUR+scalar
 BETA (two-line structural edit). Variant bit-exact vs the hand-written iso
 chain FIRST RUN; differs 45dB / 0.8 LSB from the v5 flagship (the edit
 moves values); census shows the new structure (AS from ISO_BLUR, no COH
 stream). Modification with before/after numbers against the probe baseline
 — listings are editable artifacts, not frozen text. Standing rule: every
 variant carries its own exactness gate (meaning preserved) plus a differs
 gate (edit bites) plus a census check (structure visible).
## #LIB-045: weight-edit routes — what responds, what doesn't [CLOSED]

 Four routes tried on xf_block toy weights (test_edits.py): A. whole-matrix
 zero orders V/O 36dB > MLP 43dB > Q/K 50dB (mechanism: small scores ->
 softmax near-uniform, V carries signal); B. channel zero spreads 15.6dB
 across 16 wv channels, ch3 dead across held inputs with fixed weights
 (partly weight-property, partly input — the separation method IS the
 finding); C. scale-x2 == zero in dB (36.40 vs 36.43: |2O-O|==|O-0|,
 linearity check, not new info); D. rank truncation hits flat spectrum
 (0.37..0.00, random weights) — no low-rank structure TO exploit, route
 meaningful only on trained weights (deferred with reason, not missing).
 Standard process: whole-matrix ordering -> channel sweep (fixed weights
 x N inputs, stable-important channels) -> rank spectrum -> candidate edit
 with predicted band + held-out confirm. Method gates: ordering, spread
 exists, scale-linearity, rank-monotonicity (all deterministic, seed 0).
## #LIB-046: singular directions beat channels (basis-independence) [CLOSED]

 Same probe, geometric basis (W - s_i u_i v_i^T on wv): spread 51.4dB vs
 15.6dB coordinate channels, delta tracking singular magnitude
 (corr -0.84 — the probe follows weight energy, as algebra says it must),
 near-null direction removable at 91.4dB. The coordinate finding (ch3
 dead) was statistics in a costume; directions survive re-basing, so they
 compose and channels don't. Standing rule: ablate the space's own basis
 (SVD), never the ambient coordinates — gauge freedom makes coordinate
 attributions evaporate under re-basis. Next: trained weights (flat
 spectrum here is a toy artifact; decaying spectra will separate harder).
## #LIB-047: real weights — Qwen2-0.5B L0 MLP (v1.3 gate 1 real half) [CLOSED]

 Trained weights + real embeddings through programs/mlp_qwen0.asm
 (test_realw.py, SKIPs without the HF cache): block parity 51.82dB vs
 torch.float64; spectrum decaying 17x (3.54..0.21 — the toy's flatness was
 a toy artifact, as predicted); direction spread 33dB with corr -0.81
 (tracking law reproduces); planted dominant direction recovered
 BIT-EXACTLY (control 18.8dB). Divergences, all stated: attention is
 boundary float (folded scores hit 964, 1000x past the softmax contract —
 T-transform backlog measured live, not worked around); K-bias omitted by
 cancellation proof; V-bias negligible (max 0.1); Q-bias lives in torch H;
 per-model eps first instance (eps_rms_c 68719 = 1e-6); rope_base 1e6 to
 match. v1.3 gate 1's real-family report: DONE (via cache, better than
 via handoff).
## #LIB-048: rank-1 implant with functional aim (targeted writes) [CLOSED]

 W += A·u·vᵀ with u = MID[t]/||MID[t]|| concentrates on token t by
 Cauchy-Schwarz (test_implant.py): target rewritten at NEGATIVE dB
 (-14.3 t1, -10.7 t4 — change exceeds the signal, local rewrite) leading
 the field by 35.4dB / 26.2dB on target + held-out tokens. No semantics
 used — aim is functional (the token's own activation direction), which
 is exactly why this works WITHOUT the labeling loop and exactly why the
 loop is all that remains for meaningful writes. Standing distinction:
 graffiti (random plant, 21.6dB smear-with-pattern) vs writing (aligned
 plant, negative-dB rewrite with 26dB+ specificity) — same primitive,
 different aim, both gated.
## #LIB-075: demo 2 — memory from scratch, capacity reasoning holds [CLOSED]

 programs/assoc_mem.asm (MATMUL similarities + ARGMAX + GATHER, 5 lines)
 + test_demo2.py: 16 seeded bipolar patterns, dim 64, NO trained weights
 (patterns are data, noise is draws). Recall 100% at 0/8/16 flips --
 bands asked exact/100%/>=80%, all exceeded with margin (capacity
 reasoning conservative by design: correct leads 2*(32-f) vs spread ~24).
 Behavior from composition alone (recall curves, no reference to match):
 the demo-2 brief's success criterion, met first try. Storage angle: the
 pattern bank IS a frozen store (seeded); ENGRAM framing applies verbatim
 (keys match, values return, gains uniform) -- memory without training,
 the oldest dream in the program, running in 5 lines.
## #LIB-074: demo 1 greens by coordination (9 rounds) [CLOSED]

 Teal expansion took NINE designs (dd_demo1.py): gain eaten by spectral
 norm; direction impotent on dead slots; metric blind (positive-clip);
 premise broken (q27 already teal); file drift ran Q=41 twice unknowingly
 (verified, not assumed, after); signs wash means out; aligned slot weak
 alone. Winner: rotate top-5 teal-ALIGNED slots (87/57/4/98/28, corr
 +0.78) to teal @ preserved norms -> energy x1.25, global -3.7dB, both
 bands held. Moral, stated as law: DISTRIBUTED representation yields
 only to COORDINATED writes (superposition budgets exactly, Q1); single
 slots move pixels, never aggregates. Spectral norm eats magnitude
 (edit direction), dead slots eat direction (borrow votes or pick live
 ones), means eat signs (measure distributions). Nine falsifications,
 one compositional success -- research priced honestly.
## #LIB-073: t-transform branch pushed — smaller correct design [CLOSED]

 Pushed phi-core branch ai/t-transform-bridge (their suite green):
 to_fixed_wide (new frac_hi LUT to +-2200, existing to_fixed UNTOUCHED,
 M1 assert stands) + tests/test_wide.py (additivity, 0-count exactness,
 3.6e-05 full-range parity). En route: the planned one-assert relaxation
 PROVED WRONG by units analysis (counts U_m-relative vs absolute table)
 BEFORE any commit -- analysis killed a bad design pre-birth, cheapest
 possible stage. Two fixture bugs caught by the new gates (U_BIAS=1
 coverage; lattice-vs-raw-float comparison). residuum: C mirror,
 TILE listing-form, owner review -- all stated on the branch.
## #LIB-072: T-transform specified — fold, ranges, one-assert fix [CLOSED]

 v1.4 gate 4 lands as DESIGN (docs/T_TRANSFORM.md), not experiment: the
 exp path already max-subtracts (only the BIAS bridge blocks); the bridge
 FOLDS out-of-range to +/-1.0 (never saturates -- read, not assumed);
 per-row ranges measured all 9 DDColor layers (7.4 .. 614.4) + Qwen 963.
 Fix splits at the seam: phi-core relaxes one assert (bridge param, LUTs
 untouched, BIAS callers safe) offered as staged branch; our side ships
 max-sub composition (ARGMAX+GATHER-tile+SUB at covering m) with gates
 ready (0-diff, full-range parity bar 40, in-contract regression).
 Alternatives killed WITH measurements/reasons (global shift, clip,
 temperature, triple-native exp). Tiling ids honestly named as the ugly
 corner (TILE mnemonic if demanded). Chasing every lead converged: most
 died, one assert survived.
## #LIB-071: selection-side decomposition — S08 first instance [CLOSED]

 programs/attn_mini.asm + test_select.py: output linear in V at fixed P
 (P bit-exact under ALL V masks -- precondition, not assumption),
 recompose 82.9dB (sum of single-row outputs vs full), top-beats-bottom
 ordering (26.7 vs 27.3: thin by mechanism, flat attention ~= flat
 contributions, stated IN the gate string). S08 PREDICTED-ish -> SINGLE;
 biggest-unverified-claim flag retired with honors. Two process notes:
 (1) the contract gate as first written asserted NOTHING (check True --
 wallpaper by construction); caught on re-read, now measures scoremax
 0.275 (tripwire gates must MEASURE, even trivially); (2) a tool call
 reported failure while the diff proves success -- verify writes by
 read-back AND diff, never by return code alone.
## #LIB-070: projection banks + form-tax vs floor (v1.4 gate 1) [CLOSED]

 test_projbank.py (block_inputs bundle consolidates the 5th mirror):
 6/6 banks 59-63dB (q/k/v/o/up/gate -- linear half under CRUD, no new
 math). Prune-confirm needed TWO falsifications to land: (1) cross-form
 61.3 vs 71.6 predicted -- the ~61dB FORM TAX dominated the tiny true
 delta (compare within-form instead); (2) within-form 76.1 vs 71.6 --
 LESS damage than float predicts, because tail components live UNDER
 the lattice quantum floor (removing what the lattice already rounds
 identically changes almost nothing; candidate mechanism, stated not
 gated). Doctrine refined: form-tax (compare within-form) AND floor
 (tiny removals cost less than float says) are the two corrections to
 naive linearity -- both now have first measurements. Process debit:
 another duplicated-lines slip (double RESULT print), caught by counting
 outputs -- the duplication species keeps recurring; edits get diffed,
 always.
## #LIB-069: v1.4 defined + DDColor scores kill gate 3 (measured) [CLOSED]

 docs/ROADMAP_v1_4.md (attention-side stores: projections, selection/S08,
 conditional DDColor layer, T-transform spike). Decisive probe FIRST
 (dd_score.py): DDColor cross-attn scoremax per layer reads 7/12/10/20/27/
 43/107/429/30 -- layer 7 at 429x the softmax contract. Gate 3 WAITS, by
 measurement exactly as the roadmap prescribed (no hope-based planning).
 Side finding: scores SHARPEN through depth then moderate at the last
 layer (7->429->30?) -- decoder dynamics worth their own probe later.
 Probe bugs en route (kwargs-hook arity, batch-vs-head geometry, 3-arg
 hook): all shape-loud, all fixed in minutes. Strictness works on
 mirrors too (standing observation, third instance).
## #LIB-068: edit receipts + DDColor stores frozen (handoff sequence) [CLOSED]

 docs/EDIT_RECEIPT.md: change/prediction/measurement/footprint/verdict
 with honesty grades per predictor (factorization +/-3dB, analog +/-10dB
 triage, novel = wide/reported) + filled receipt #001 (q56 SPLIT, residue
 intact). test_engram.py grows DDColor stores (dd_qe/dd_qf/dd_refine
 roundtrips 1e-14..1e-16): edits now start from versioned data, not the
 211MB blob. Sequence position: receipts make TODAY transferable; stores
 make edits addressable; attention-side stores come next; T-transform
 waits at the end, measured at 964x and getting no closer on its own.
## #LIB-067: disentangle (place+color pair) + resonant bridge fails [CLOSED]

 Disentangle (dd_modify modes, f_014): query-only 36.0dB (vote mass flips,
 post-silence 36.1dB), refine-only 19.0dB (votes UNCHANGED -14645 both,
 post-silence 44.0dB), both 14.5dB (super-additive: coherent, not linear).
 Predicted query<25 AND refine>25 -- BOTH falsified inverted. Attribution
 RESOLVED anyway: query rows = WHERE/how-loud, refine row = WHAT hue;
 the slot is a (place, color) PAIR, neither half "the" cause. Hue mass
 still flat because 135deg saturates nothing and moves little alone --
 the pair writes together or not at all.
 Resonant bridge (dd_resonant.py): scalar phase 0.01, H=8 signatures 0.04
 (chance 0.03) -- SCRAMBLES twice. Mechanism: mod-2pi folding destroys
 metric structure after random projection (wrap to uniform); resonance
 promises EXACT-match + uniformity (dedup!), never high-dim similarity.
 Correct bridge, restated: phase-address SCALAR engram attributes (hue
 angle IS circular 1D: nearby phases = nearby hues, the scalar case that
 works), never projected key vectors. Failed bridge > no bridge: the
 constraint is now stated, not guessed.
## #LIB-066: add/remove demo — norm-fit limits, slot live, hue open [OPEN]

 Norm-fit LOO: p50 err 4.6dB, p90 10.9dB, extremes -9/+20dB -- TRIAGE-grade
 (+/-10dB bands), NOT preview-grade: predictability tracks LINEARITY of
 the path (SVD law exact -> +/-0.3dB; palette removal through 9 nonlinear
 layers -> +/-10dB). Errors at extremes, honesty about the middle.
 ADD (new hue 135deg at dead q56, dd_modify.py): hue-appears FALSIFIED
 (query-hue sparsity != output absence -- 135deg already at 16% mass);
 bounded CONFIRM (14.5dB < 35). Follow-up: slot IS live (vote mass flips
 sign, post-write silence moves 21.4dB) but hue unproven -- refine-row vs
 query-row attribution UNDISENTANGLED (open: run each edit alone).
 Reliability posture, stated: act on MEASURED costs with margins (catalog:
 100 x 0.2s), predict for triage only. "Error free" = dB-bounded with a
 stated bar, never bit-exact multi-hop (ceiling stands). (Disentangle
 closed in LIB-067: slot = place+color pair.)
 Resonant-array DNA verdict (owed): shared = associative key-value
 content addressing with similarity match; different = deterministic-
 constructive (Riemann phases, exact, designed) vs learned-measured (SVD
 keys, dB-gated, discovered). Bridge proposal (untested): index engram
 keys by resonant phase for exact cross-model lookup -- deterministic
 addressing over discovered content.
## #LIB-065: full palette catalog — statics beat dynamics [CLOSED]

 docs/PALETTE_CATALOG.csv (dd_catalog.py, 28s for 100 queries x 2 images):
 per-query hue + refine norm + vote masses + causal dBs. Findings: causal
 range 3-59dB (q77 alone moves output at 3.2dB -- concentration worth
 noting); cross-image causal corr 0.988 (hierarchy nearly identical);
 refine-norm predicts causal share at 0.78 while vote MASS manages -0.40
 (inverted-ish: the STATIC weight beats the DYNAMIC vote -- read the
 weights, again); top-20 causal hue bins cluster blue-purple-magenta
 (225x7/270x5/315x4); dead both images: only q56. The modification
 premise this enables: add/remove palette entries with refine-norm as
 the cost predictor (no runs), causal silence to confirm. Next: reliable
 add/remove with predicted cost (the user's brief).
## #LIB-061: DDColor query mining — slots, votes, footprints [OPEN]

 Triage (2026-10-01): NO English anywhere (no text/CLIP in arch or keys --
 the "identification" is 100 unlabeled 256-dim slots, not language);
 compression worry CONFIRMED (our V20 port deleted the transformer; mine
 the original 211MB .pth only); unexpected load keys = unused cls head
 (non-load-bearing). Weight-space: queries alive + evenly spread (flat
 spectrum 23->21, max cos 0.28, norms 12-16), feat matrix distinct (4.1x).
 Vote maps (1,100,256,256) diffuse (entropy ~0.95): queries read as a
 GLOBAL palette, not object slots -- affinities don't segment. BUT causal
 footprints do: silence q39 = 26.1dB biting DARK pixels 3.7x over bright
 (corr -0.72 with L); silence q0 = 13.0dB (mass != causal share, again).
 First functional label candidate: q39 dark-region colorizer. Open: same
 query second image (loop material), query attention maps (reads, not
 votes), 98 queries unprobed. Scripts: dd_mine.py, dd_footprint.py
 (one-shots, HF-cache + ddcolor_reverse checkout needed).
## #LIB-062: q39 loop closes — dark-region label x2 images [CLOSED]

 dd_footprint.py now loops IMGS (f_014 + f_012): q39 silence on the FRESH
 image reads 25.8dB (vs 26.1) with footprint-vs-L corr -0.74 (vs -0.72),
 darkmass 0.467 vs bright 0.160 (~2.9x). Both bands held (corr <= -0.5,
 global < 40dB): q39 IS a dark-region colorizer, twice in a row, same
 signature to a decimal. q0 reproduces too (11.3dB/-0.38 vs 13.0/-0.33,
 measured rows). Second instance of a loop-verified label class: match on
 fresh content + verify bands, no new machinery. The loop is now routine
 (third close counting Qwen x3): hypothesize from fingerprints, silence
 for match, write for verify, record both bands either way.
## #LIB-063: brightness vs object — neither, it's band-pass [CLOSED]

 q39 footprint across L deciles (both images, ~identical): 1.25 1.49 1.73
 1.69 1.66 1.15 | 0.41 0.29 0.26 0.08 -- rises to deciles 2-3, plateaus,
 then falls off a CLIFF at decile 5-6. Not monotonic brightness-gating
 (would peak at 0), not object-gating (within-dark grad corr -0.04/-0.16:
 no edge preference). q0 mirrors with its own peak (decile 5: 2.19) and
 the SAME cliff (2.19->0.31). Verdict: queries DIVIDE THE LUMINANCE AXIS
 among themselves -- band-pass specialists with a shared bright cutoff.
 The brights (deciles 6-9, ~0.1-0.4 here) belong to other queries
 (41/94/24/64 candidates, unprobed). Zero new runs (saved footprints);
 analysis only. Next: hunt the bright owners.
 (CORRECTED by LIB-064 below: the band-pass was need-driven, not
 specialization. This entry stands as the record of the wrong turn.)
## #LIB-064: palette verdict — queries divide HUE, not space [CLOSED]

 Bright-hunt prediction FALSIFIED (all four top-mass queries bite dark
 too, bright/dark <= 0.26 -- no mirror anywhere). Control decided it:
 base |ab| itself reads 7..13 dark vs 1..4 bright (same shape as every
 footprint) -- dark-bite is NEED-driven. Pairwise footprints correlate
 +0.73..+0.99 (same places, up to scale); normalized profiles flat
 (q39 ~0.06 everywhere, rest ~0.01). Then the zero-run kill: refine conv
 rows give each query a chroma DIRECTION -- 100 vectors spanning the FULL
 hue circle (all 12 bins, 4-13 each; q41 strongest 0.45@353deg, q39
 207deg; spectral /sigma preserves directions exactly). VERDICT: the 100
 queries are a learned PALETTE (spatially overlapping, chromatically
 distinct), not object slots. q39-dark label REASSIGNED (effect real:
 26dB twice; mechanism = 207deg hue needed most in shadows -- shadows run
 blue, consistent). Specialization found on the THIRD axis tried (space,
 then brightness, then hue): keep two hypotheses dead for every live one,
 and READ THE WEIGHTS before running the model.
## #LIB-060: storebanks as listing data (native, both fixtures) [CLOSED]

 stdlib/storebank.asm (storebank_apply DEF -- implant_apply's K>=1 twin,
 named for what it is) + engram.bank() (gains folded host-side, Ub/Vb
 ready to encode) + variant listings both fixtures. Toy: bank parity
 96.1dB, pruned bottom-8 reported 49.5dB. Real: parity 55.8dB (>=50
 predicted: toy 96 minus coarser quanta), pruned-27 lands 42.7 vs 42.9
 predicted -- AND matches test_prune's surgery-form 42.8 within 0.1dB
 (three forms agree: surgery, bank, statics). Pruning is now an
 assembler-side edit (rebuild bank with fewer columns, predicted cost
 via read.py); listing-text pruning (masked sums) stays v1.4 horizon,
 stated in the DEF header. Gains live in Ub (retune = rebuild, stated).
## #LIB-059: native ENGRAM storage (v1.3 gate 6) [CLOSED]

 chain/engram.py (decompose/freeze/load/recompose, STORAGE only -- cost
 stays in read.py, no duplicate) + test_engram.py: roundtrip 6.4e-16,
 freeze-twice identical bytes, loaded stores bit-exact vs direct weights.
 Blobs OUT of git (41MB vs 1.9MB repo; stores/*.npz ignored, .json
 sidecar tracked): S17's commit-priors rule bows to hygiene WITH the
 determinism gate as compensation (reproducible, not precious).
 v1.3's definition extended in ROADMAP_v1_3 (gate 6 added; gate 4's
 deferral lifted for the implemented loop). Remaining for native:
 STOREBANK sections + assembler-side SVD + cost model over stored data.
## #LIB-058: CRUD round — retune, create, null-shelf, layer-1, ENGRAM [CLOSED]

 Q2 retune = signed ablation (gain-x2 22.1dB == ablation 22.1dB, exact to
 0.1). Q3 CREATE from the (dir24, France) label on a THIRD prompt: key
 from old France-MIDs, value Vt[24], gain s24 -> France-pos rank 1/8 at
 8.4dB, gap 22.6dB (creation from knowledge, not fit; proper nouns leak
 slightly -- Germany/Switzerland move too, stated). Q4 shelf null: same
 write on pruned vs full weights identical at 57.8dB -- shelves are
 CAPACITY (free space + headroom), not interference; freeing changes
 nothing about what a write does. Q5 layer 1 reproduces the pattern
 (spectrum 12x, giant 14dB, spread 21dB, corr -0.54 -- weaker tracking
 with n=8 caveat, holds past bar). NAME: the directional store is an
 ENGRAM (S23 CONFIRMED: instances across toy + L0 + L1; second model
 family wanted). Tests: test_prune.py grows retune; test_create.py
 (create + null); test_layer1.py (spot-check).
## #LIB-057: pruning by prediction — combined linearity (CRUD Q1) [CLOSED]

 All 27 predicted-dead dirs removed in ONE edit (test_prune.py): combined
 cost predicted statically from linearity (sum of delta vectors, no runs)
 at 42.9dB, measured 42.8dB in a single run (err 0.1dB). Two lessons:
 (1) combined-cost linearity holds across 27 components -- superposition
 is exact enough to budget; (2) "dead" is PER-DIRECTION: individually
 >55dB each, 42.9dB together (small deltas accumulate incoherently).
 Pruning 27 shelves is above bar but NOT free -- the honest hundredth
 decimal of the shelf story. chain/qwen_mirror.py grows mlp_forward
 (promotion working: 4th inline copy avoided).
## #LIB-056: blind prediction — edit-with-preview (0.2dB) [CLOSED]

 Calibrate C once (29.347), predict never-run dirs from statics: 39.3->39.5,
 43.6->43.8, 43.2->43.5 (errs +0.2/+0.2/+0.3, locked before running).
 Predictor standardized (calibrate_C/predict_db; logic gated exact,
 lattice accuracy measured). Consistent +0.2 bias noted unexplained
 (quantization-floor candidate, one line not a theory). Uses unlocked:
 cheap readouts (statics + handful), principled pruning (cut by predicted
 share at stated cost), implant aiming. Second no-op edit caught en route
 (whitespace-only change reported success -- verify writes by read-back,
 the standing rule that keeps earning).
## #LIB-055: factorization — content is key alignment (0.998) [CLOSED]

 residual = dB + 20log(s) vs -10log(mean key-alignment^2): corr 0.998 on
 the 112-direction real readout (form gated at 1e-12 -- first drafted as
 bit-exact, FAILED honestly: BLAS reorders float sums, 9e-16 measured;
 the gate now states what floats actually guarantee). Ablation deltas = energy x alignment, both static; runs only confirm.
 Selectivity 0.64 (moderate, stated). Consequences: magnitudes predictable
 run-free; loop match-step drops from ~100s (ablations) to ~ms
 (alignments); labels ARE alignment profiles. The mathematically
 fundamental thing: the key-value store model is not an analogy here, it
 is the exact computation, and every content question reduces to "which
 content aligns with which key".
## #LIB-054: three labels, S21 CONFIRMED, generalization discipline [CLOSED]

 Loops 2+3 via generalized loop_turn.py (argv over dir/token/prompt/pos):
 (dir32, capital) rank 1/8 + -12.2dB/32.7 gap; (dir328, Paris) rank 1/8 +
 -15.1dB/25.9 gap. 3/3 matches rank ONE (bands asked top-3); S21 SINGLE ->
 CONFIRMED. Caught: generalization shipped s[24] unparameterized
 (wrong-magnitude silence) -- found by reading, fixed, replay-verified
 bit-identical at 33.9/-12.7. Rule: generalize ONLY with replay proof
 (mirrors drift exactly this way, cf LIB-014 postscript); loop1.py kept
 as the original record, loop_turn.py as the instrument.
## #LIB-049: directional stores as structure (implant DEF) [CLOSED]

 stdlib/dirstore.asm implant_apply (2 MATMULs) + variant listings
 (xf_block_implant, mlp_qwen0_implant): the rank-1 write as TEXT instead
 of weight bytes. Toy: sham 99.8dB (parity not exact -- ADD re-bridges),
 listing-vs-surgery 91.2dB, A=0.2 keeps both paths in-envelope (A=2.0
 saturates each side differently, 16dB -- envelope discipline). Real:
 transfer 36.8dB vs predicted 60 -- FALSIFIED, bisected to small-vector
 ENCODE quantum (u at +-0.009 pays ~0.2% per materialization; surgery
 encodes once at +-0.44), NOT saturation; the gauge "fix" made it worse
 (36.8<38.6), which is how the true mechanism got found. Doctrine:
 materialization pays quantum tax per hop (LIB-006's lesson, generalized);
 gauge must balance quantum (u,v LARGE) vs envelope (intermediates SMALL);
 m_acc headroom is the untested release valve. Falsification with a
 bisected mechanism beats a tuned pass.
## #LIB-050: first model read + searches (labeling-loop input) [CLOSED]

 chain/read.py (readout tables + dead_shelves/movers/selectivity) +
 test_read.py (instrument gates on toy) + docs/MODEL_READ.md (the actual
 112-direction read of Qwen down_proj). Answers: 2 dead shelves
 (864/872); France/Paris share giants (0/8/72) with distinct leaners
 (16/24 vs 96/328); selectivity spreads 29dB (328->Paris, 24->France).
 Caveats fenced in the doc (sampled 1/8, positional-not-semantic, one
 prompt). Standing point: the read is a TABLE, searches are QUERIES --
 information retrieval over weights, and the labeling loop's input is now
 a concrete artifact instead of a wish.
## #LIB-053: loop one closes — (dir24, France) verified [CLOSED]

 docs/LABELING_LOOP.md's first full turn (loop1.py): hypothesis from Q2
 (dir24 selective for France) -> fresh prompt, France moved 3->5 ->
 silence: France-pos rank 1/8 at 33.9dB (band top-3 + <=45) -> write
 dir24's Vt output keyed at France: -12.7dB rewrite, 24.4dB gap (band
 >6). Both held with margin; S21 PREDICTED -> SINGLE (first verified
 label; second instance wanted). Process notes: (1) chain/qwen_mirror.py
 promoted (third inline copy of the boundary mirror -- the rule fires on
 helpers, not just ops; old copies migrate on touch); (2) a shape bug
 (U-side vs Vt-side of the write) caught on re-read BEFORE running --
 reading the geometry first is now habit; (3) ~6 listing runs per loop
 turn: labels are cheap to verify, expensive only to hypothesize.
## #LIB-052: shelf map + labeling-loop draft (both) [CLOSED]

 shelf_map (chain/read.py: intersection/union over readout grids, mismatch
 fails loud; toy-gated): three contexts give per-context dead
 [864,872]/[]/[] with EMPTY intersection at 55dB -- no direction is dead
 everywhere (near-misses 54.9/54.4). Threshold-fragility stated, bar left
 untuned; design consequence: target the union with per-context budgets,
 never assume a safe intersection. docs/LABELING_LOOP.md (draft v0.1):
 hypothesize (d,P) -> match P at new positions -> verify BY IMPLANT
 (labels predicting interventions are knowledge; the rest stories) ->
 close in INVENTORY. The loop's verifier already exists (26-35dB implant
 gaps); its inputs exist (readouts + stability columns); only the first
 hypothesis is missing. Giant dir0 invariant to ~1dB across all three
 domains (22.1/22.0/23.0) -- shelves for the loop to stand on.
## #LIB-051: stability splits on the predicted seam (second prompt) [CLOSED]

 Same 112 directions, quantum-domain prompt: global-dB rank Spearman 0.55
 (giant dir0 identical at 22.0-22.1dB, top-10 overlap 6/10) vs selectivity
 rank Spearman 0.25. HOW MUCH is partly stable (weight property), WHICH
 TOKEN is contextual (input property) -- the positional caveat from
 LIB-050 calling its own shot. No gate (measured rows per LIB-015; bands
 when the claim hardens). The labeling loop now has two columns to join
 on: stable magnitudes across contexts, fingerprints within them.
 ## #LIB-036: elementwise CUDA batch (v1.2 gate 2b) [CLOSED]

 9 kernels (holo_elem.cu: sqrt/tmul/tdiv/binop/square-clip/sigmoid/abs/
 relu/clip) + tasks 2..10 on the units driver, each vs its proven numpy
 fn — 12/12 bit-exact incl. zeros/negatives and a wide-range sigmoid row.
 Three slips: (1) phi-core's ".cuh comment says LUTs fit 32 bits" is FALSE
 for SIGX (3.9e17 — int64 device pointer, measured not assumed);
 (2) stale-binary false failure (tests skip rebuild when the binary
 exists — Makefile now tracks holo_elem.cu); (3) my own fixture size
 slip (caught by reshape, not by values). Encode stays HOST-side
 (S.encode needs float log — the float→int seam lives at the boundary
 by doctrine, same values, conversion location documented).
 ## #LIB-033: recognition-first parse, negative result recorded [CLOSED]

 Roadmap gate 3 (S19 bet): parsed diffusion DDIM sampling into INVENTORY
 before writing any code — dated negative (schedule-blend ≠ dyadic mix,
 no warp, noise-seed ≠ first-frame-still, time-conditioned net), S19 stays
 SINGLE, matrix cell updated, residue assigned to S09 + STATE discipline.
 Method note: negatives get the same writeup standard as positives
 (evidence, reasons, residue) — an unrecorded parse is a rumor; a recorded
 negative is a result. Time-box honored: static parse only, no code run.
