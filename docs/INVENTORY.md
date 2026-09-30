# INVENTORY (DRAFT v0.1 — the structure list; correct me)

Claim under construction: the models are compositions of ~15-20 recurring
structures. This file is the list. Each entry: canonical form, known
instances (model:location), substrate coverage, status. Status values:
CONFIRMED (2+ independent instances, gated) / SINGLE (one instance — rumor,
needs a second) / PREDICTED (no instance yet, address reserved) /
FAMILY-SPLIT (one shape, parameterized variants — the split itself is gated).
"?" marks what the drafter couldn't confirm — fill these in.

The traversal matrix lives at the bottom: structures x models. Empty cells
are the program. A new model should be PARSED into this list first
(recognition-first porting); anything unparseable is either a new structure
or a family split — both get entries here before code.

## L0 — substrate

### S01 lattice codec [CONFIRMED]
sign x PHI^(exp/512) triples + int64 fixed. Instances: phi-core/lattice.py,
rife vendor copy, holo (imports phi-core), esrgan, dav2 (encode/decode triple
form), diffusion S3 (?). Coverage: numpy everywhere; C (bridge.c); CUDA;
torch emu. Open: DAV2's exact triple layout vs phi-core's — same or drifted?

### S02 bridge + tagged scales [CONFIRMED]
to_fixed/from_fixed, fix@m tags, rescale-only, tag-mismatch fails loud.
Instances: rife, esrgan (same datapath — the reuse that motivated phi-core),
holo (rescale_, tag asserts gated). Family split: DAV2 accumulates in LOG
domain (phi_add/sub LUTs) instead of fixed bridge — same role, different
family. Is log-domain a variant of S02 or its own structure (S02b)? OPEN.

### S03 frozen transcendental LUTs [CONFIRMED]
EXPACT gather + table (sigmoid; SiLU 2D 9KB in phi_lattice). Instances: rife
sigmoid_int, holo sigmoid_trip — SUBSTITUTION-PROVEN bit-exact across
numpy<->torch/CUDA (test_substitute.py). Coverage: numpy, torch, C
(holo_ops). The first proven cross-model organ transplant.

## L1 — compute motifs

### S04 integer conv [CONFIRMED, FAMILY-SPLIT]
Per-tap tmul + to_fixed@m_acc + order-free int64 accumulate + from_fixed +
rescale. Instances: rife, esrgan, holo conv_trip, phi-core phi_conv (C),
diffusion S3 matmul (? — matmul vs conv: same structure or split? OPEN).
Split (gated): edge semantics — replicate (holo/rife-warp convention) vs
zero-pad (phi_conv). test_substitute edge-refusal characterizes it.
Deconv (esrgan, scatter-add + bias-adds rule)? Split or same — OPEN.

### S05 warp / bilinear resample [CONFIRMED]
nihui form (unclamped-floor alphas, clamped indices, replicate), flow in
2^-14. Instances: rife (float ref + torch int, border-divergence noted),
holo temporal warp_q (73dB parity, identity-exact). Convention split, gated:
OUR seam is (dy,dx); rife's warp_fixed reads [...,0] as X (#LIB-018).

### S06 exact select / gating [CONFIRMED]
np.where chains on integer masks (prelu sign-select -> select_mux triple
streams -> beta_field -> depth_mult -> flow consensus selects). Instances:
rife prelu_int (prior art), holo x4 (bank mux, beta, depth, flow). PROMOTED
(select_mux + holo_mux). Next: `decide`-family helper (3rd use reached:
motion-static, depth-median, router verdicts, #LIB-019).

### S07 normalization shapes [SINGLE?]
RMSNorm (phi_lattice), groupnorm/layernorm-rows (promoted from diffusion S3
per phi-core log). Instances live in phi-core/diffusion — NOT verified by
drafter (?). Confirm: which models, gated where?

### S08 attention as selection [PREDICTED-ish]
phi-form softmax = algebraic identity; attention as phi-level selection;
T-transformed integer softmax (phi_lattice section 08). Integer transformer
chain corr 0.9938 claimed in the book. Lowered in phi-core? softmaxN promoted
per log — confirm instance + gate (?). This is the biggest unverified claim
in the program. Mark honestly.

### S09 residual accumulation + schedules [CONFIRMED]
Adds with tag asserts, schedules as versioned data (IR.md rule). Instances:
everywhere (rife residuals, esrgan RRDB, holo binop adds, holo temporal mix
as IIR variant). The IIR mix (a=0.5 dyadic, state discipline) may be its own
structure (S19 below) rather than an instance — SPLIT CANDIDATE, decide.

### S10 geometric resampling moves [CONFIRMED]
pixelshuffle, dyadic interp (shifts only; non-dyadic is a LOWERING ERROR),
nearest2, concat/split/crop/transpose. Instances: esrgan (pixelshuffle),
rife (interp), holo (pads/crops). Exact-by-construction family.

## L2 — control motifs

### S11 structure tensor + coherence [SINGLE]
Tensor products + smoothing + discriminant-at-m_acc + aniso/trace ratio.
Instance: holo only. Second instance wanted: does DAV2 or RIFE compute
anything isomorphic (flow confidence? depth edges?)? OPENplusplus — a second
instance promotes this from trick to structure.

### S12 oriented bank + mux/blend [SINGLE]
V/H/diag bank, hard mux / fused gather / relu-blend v4. Instance: holo only.
(Predicted instance: any directional filter bank elsewhere? check diffusion
denoiser directional behavior?rk.)

### S13 beta-field decision tables [CONFIRMED lineage, SINGLE model]
v1 thresholds -> v2 bands -> v3 fitted table -> v4 sigmoid blend -> v5
learned gate. One model, but FIVE generations with the scaffolding reused
unchanged (pairs/score/freeze/gates) -- the lineage itself is the evidence
this is a structure (a *learnable slot*) rather than a hack. Second model
with a fitted per-pixel field wanted.

### S14 prior modulation (multiplicative caution) [CONFIRMED]
Offline prior -> boundary mask -> exact select -> tmul caution onto boost.
Instances: holo TWICE (depth_mult near/far, flow_scale fast/slow -- same
shape, different priors). Near-promotion: generalize to prior_mult() helper?
(Currently 2 uses; threshold is 3.)

### S15 meta-selection / router [SINGLE]
Decide among configurations at the boundary, consume exact flags; state
always refreshes. Instance: holo router (still/temporal per frame). Predicted
instances: per-tile routing (backlog), codec-level toggles as data (COMPOSE
toggles today are human-set -- routing them is the obvious next instance).

## L3+ — composition + process motifs

### S16 triple-direct handoff (L2 seam) [CONFIRMED]
No decode/re-encode at model boundaries; seam dividend priced in dB (51dB
RIFE->ESRGAN). Instances: RIFE->ESRGAN (files->triples), depth->holo,
motion consensus inputs. Compatibility contracts (versions, scale tables)
are the known gap -- first silent seam mismatch hasn't happened yet (when it
does, it gets an entry).

### S17 hashed-cache offline priors [CONFIRMED]
Compute once (any framework), hash-key sidecar, consume hot as integers;
collision raises. Instances: depth (samples/depth, committed), flow reserved.
Recipe status (#LIB-013).

### S18 fit/freeze loop [CONFIRMED]
Deterministic pairs + score + must-beat-to-rewrite + pairs-hash + schema
completion. Instances: holo calibrate, fit_ctrl, fit_v5 (three copies --
promotion to framework harness DUE, tomorrow item #1).

### S19 temporal IIR + state discipline [SINGLE]

Warped-detail memory, dyadic mix, first-frame==still, always-refresh store.
Instance: holo temporal. Second instance wanted (RIFE's own temporal
behavior? any recurrent use in diffusion sampling loops? -- diffusion
iterates the SAME net N times: is that S19 with a=1.0? Delicious question,
OPEN).
Parse 2026-09-30 (diffusion_reverse/demo_denoise.py, DDIM 10 steps):
DATED NEGATIVE, stays SINGLE. Shares loop-carried state + always-refresh
store + same-net-each-step, but fails the load-bearing elements: the mix
is a schedule blend (time-varying float coefficients from the noise
schedule — their own words: "float scheduler scaffolding"), NOT a dyadic
(1-a,a) mix; there is no warp/align of the memory; the first step starts
from pure noise (no first-frame==still identity — holo frame 0 output ==
still output exactly, gated); and the net is time-conditioned (temb per
step, no holo counterpart). a=1.0 would mean pure memory (Dmix=warp(Dprev))
— diffusion blends current latent AND net output, so the question as posed
answers NO. Positive residue: the schedule-as-data half parses as S09
("schedules as versioned data", already CONFIRMED) and the store half as
STATE discipline — recognition-first porting working (parsed into the list
before any code, no new structure claimed).

### S20 parity-with-basis + substitution gates [CONFIRMED, meta]
dB parity with declared basis; file-exchange exactness; mirrored-bug
awareness (rotation/symmetry manifests); tripwires against vacuous passes;
measured-rows doctrine for unreachable bars. This structure gates all others.
Instances: everywhere since holo started; phi-core's 0-diff extraction gates
are the prior art. The suite is the instrument -- and per S20's own logic,
 this entry needs no gate (it IS gates). Cute. Moving on.

## L4 — content motifs (v1.3 frontier: what weights CONTAIN, not their shape)

### S21 causal feature streams [PREDICTED]

A stream with a measured causal delta: zero/swap intervention moves the
output by a stated dB (basis declared), sham-exact control, determinism
to 0.01dB. Canonical sketch: (stream, intervention, ΔdB, control).
CANDIDATES (not instances — toy magnitudes, need real-model confirmation
before claiming): flagship D (38dB) and COH (44dB); xf O (36dB) and DOWN
(43dB) — all in test_probe.py's table. Related but NOT qualifying:
motion consensus direction-agreement (rescue-V 0.36->0.81) corroborates
directions across models without measuring a stream's causal share.

### S22 cross-model content correspondence [PREDICTED]

Same-scene structure claims from different models that covary above
chance with a stated mechanism. First measurement (2026-09-30, f_012):
holo coherence vs DAV2 depth-edge strength — pearson 0.27, top-decile
overlap 0.29 (~3x chance): consistent with "both respond to scene
structure", far from isomorphism (texture without depth, depth without
texture). Recorded as the method working, not as evidence. Instances
wanted: flow-confidence vs coherence on real motion (aperture-matched).

---

## Traversal matrix (rows x columns; cell = instance+gate / absent / ?)

| # | structure | rife | esrgan | dav2 | holo | diffusion | zoo |
|---|-----------|------|--------|------|------|-----------|-----|
| S01 | lattice codec | Y | Y | Y?layout | Y | ? | ? |
| S02 | bridge/scales | Y | Y | LOG-split? | Y | ? | ? |
| S03 | frozen LUT fns | Y(sigm) | ? | ? | Y(subst!) | silu? | ? |
| S04 | int conv | Y | Y | ? | Y(rep-split) | matmul? | ? |
| S05 | warp | Y(nihui) | ? | ? | Y((dy,dx)) | ? | ? |
| S06 | exact select | Y(prelu) | ? | ? | Yx4 | ? | ? |
| S07 | norms | ? | ? | ? | - | Y? | ? |
| S08 | attention | ? | ? | ? | - | ? | ? |
| S09 | residuals | Y | Y | Y? | Y | ? | ? |
| S10 | resample moves | Y(interp) | Y(shuf) | ? | Y(pad) | ? | ? |
| S11 | tensor+coh | ? | ? | ? | Y | ? | ? |
| S12 | oriented bank | ? | ? | ? | Y | ? | ? |
| S13 | beta tables | - | - | - | Y(v1-v5) | ? | ? |
| S14 | prior mult | - | - | - | Yx2 | ? | ? |
| S15 | router | - | - | - | Y(frame) | ? | ? |
| S16 | L2 handoff | Y(out) | Y(in) | Y(out) | Y(in) | ? | ? |
| S17 | hashed priors | - | - | - | Y | ? | ? |
| S18 | fit/freeze | ? | ? | ? | Yx3 | ? | ? |
| S19 | temporal IIR | ? | - | - | Y | no (DDIM parse 2026-09-30, see S19) | - |
| S20 | parity gates | Y(0-diff) | Y | ? | Y | ? | ? |
| S21 | causal streams | ? | ? | ? | cand | ? | ? |
| S22 | content corresp. | ? | ? | 0.27meas | ? | ? | ? |

(- = believed absent, which is also a claim and should be checked.)
(cand = measured candidate, not yet an instance; 0.27meas = method demo.)

## Questions for the owner (answer in prose, I'll fold in)

1. What's missing (your 15-20 vs my 20 -- which entries aren't structures)?
2. What merges (e.g. is S19 just S09? is log-domain S02 or S02b? matmul vs conv?)?
3. What splits (which FAMILY-SPLITs deserve full entries)?
4. Fill the ? cells you already know (especially diffusion/zoo/dav2 columns).
5. The first cross-model shape reuse -- which one was it, in your telling?
   (My candidates: bridge rife->esrgan, sigmoid substitution, select_mux.)
6. Where should this file LIVE (here as program map, or phi-core as framework
   law)? My vote: draft here, promote to phi-core when the ? count halves.
