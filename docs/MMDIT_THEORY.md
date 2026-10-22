# MMDiT theory — founding block mapped to our failures (read-only)

Acquisition (documented): (1) pip-installed `denoising-diffusion-pytorch`
2.3.4 (+ einops/ema-pytorch/pytorch-fid) into `~/marigold-env` — additive
only, no existing package touched; searched, NO MMDiT source inside
(discarded path, stated). (2) Cloned `github.com/lucidrains/mmdit` as
sibling `../mmdit-ro` @ `76a5846`, MIT licensed, set read-only
(`chmod -R a-w`; zero sibling imports, zero writes). Primary source:
`mmdit/mmdit_pytorch.py` (422 lines: JointAttention L48–165,
MMDiTBlock L169–350, MMDiT L354–422); secondary:
`mmdit/mmdit_generalized_pytorch.py` (250 lines: AdaptiveLayerNorm
L38–77, N-modality block L81–204). NOT vendored into our tree.
Second reading: Esser et al., arXiv:2403.03206 (SD3 report), §2 (CFM
objective Eq.8), §3.1 (logit-normal/mode/cosmap timestep densities),
§4 + Fig.2 (MM-DiT: separate weights, joint attention, DiT modulation,
QK RMS-Norm optional), §5.3.2 (QK-normalization), Table 3 (d=16 VAE).
No building, no training — a line of implementation is scope breach.

## Block diagram in words (founding block)

Two streams (text, image) with SEPARATE QKV linears per modality
(`to_qkv` ModuleList, L76) and SEPARATE out-projs (L89); per-modality
masks default all-true; pack-concat (L143–144) → ONE `Attend`
(L150) → unpack (L155) → separate out-proj (L161–163). Optional
per-modality MultiHeadRMSNorm on q,k only (L91–97, flag default False
L55 — and the BLOCK never passes it, L235–240). Conditioning:
time-cond → SiLU → Linear with WEIGHT+Bias ZEROED and gamma-bias ONES
(L221–223, AdaLN-Zero) → per stream 4 pre/post gammas (attn/ff) +
PRE-ONLY betas (L200–211); norms go affine-FREE when conditioned
(L227–231, no double-affine); applied x·γ+β pre, ×γ post (L281–282,
L294–295). Text stream SKIPS its FF on the last block by default
(L252, L332). Transformer holds learnable IMAGE-side register tokens
(default 0, L360–371; prepend L397, strip L415) and defaults to
4-stream HyperConnections (L362) while blocks default to plain Residual
(L187); final RMSNorm on IMAGE stream only (L384, L420). Logit
softclamp exists but defaults off (L57–58). Generalized file restates
modulation cleanly (AdaptiveLayerNorm L38–77; post-branch-only gamma
variant L115–123) and lifts the block to N modalities (L84).

## Per-mechanism failure mapping (status: tabled / parked / absent-with-price)

| # | Mechanism (source cite) | Banked failure it addresses (round cite) | Table status (rows, never memory) |
|---|---|---|---|
| M1 | Joint attention, separate QKV/out per modality (mmdit L76/L89/L143–163; Esser §4 "two separate sets of weights... joining the sequences") | Mixer wall 0/156 (`STUDENT.md:503`, `PATTERN_GAPS.md:193`); joint-attention 0/12 (`STUDENT.md:585`) — our trunks share everything, Esser says modalities need own space + joint mixing | Tabled-as-composition: BMMV `:169` + SOFTMAX_WIDE `:205` + CONCAT `:171` (`stdlib/attention.asm:7-18`); fused kernel absent-with-price A4/M, DEFERRED with evidence (`PATTERN_GAPS.md:123-129`, unfused wins 0.31x/0.50x) |
| M2 | Per-modality QK-RMSNorm, optional, default OFF (mmdit L55/L91–97, block never passes it L235–240; Esser Fig.2 "Opt. RMS-Norm... can be added to stabilize") | Attention wall 0/36 (`STUDENT.md:340`); teacher HAS QK-norms frozen-no-LoRA (`STUDENT.md:430`) — do our DiT students even wire it? (audit question, stated) | Tabled: RMSNORM `:197` (C+CUDA+T). Price S (wiring audit only — op exists). Two-source: Esser-optional + lucidrains-off-by-default + teacher-present-frozen |
| M3 | Per-stream AdaLN-Zero: zero-init cond-linear (L221–223), 4 gammas + pre-only betas (L200–211), affine-free norms (L227–231), post-branch gates (L294–295) | AdaLN-blindness (`STUDENT.md:1771-1790`: loss blind to modulation ×1e5–1e6 blowup; fix space named :1790 "modulation budgets"/:1819 "vars < 1e2"); teacher mod-producers frozen (`STUDENT.md:430`); our students run STATIC const modulation, no timestep/text (`STUDENT.md:1233,1567`) | Composes from tabled ops (MATMUL `:189`, SILU `:202`, MUL `:191`, ADD `:159` — QWEN_ARCH precedent); modulation BUDGETS absent-with-price S (CLIP `:170` parked → unpark, or SELECT `:200` compose) |
| M4 | Register tokens, image-side, default 0 (L360–371, prepend/strip L397/L415) | Shuffle-collapse Phase 14 (`STUDENT.md:904-973`: teacher 1.27 collapse on patch-shuffle, locals flat — NEEDS JOINT REASONING; `tests/test_shuffle_probe.py`) — registers are dedicated global-summary slots that survive permutation | Absent-with-price S: CONCAT `:171` + BETA consts `:167` + strip logic — plumbing only, ZERO new patterns |
| M5 | skip_feedforward_text_tokens, default True (L252/L304/L332) | NONE (our depth students are unconditional — no text stream exists to skip) — documented for completeness, not ranked as theft | N/A (a deletion, not an op). Price zero |
| M6 | HyperConnections, transformer-default 4 streams (L362; generalized: manifold-constrained variant) | Fusion/interior-tap demand (A5 territory, `PATTERN_GAPS.md:131-148`; `docs/FUSION.md`) — learned depth-mixing vs our fixed residuals | Absent-with-price M/L (protocol design bulk; advisor exists). Inherits A5's falsifier, stated |
| M7 | Final RMSNorm, image stream only (L384/L420) | CONFIRMATION, not theft: our norm_out.head harvest point (prototypes round) sits exactly here — validates the harvest choice post-hoc | Tabled: RMSNORM `:197`. Price zero |
| M8 | Logit softclamp, default off (L32–33 helper, L57–58 flag) | Phase-25 telemetry blowup (acts 1.4→6.6e8; QWEN_ARCH fp16 ±65504 clip guards) — clamp WHERE it matters (logits, pre-softmax) | Absent-with-price S: no TANH row (template op like DIV `:174`, or LUT like SIGMOID `:201`); composes otherwise (MUL/DIV) |
| M9 | AdaptiveLayerNorm clarity form + post-gamma-only variant (generalized L38–77, L115–123) | Same AdaLN-blindness as M3 — cleaner reference shape for the budget fix (γ-bias 1 / β-bias 0 init, L61–62) | Same as M3 |
| M10 | Rectified-flow framing: velocity regression (Esser Eq.8) + logit-normal timestep weighting §3.1/Tab.1 (intermediate scales matter; rf/lognorm(0,1) wins) | Bar-blind schedule sweeps (`STUDENT.md:634,709,850` — schedules moved, bars never did; no round ever swept noise weighting) | Price ~ZERO: host-side timestep sampling (sampling-on-host precedent, README contracts) — no patterns at all |
| M11 | Separate-weights doctrine (Esser §4: "two independent transformers... joining the sequences") | Mixer wall 0/156 + joint 0/12 (same as M1, training-scale reading: two-stream STUDENT at equal params vs shared trunk) | Config + training, not patterns — price M/L. Two-source with M1 |
| M12 | 16-channel VAE + frozen encoders (Esser Table 3 d=16; §4 frozen text models) | CONFIRMATION: our VAE z_dim=16 matches (teacher blueprint); text-encoder-never-loaded already our doctrine | A3 CONV scope unchanged (M). Price held |

## Ranked steal-list (failure-addressed × price; each with its falsifier)

1. **Register tokens (M4)** — hard failure (shuffle-collapse, teacher-only
   behavior our models lack) × S plumbing (zero new patterns). Falsifier:
   shuffle-delta WITH registers ≈ WITHOUT (Phase-14 harness rerun) → kill.
2. **Logit-normal timestep weighting (M10)** — medium failure (bar-blind
   schedules; never-swept dimension) × ~zero price (host-side sampling).
   Falsifier: uniform-vs-lognormal A/B arm on a live student shows no
   delta → kill.
3. **Modulation budgets + zero-init discipline (M3/M9)** — hard failure
   (AdaLN-blindness blowup ×1e5) × S (CLIP-unpark or SELECT-compose +
   init change). Falsifier: blowup persists WITH budgets, or mod-vars
   stay degenerate → kill (fall back to static-const, priced).
4. **QK-RMSNorm wiring audit (M2)** — medium failure (attention wall;
   teacher-frozen vs student-unknown) × S (op tabled; audit first).
   Falsifier: students already wire it, or adding changes nothing → kill.
5. **Logit softclamp (M8)** — medium failure (Phase-25 blowup) × S (tanh
   template/LUT). Falsifier: telemetry shows blowup in VALUES not logits
   → kill (clamp mistargeted; budgets M3 cover instead).
6. **Two-stream student trunk (M1/M11)** — hard failures (0/156 + 0/12) ×
   M/L (training-scale build). Falsifier: equal-param two-stream arm ties
   shared trunk → kill (sharing exonerated).
7. **HyperConnections (M6)** — soft failure (fusion demand unproven) ×
   M/L. Falsifier: A5's existing falsifier fires first (zero interior-tap
   chains) → never open.
8. **skip-text-FF (M5)** — NOT ranked (non-applicable, unconditional
   students). Documented so nobody re-derives it.

## Unknowns + retiring experiments

- Whether OUR DiT students wire QK-norm (M2 audit: read the student
  listings — minutes, no runs).
- Register-strip interaction with our ax1 CONCAT doctrine (M4 design
  question for the build round, not this one).
- Whether logit-normal weighting interacts with our fixed 16k-step
  budgets (M10 A/B arm IS the falsifier — same experiment).
- Esser §5.3.2 QK-norm-at-scale details (beyond the Fig.2 caption read
  here — deeper read only if M2 audit says our students lack it).
- tanh-pattern pricing for M8 (template vs LUT — builder's call at build
  time; priced S either way).

Gate: `tests/test_mmdit_theory.py` (CPU seconds): source-integrity
(sibling paths + rev + key symbols; README rows; banked-round files),
mapping-completeness (M1–M12 each has a status + cite), tripwire (no
MMDiT/register/modulation implementation files in our tree; emitter
files untouched). Neighbors: none touched; `test_asm.py` green.
