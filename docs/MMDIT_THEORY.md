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
| M4 | Register tokens, image-side, default 0 (L360–371, prepend/strip L397/L415) | Shuffle-collapse Phase 14 (`STUDENT.md:904-973`: teacher 1.27 collapse on patch-shuffle, locals flat — NEEDS JOINT REASONING; `tests/test_shuffle_probe.py`) — registers are dedicated global-summary slots that survive permutation | Absent-with-price S: CONCAT `:171` + BETA consts `:167` + strip logic — plumbing only, ZERO new patterns (all ops tabled) |
| M5 | skip_feedforward_text_tokens, default True (L252/L304/L332) | NONE (our depth students are unconditional — no text stream exists to skip) — documented for completeness, not ranked as theft | N/A (a deletion, not an op). Price zero |
| M6 | HyperConnections, transformer-default 4 streams (L362; generalized: manifold-constrained variant) | Fusion/interior-tap demand (A5 territory, `PATTERN_GAPS.md:131-148`; `docs/FUSION.md`) — learned depth-mixing vs our fixed residuals | Absent-with-price M/L (protocol design bulk; advisor exists). Inherits A5's falsifier, stated |
| M7 | Final RMSNorm, image stream only (L384/L420) | CONFIRMATION, not theft: our norm_out.head harvest point (prototypes round) sits exactly here — validates the harvest choice post-hoc | Tabled: RMSNORM `:197`. Price zero |
| M8 | Logit softclamp, default off (L32–33 helper, L57–58 flag) | Phase-25 telemetry blowup (acts 1.4→6.6e8; QWEN_ARCH fp16 ±65504 clip guards) — clamp WHERE it matters (logits, pre-softmax) | Absent-with-price S: no TANH row (template op like DIV `:174`, or LUT like SIGMOID `:201`); composes otherwise (MUL/DIV) |
| M9 | AdaptiveLayerNorm clarity form + post-gamma-only variant (generalized L38–77, L115–123) | Same AdaLN-blindness as M3 — cleaner reference shape for the budget fix (γ-bias 1 / β-bias 0 init, L61–62) | Same as M3 (tabled composition) |
| M10 | Rectified-flow framing: velocity regression (Esser Eq.8) + logit-normal timestep weighting §3.1/Tab.1 (intermediate scales matter; rf/lognorm(0,1) wins) | Bar-blind schedule sweeps (`STUDENT.md:634,709,850` — schedules moved, bars never did; no round ever swept noise weighting) | Price ~ZERO: host-side timestep sampling (sampling-on-host precedent, README contracts) — no patterns at all (nothing to table) |
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

## Register tokens — the top steal, built and judged (constitution, FROZEN pre-run)

Build + train ONLY what is frozen here (no emitter/opcode/assembly
changes — gap hit = stop-finding).
Register count frozen: 4 (ONE count, no sweep — count is the variable's
shadow). Reason, two strands: (a) the banked host recipe already runs
nctx=4 (`attn_run_jattn64b2-8k-cos.json`: width/blocks/stride/nctx =
64/2/32/4) and the same-recipe rule forces it; (b) DINOv2's 4-register
precedent (single-source public knowledge, marked as such).
Host frozen: jattn attention (registers join the joint sequence as
attention participants — the MMDiT design: prepend, attend, strip,
`train_attn_student.py:88-138` `ctx`/`encode`/`head`). mscale REJECTED
with reason: token-free per-pixel MLPs reduce "registers" to per-pixel
const appends = Phase-6 glob rerun (`STUDENT.md:307-330` globw64d2,
0/12 banked) — a strawman that tests nothing MMDiT. The price (slower
training) is the cost of testing the actual mechanism.
Registers as const-fill + CONCAT (Phase-6 global-summary precedent:
`STUDENT.md:307-330` coarse-global broadcast; `train_student.py:78`
const-fill + CONCAT both tabled — cited; no new machinery). Learned
`ctx` Parameter in training (= MMDiT registers, `mmdit L369-371`);
const-fill is the DEPLOYED form (BETA `:167` + CONCAT `:171`, tabled).
Training frozen — verbatim `jattn64b2-8k-cos` recipe: --width 64
--blocks 2 --heads 1 --stride 32 --feat rgb --steps 8000 --lr 0.01
--schedule cosine --seed 7 --train church dogs squirrel train
--heldout geo_webcam_0_combined geo_webcam_1_combined. The ONLY delta:
--nctx 4 (registers arm, tag jattn64b2-reg) vs --nctx 0 (ablation arm,
tag jattn64b2-noreg) + tag/outdir. Retrain-match rule (frozen):
registers arm strata cells vs banked `attn_run_jattn64b2-8k-cos.json`
(6 frames × 6 strata) within abs 0.01/cell AND heldout mean within 5%
of banked 0.4062 (GPU-training tolerance ceiling, stated — tighter if
the rerun lands tighter, reported exactly; tighter never loosened).
Winner rule — primary (shuffle-robustness, Phase-14 acceptance):
registers-arm mean_rel@p16 (36 cells: 6 frames × 6 strata, seed 7,
rel = raw/std(intact), `shuffle_probe.py:107-113` metric verbatim)
> control-arm mean_rel@p16 (ablation proves the delta is the registers)
AND ≥ 1.0 (teacher-scale unity: teacher 1.2742 banked, target ~1.2,
locals ~0.2-0.8 — derived, not just stated). Secondary (no silent
magnitude trade): held-out pass count ≥ banked 0/12 AND held-out mean
≤ 0.4062 × 1.05. Empty (robustness flat, bars flat-or-worse) is
complete. Shuffle eval mirrors the probe's student path exactly
(CPU weights, intact + shuffled seed-7 p16/p64, unshuffle, banked
teacher numbers reused read-only for the ~1.2 reference — no teacher
reruns).
Ruling: registers + training only; weights out-of-tree
(`~/student_runs/jattn64b2-{reg,noreg}/`, metrics + configs banked
separately in `samples/marigold/registers_shuffle.json`, primaries
untouched); production weights untouched (teacher AND Qwen-7B never
loaded — fixtures suffice, no new teacher runs); GPU box to itself
for training runs (nvidia-smi pre-run: zero compute procs, 487MiB
display-only, 23.6GB free — stated); no downloads; no new deps
(marigold-env stack, CLEAN env — `cuda_env.sh` poisons torch runs,
established).
Gate: EXTEND `tests/test_shuffle_probe.py` in place (register legs:
schema + determinism + shuffle-robustness recomputed from banked
numbers vs frozen bar + bars legs + ablation-equality legs proving
the no-registers control matches its banked baseline AND the
registers arm matches banked 8k-cos per the retrain-match rule;
existing legs untouched). Neighbors: shuffle + strata + teacher
green (nothing moves underneath).

### Register results (TBD — training lands here)

### Register results (2× ~380s GPU trains + CPU evals, fixtures only)

Arms (frozen jattn64b2-8k-cos recipe verbatim; ONLY delta --nctx):
reg = nctx 4 (tag jattn64b2-reg, weights
`~/student_runs/jattn64b2-reg/weights.pt`, train wall 377s);
control = nctx 0 (tag jattn64b2-noreg, wall ~392s). GPU solo both
(zero compute procs pre-run; evals CPU, deterministic — rescue eval
reproduced training-print cells exactly, e.g. church far 0.6588).

Robustness table (mean_rel over 36 cells each, seed 7):

| system | p16 | p64 | vs teacher 1.27 |
|---|---|---|---|
| teacher (banked ref) | 1.2742 | 1.2143 | — |
| control nctx=0 | 1.3382 | 1.2261 | ABOVE teacher |
| registers nctx=4 | 1.0324 | 0.9545 | below control |

Bars table (held-out geo frames, Phase-1 bars):

| arm | held-out pass | held-out mean | vs banked 0.4062 |
|---|---|---|---|
| banked 8k-cos (ref) | 0/12 | 0.4062 | — |
| control nctx=0 | 0/12 | 0.4042 (−0.5%) | hold |
| registers nctx=4 | 0/12 | 0.4150 (+2.2%) | hold (≤5%) |

Retrain-match: VOID, not failed (model mismatch, evidence-stated):
arm nparams 157,249 vs banked 108,033 (Δ = 49,216 = dino_proj
768×64+64 exactly — banked run predates Phase-15; its JSON lacks
feat/heads keys). 24/36 cells breach abs-0.01 — expected across
different models, not a reproduction failure. Internal control
(same code, same box) carries validity instead.
Provenance incident (disclosed): the trainer banks fixed-name
`attn_run.json` — it overwrote the TRACKED jattn64b2 primary twice
mid-round. Restored byte-identical via `git checkout` (status clean
after; arm cells live only in `registers_shuffle.json` + out-of-tree
weights). Primaries untouched at rest.
Verdict: **registers teach shuffle-INVARIANCE — the top steal dies,
inverted.** Primary bar fails directionally (1.03 < 1.34; ≥1.0 alone
holds — reported, not rounded up); bars hold both arms. Mechanism,
stated as hypothesis (not asserted): ctx consts ride every forward
identically under permutation — a shuffle-invariant pathway that
DILUTES position-dependent response; removing them forces all signal
through position-bound RGB tokens → maximal shuffle response. Against
noise (noted, not hidden): single run per arm (frozen — no sweep);
but strata means agree ±2–3% while shuffle differs ±30% — noise does
not dissociate like that. Bigger finding, free: the 8k-cosine recipe
ALREADY reaches teacher-scale shuffle dependence (1.34 > 1.27)
WITHOUT registers — the host, not the theft, holds the structure.
The MMDiT steal-list survives minus its king (M10 schedule, M3
budgets, M2 audit, M8 clamp, M1/M11 trunk, M6 all unaffected —
stated, not re-ranked here).
Gate: `tests/test_shuffle_probe.py` extended in place (registers-*
key: schema, determinism, robustness-vs-bar, bars, ablation-equality
with the VOID documented, CPU spot 0.0e+00; all old legs green).
Neighbors: shuffle + strata (84 OK, zero : FAIL) + teacher green
(read-only reuse — nothing moves underneath).
Bugs caught (all mine): (1) eval script ran over ALL manifest
fixtures incl. synth .npys of foreign shape (lstsq death) — fixed
to the frozen 6 frames, nothing banked under it. (2) int-vs-string
JSON keys twice in the gate draft (p16/16) — failed loud, fixed.
(3) the attn_run.json overwrite above — caught by `git status`,
restored, disclosed. Gaps named: run-to-run variance scale (one run
per arm, frozen); dino-feat arms (different information plumbing —
not this round); register-count sweep (explicitly refused by freeze).
