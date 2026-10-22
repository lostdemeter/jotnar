# QWEN_ARCH — Qwen Image Edit architecture survey (read-only)

Order honored: local venv first, internet second. Read-only everywhere
(venv/sibling trees/packages — ran nothing that trains, modified nothing
anywhere). No emitter/opcode/assembly changes. No new dependencies. No
downloads beyond public web pages (no model/dataset fetches — everything
needed is installed or online-read). Production weights untouched (never
loaded — architecture, not weights, is the target). CPU only (reading +
numpy shape arithmetic — no GPU need exists). Qwen-7B never co-loaded.

Venv: `~/marigold-env` (diffusers 0.38.0 — version matters, see A2 delta).
Primary source: `lib/python3.12/site-packages/diffusers/models/
transformers/transformer_qwenimage.py` (993 lines).

## Block diagram in words

Conditioning (dual path): input image → Qwen2.5-VL-7B (semantic) AND VAE
encoder (appearance/reconstructive); text prompt → Qwen2.5-VL with
task-specific system prompt; last-layer hidden states = guidance latent.
Backbone: 60× dual-stream MMDiT blocks, inner_dim 3072 (24 heads × 128),
joint dim 3584. Per block, image stream + text stream each get:
LayerNorm(affine=False) → AdaLN-style modulation (SiLU+Linear temb→6×dim,
split scale/shift/gate × norm1/norm2; x·(1+scale)+shift, gated residuals)
→ JOINT attention (separate QKV linears per stream, per-head RMSNorm QK on
both streams, 3D RoPE on both, concat [text,image] → single SDPA →
split → separate out-projs) → second modulated LayerNorm + MLP
(Linear→GELU→Linear) → gated residual. Time: sinusoidal Timesteps(256,
flip_sin_to_cos, scale 1000) → TimestepEmbedding MLP (Linear→SiLU→Linear)
= temb (3072). Positions: 3D RoPE, axes (16,56,56) = 128 = head_dim,
theta 10000, scale_rope=True (centered); text treated as 2D-diagonal
(MSRoPE per report). Edit: noisy latents + VAE image_latents CONCATENATED
along seq dim; img_shapes carries [[noisy-FHW],[condition-FHW]] per batch;
Layer3D rope indexes the condition layer at −1 (distinct band, not
overwrite). Output: AdaLayerNormContinuous + Linear proj to
patch²·out_channels. fp16 clip guards (±65504) per block.
VAE: 3D-causal conv ResNet encoder (shared image/video) + dual decoders
(image/video); RMS_norm (custom channel-first); nearest-exact 2× upsample
+ conv; 1×1-conv attention blocks; z_dim 16, base 96, 8×8 compression.
Name note: code has BOTH QwenEmbedRope (line 199) and
QwenEmbedLayer3DRope (line 341); the Edit pipeline's layered img_shapes
+ condition-freq path is the Layer3D variant's documented purpose
(code-only single-source — checkpoint config not fetched).

## Per-component mapping (status: tabled/parked/absent + cite)

| # | Component (venv cite) | Kind / placement | README pattern table |
|---|---|---|---|
| 1 | img_in/txt_in Linear (`transformer_qwenimage.py:799-800`) | MATMUL+bias | tabled — MATMUL `README.md:183`, BETA `README.md:167` |
| 2 | txt_norm RMSNorm(3584) (`:797`) | pre-input norm | tabled — RMSNORM `README.md:191`, CUDA `chain/emit_cuda.py:130` |
| 3 | time_text_embed: Timesteps sinusoid + TimestepEmbedding Linear-SiLU-Linear (`:174-197`, `embeddings.py:1261`) | time conditioning | tabled — SILU `README.md:196` + MATMUL; sinusoid emits via TRANSPOSE-free const math (composes, no pattern) |
| 4 | pos_embed 3D RoPE, axes (16,56,56), scale_rope (`:199-341`, `:788-794`) | positions | tabled — ROPE `README.md:164` / ROTARY `README.md:192`, CUDA `chain/emit_cuda.py:332,461`, C `chain/emit_c.py:422` |
| 5 | img/txt_norm1+norm2 LayerNorm(affine=False,eps 1e-6) (`:613,629-632`) | per-block norms | tabled — LAYERNORM `README.md:181` (A1 built) |
| 6 | img/txt_mod SiLU+Linear(dim→6dim) (`:609-612,634-637`) | modulation proj | tabled — SILU+MATMUL (same rows) |
| 7 | _modulate x·(1+scale)+shift, gate×resid (`:644-681`) | elementwise | tabled — MUL `README.md:185`, ADD `README.md:159` |
| 8 | QKV + out linears per stream (`:490-560` processor) | projections | tabled — MATMUL/BMMV (`README.md:183,169`) |
| 9 | QK RMSNorm per head BOTH streams incl. added (`attention_processor.py:207-209,278-288`, block `qk_norm="rms_norm"`) | QK-norm | tabled — RMSNORM `README.md:191` |
| 10 | joint SDPA concat[text,image] (`:545-560`, `dispatch_attention_fn`) | attention math | tabled as unfused composition — BMMV+SOFTMAX_WIDE (`README.md:169,199`), `stdlib/attention.asm:7-18`; fused encoder kernel absent (tripwire `tests/test_fused_defer.py`, A4 deferred) |
| 11 | img/txt_mlp FeedForward(gelu-approximate) (`:628,638` → `attention.py:1714-1715` → `activations.py:65-91` GELU tanh) | MLP activation | PARKED — GELU `README.md:178`; composes NOT (tanh ≠ SILU-form; see A2 delta) |
| 12 | norm_out AdaLayerNormContinuous (`:820`) + proj_out Linear (`:821`) | head | tabled composition — norm `README.md:181` + MUL/ADD; AdaLN fused pattern absent (composes, none needed) |
| 13 | VAE causal-3D convs + ResNet (`autoencoder_kl_qwenimage.py:40,220`) | conv | partially tabled — CONV `README.md:172` covers single-ch 2D same/stride-1 ONLY; 3D/causal/strided/bias gaps named |
| 14 | VAE QwenImageRMS_norm (`:85`) | norm variant | tabled-math, variant gap — RMSNORM row holds math; channel-first+images parametrize unpriced |
| 15 | VAE upsample nearest-exact (`:155,160`) | upsample | degraded-default covers shape — INTERP `README.md:179` is bilinear; nearest stays shipped degraded (P2 doctrine); fidelity gap named, no pattern |
| 16 | VAE 1×1 attention (`:294`: `to_qkv` Conv2d 1×1, proj 1×1) | attn-via-conv | tabled composition — 1×1 conv = MATMUL-class + SOFTMAX_WIDE |
| 17 | text encoder Qwen2.5-VL-7B (`pipeline_qwenimage_edit.py:176-178,191`) | conditioner | absent — not a pattern (never loaded this round; out of compiler scope by design) |
| 18 | Edit seq-concat + layered shapes (`pipeline_qwenimage_edit.py:762`, `:749-752` latent cat) | conditioning | tabled — CONCAT `README.md:171` (ax1 seq-concat) |

Shape probes (numpy arithmetic, no weights): inner 24×128=3072;
mod out 6×3072=18432; rope axes 16+56+56=128=head_dim; patch 2;
VAE z 16 / base 96 / 8×8 factor; MMDiT 60 layers (code `:770-779`
× report Table 1 — two-source).

## Internet sources (second) — training recipe, VAE, code-silent items

- HF card `Qwen/Qwen-Image-Edit` (fetched): 20B, BF16, Apache-2.0,
  `QwenImageEditPipeline`, arXiv 2508.02324; dual path stated (image →
  Qwen2.5-VL semantic + VAE appearance) — two-source with report
  abstract + code concat path. Single-source: 20B count (code gives
  shapes, not the total).
- HF card `Qwen/Qwen-Image` (fetched): same backbone family,
  true_cfg_scale 4.0 usage, aspect tables — single-source usage items.
- Tech report arXiv:2508.02324 (abs + HTML §2-§4, fetched — cite, not
  re-derive): §2.1-2.4 architecture (double-stream MMDiT; QK RMSNorm +
  all-other LayerNorm — two-source with code; MSRoPE diagonal-text —
  two-source with Layer3D code path; VLM 32+28 / VAE enc 11 dec 15 /
  MMDiT 60 Table 1 — two-source with code MMDiT column); §2.3 VAE
  (Wan-2.1 encoder frozen + image decoder fine-tuned on text-rich
  corpus; recon+perceptual only, adversarial dropped; grid-artifact
  note — SINGLE-SOURCE, code-silent); §4.1 flow-matching (rectified-flow
  velocity MSE, logit-normal t — single-source); §4.1.3 curriculum
  (low→high res, non-text→text, massive→refined, unbalanced→balanced,
  real→synthetic — single-source); §4.3 multi-task (T2I+TI2I+I2I
  reconstruction aligning Qwen2.5-VL↔MMDiT — two-source with dual-path
  code); §4.1.1-4.1.2 infra (Producer-Consumer/TensorPipe/Megatron —
  single-source, irrelevant to patterns).
- Marigold paper §3: already read (prior round) — not re-derived.

## Deltas to prior gap pricings (prices unchanged where unaffected)

- **A2 GELU: KILLED → REOPENED (version-dependent).** PATTERN_GAPS
  `docs/PATTERN_GAPS.md:62-71` killed A2: audited sig-1.702
  (ApproximateGELU) composable from SILU at 314.6dB. Installed
  diffusers **0.38.0** routes `gelu-approximate` → `GELU(approximate=
  "tanh")` (`attention.py:1714-1715` → `activations.py:65-91`), NOT the
  sig-1.702 class (`activations.py:149-169`). Tanh-GELU does not compose
  from SILU. Price UNCHANGED (S, `PATTERN_GAPS.md:59`); verdict flips
  for this venv: DiT MLPs need an erf/tanh pattern after all. Falsifier
  for re-kill: parity leg showing sig-1.702 ≈ tanh within 40dB at live
  pre-activation magnitudes (the 38.4dB number says it fails — rerun
  against 0.38.0 to confirm, not assumed).
- **DiT verdict: "zero further patterns" → +A2(S).**
  `docs/PATTERN_GAPS.md:328-342` opened faithful-DiT with zero further
  patterns (LayerNorm built + GELU composed + unfused attention). With
  tanh-GELU, the faithful path costs one S pattern. Everything else in
  that verdict stands (unfused wins 0.31x/0.50x, LayerNorm landed).
- A1 LayerNorm: NO DELTA — survey confirms all block norms are
  affine-free LayerNorm (built) and QK is RMSNorm (tabled). Price/built
  status held.
- A4 fused attention: NO DELTA — joint SDPA via dispatch (unfused math
  = BMMV+SOFTMAX_WIDE composition) matches the deferral evidence.
- A3 CONV: price HELD (M), scope gap WIDENED — v0.1 covers 2D
  single-channel same/stride-1; VAE needs causal-3D + strided + bias +
  zero-pad downsample (`autoencoder_kl_qwenimage.py:40,133-218`).
  VAE branch stays M-minimum, not S.
- P2 INTERP: NO new pattern — VAE upsample is nearest-exact (code
  `:155,160`), which is the shipped degraded default, not bilinear.
  Fidelity gap named (nearest vs bilinear), price unchanged (nothing).
- B-track / A5 / A6: unaffected (no control-flow or fusion interiors
  found in the surveyed paths).

## Unknowns + the experiment that retires each

1. GELU live-magnitude parity (tanh vs sig-1.702 at real DiT
   pre-activations on 0.38.0): activation-histogram + 40dB parity leg
   (the re-kill falsifier above).
2. Edit checkpoint flags (use_layer3d_rope / zero_cond_t / additional
   t_cond as shipped in `Qwen/Qwen-Image-Edit` config): read the model
   `config.json` (no weights) — code supports all three, checkpoint
   picks; single fetch of a JSON, not weights.
3. VAE loss-mix claim (recon+perceptual, no adversarial) + decoder-only
   fine-tune: report-only; retire by VAE recon benchmark (§5.2.1)
   reproduction — out of scope for patterns (no experiment proposed
   here).
4. Scheduler/shift numbers (base 256 / max 4096 / 0.5→1.15,
   `pipeline_qwenimage_edit.py:68-80`): code-read; retire by a
   CPU shape probe of `calculate_shift` (seconds, no weights).
5. Text-encoder prompt templates' effect on hidden-state geometry:
   retire by dumping Qwen2.5-VL last-layer hists — needs weights+GPU,
   explicitly NOT run (stated, not hidden).

Gate: `tests/test_qwen_arch.py` (CPU seconds): source-integrity
(every cited file/symbol exists at stated path), mapping-completeness
(every surveyed component has a status + cite), no-implementation
tripwire (no DiT/VAE implementation files added). Neighbors: none
touched; `test_asm.py` green as sanity.
