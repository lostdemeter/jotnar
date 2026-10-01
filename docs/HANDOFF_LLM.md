# HANDOFF: building the designed LLM (v1.9 readiness assessment)

For a fresh instance (or a forgetful owner): can we start working on the
designed LLM in earnest? Answer below, per subsystem: READY (gated) or GAP
(with evidence + next step). No aspirations; only what the suite proves.

## Map (where everything lives)

- `docs/LANGUAGE.md` -- the language (46 mnemonics + tutorial + process).
  Start here; the stranger test proves it suffices (test_stranger.py).
- `docs/LLM_DESIGN.md` -- the formal design (anatomy table + decisions +
  three laws + plan). The blueprint.
- `programs/` -- listings (flagship, xf_block, mlp_* per family, assoc_mem,
  bigram_lm, scan_demo, attn_mini + _implant/_storebank variants).
- `chain/` -- asm.py (assembler), asm_ops.py (mnemonics), census.py,
  read.py (predictor), engram.py (storage), substrate.py (emission),
  qwen_mirror.py (boundary mirrors), wide.py (vendored bridge).
- `stdlib/` -- shared DEFs (attention, mlp, dirstore, storebank).
- `data/` -- frozen LM counts/vocab/manifest/test split.
- `chain/M.json`, `chain/CTRL.json` -- frozen scales/levels (doctrine).
- `test_*.py` -- 44 suites, all green at v1.6 (+2 since: test_anatomy
  grew GPT-2, test_asm grew SCAN/LAYERNORM/eps gates).

## Capability inventory (gated, with pointers)

| capability | status | evidence |
|---|---|---|
| matmul/softmax/norms/activations in lattice | READY | test_asm.py (0-diffs + parities) |
| transformer block end-to-end | READY | test_xf_block.py (84dB) |
| per-block scales + true-eps | READY | test_xf_block (65dB mixed) + eps law |
| whole-block on real weights (3 families) | READY | test_realw/mmlp (Qwen 56dB), test_anatomy (SmolLM2 49dB, GPT-2 LN-GS 58dB) |
| recurrence (SCAN + repeat threading) | READY | test_asm (bit-exact) + test_mamba (46dB real) |
| full-range attention | READY* | test_tshift.py (89dB); *awaits phi-core merge of ai/t-transform-bridge |
| read/modify/predict weights | READY | test_read/edits/probe (+0.2dB blind) |
| stores as data + emission | READY | test_store(bank)/emit (matrix green) |
| construct from counts | READY | test_lm.py (top1 0.41, speaks) |
| labeling loop | READY | S21 CONFIRMED (3 labels) + loop_turn.py |

## Feasibility per subsystem (the honest table)

| subsystem | verdict | evidence / next step |
|---|---|---|
| tokenizer (tiny, own data) | GAP (small) | word-level top-512 exists in freeze_lm.py; BPE-or-better unbuilt. Next: measure OOV pain on wider data, then decide. |
| embeddings (random, calibrated) | READY (pattern) | SmolLM2 probe measured magnitudes first; same procedure for own inits. |
| block (norm/attn/MLP listings) | READY | xf_block + mlp_* + attn_mini all run; RoPE/SwiGLU/RMSNorm chosen by design. |
| scales + eps per block | READY | goldilocks + eps laws gated; two-stage composition proven. |
| full-range attention | READY* | *needs the phi-core merge OR the vendored wide.py (already running). Decide at build start. |
| sampling/decoding | READY (greedy+topk) | demo_lm.py (seeded, gated determinism/diversity). Nucleus/top-p unbuilt, backlog. |
| eval (batteries + ppl) | READY (weak) | qa batteries inherited (Echion), ppl on counts. Comprehension eval = horizon. |
| depth compounding | GAP (known) | per-block budgets must SUM; first depth run decides (design says so). |
| fitting (beyond counting) | GAP (known) | fit/freeze loop handles ~dozens of params; anything more needs design (count-first doctrine). |
| training (gradients) | OUT | explicitly not planned; construction only. Say it plainly. |

## Overall verdict: GO with the gaps named

Everything with a READY above is runnable today; the two small GAPs
(tokenizer reach, depth compounding) are first-build discoveries, not
research blockers; training is out by design, not by omission. The one
external dependency: phi-core's ai/t-transform-bridge review (or keep
the vendored wide.py -- it is gated and running).

## First week (if I forgot everything, start here)

1. Read docs/LANGUAGE.md, run test_stranger.py (docs suffice: proven).
2. Run test_lm.py + python3 demo_lm.py (see the creature speak).
3. Read docs/LLM_DESIGN.md + docs/MODEL_READ.md (what's known).
4. Pick the narrowest domain slice; freeze counts (scripts/freeze_lm.py
   pattern); build block one (programs/mlp_qwen0.asm pattern).
5. Mirror alongside (chain/qwen_mirror.py pattern), gate parity, loop it.
