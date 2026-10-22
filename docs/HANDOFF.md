# Session handoff — Jotnar (main @ eba4a2c, all synced, 2026-10-22)

Read this first in a fresh session, then `roles/` (3 short files:
worker/integrator/flow). Nothing below is memory — verify the head
lines yourself (`git status --short` must be empty,
`git log --oneline -1 main` should say `eba4a2c`).

## Repo state (verified 2026-10-22)

- Repo: `/home/balrog/Documents/OpenCode/holographic_enhancement`,
  remote `origin` (`github.com/lostdemeter/jotnar.git`), gold `main`.
- `main` @ `eba4a2c` == `origin/main`; every branch local == origin;
  no stash; tree clean; GPU idle.
- Transport: HEALTHY (was a multi-day IPv6-blackhole outage; fixed
  via `AddressFamily inet` for github.com in `~/.ssh/config` — do not
  remove that block). If pushes stall again: `ssh -T git@github.com`
  first, IPv6 route prime suspect, HTTPS remote is the fallback
  (needs a PAT — none stored).
- Box: RTX 3090 Ti 24GB, shared + shared /tmp — one GPU session at a
  time. `source scripts/cuda_env.sh` before every CUDA command
  (user-local nvcc/libcublas; forgetting it breaks CUDA legs with
  `libcublas.so.13` errors — diagnosed 3 times, always the harness).
- Siblings (read-only unless tasked): `../phi-core`,
  `../dav2_revisited` (geometric DAV2 + webcam captures),
  `../marigold-v2` (teacher stack), `~/marigold-env` (teacher venv:
  torch 2.14+cu130, diffusers 0.38.0, scipy). Qwen-7B snapshot present
  (scripts/download_qwen7b.py); NEVER co-load with teacher (19.5GB).

## What Jotnar is (60 seconds)

Geometric assembly language, 63 mnemonics (F/T/I kinds), compiler
(`chain/`: assembler, builder, census, C/CUDA/integer-only-C
emitters) lowering real programs: Qwen2-7B decode at 27ms/tok
(1.125x HF), image pipelines, DPT-tiny executing deterministically
(77.6/84.8dB) with a live webcam demo (1.2s/frame, bannered noise).
Protocol M1–M6 (controller/node split, calculator/brain, 8 saturated
lanes on gimli). Test roster ~45 files, all green at every merge.

## Load-bearing facts (do NOT re-derive; each cost a hunt)

- Determinism: cublas workspace in emitted setup; bit-exact re-runs
  gated. Never loosen.
- Mask floor `QWEN_MASK_NEG=-1e9` in `chain/qwen7b.py`.
- GATHER static-rows rule; shadow CONVERT refresh after in-place
  writes; emitter reloads (never memsets) startup files.
- Stamp discipline `build.json` + `QWEN_REUSE_BIN=1` reuse-only.
- argv: inputs interleaved kP0,vP0…, outputs blocked all-K then all-V.
- Workdirs `/tmp/dec7b512` (REPL) + `/tmp/dec7b*` (tests); NEVER
  `/tmp/dec7b` (contaminated graveyard).
- `test_emit_c` fails on missing system gsl headers (pre-existing,
  environmental — verified identical on pristine main twice).
- Counts: README line says C36/CUDA37/NF31 over 63 mnemonics —
  re-verify operationally before trusting (drifted twice already;
  `asm-lang-coverage` gate is truth, not the prose).

## Open threads (in order)

1. **Ordered readout VOID follow-up:** stabilized-readout training
   (affine-blindness diagnosis banked in Phase-30 spec section).
2. **Lesion-matched follow-ups:** basin-C directional move replicated
   bit-identically (only positive in 30 rounds) — mechanism chase.
3. **DPT speed:** burn-down ordered (resident serving → MATMUL-led
   21x); profiler exists (`chain/geo_profiler.py`, `--budget` flag).
4. **Constraint debate:** priced in `docs/PATTERN_GAPS.md` (A5/A6/B-track,
   VAE-grade) — needs user signature, never worker prompts.
5. **Raw scale (10M+ params):** never swept. Textured-domain refined
   verdicts. Ordinal probe (unrun — audits all verdicts if positive).

## Standing rules (both roles, no exceptions)

Worker implements on branches + reports with pastes (never merges);
integrator verifies-by-running + merges `--no-ff` + smokes + pushes.
On any error: stop and report — no `reset --hard`, no force-push, no
rebasing pushed branches. Never merge red (verdict-reported form for
honest reds — Phase-26 precedent). History append-only; `research`
branch never merges. `git ls-files` proof for binaries (status is
blind to gitignored files — Phase-29 lesson). Banked numbers must
postdate final-tree code (Phase-4 lesson). Prompt-drafting rule:
schema-check banked artifacts + cited file:lines + base tip BEFORE
drafting (`roles/integrator.md`).
