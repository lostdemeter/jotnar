# Listing debugger — breakpoints, watch, step, lesion tables

## Constitution (FROZEN pre-build — violations bounce)

Surface (frozen — `chain/debugger.py:Session`):
- `break_at(indices)`: op positions (0-based into the bound program —
  listing text lines shift under comments; positions don't). Bad
  index refuses loud.
- `watch(names)`: stream names, validated eagerly against the census
  stream set (unknown refuses loud at watch time, never a silent
  missing at stop time).
- `step(n=1)`: advance n ops unless a breakpoint hits first; returns
  the stop record (pos/op/line/watched values). Stepping PAUSES the
  existing op loop — no reimplementation of execution: a store_hook
  routes writes to debugger-owned env (value-identity proven by the
  hook's own contract: ChainMap env-first reads execute identical
  values) and raises a private pause at the stop op; each step is a
  fresh bounded run (O(n²), stated — the debugger is for small
  listings; served/CUDA stepping explicitly out of scope).
- `lesion(op_idx)`: zero-Out intervention through the EXISTING
  overrides mechanism (built for probes — cited, not reinvented):
  the op's OUT streams replaced by zeros_like post-production,
  downstream reads the zeros; tabled as per-OUT-stream
  (maxabs, mean) deltas vs the reference run. Unknown op refuses
  (overrides' own never-assigned tripwire backs it).
- `diff(other)`: run-vs-run byte-compare (exact array equality incl.
  dtype/shape) with first-divergence reporting (op index, stream,
  maxabs) or identical.
Read-only w.r.t. listings (assemble + run only — a mutation
primitive is scope breach). CPU-first, lattice listings.
Determinism: same debugging session twice → identical tables
(byte-digests; flagship 24×32 verified 3/3 bit-identical
pre-round, so strictness is earned, not hoped).

## Lesion semantics

A lesion answers "what does the output owe this op": zero its
OUTs, rerun, diff the final OUTs. Zeros are zeros_like the
reference values (per-component for triples tuples) — same
shapes/dtypes/kinds, so kind checks pass and only VALUES change.
Deltas in float64 (maxabs for the worst pixel, mean for the mass).
Zero-op control (empty overrides) must read all-zero deltas —
the tripwire that the rig measures the lesion, not itself.

## Example table (flagship 24x32 synthetic rgb — `lesion_table()`)

```text
op_idx op           outs            worst_maxabs sink:maxabs,mean...
     0 SRGB_DECODE  LIN             255          OUT:255,128.9
     1 LUMA         Y               83           OUT:83,36.36
     2 SQRT         A               83           OUT:83,36.36
     3 SPLAT_BLUR   AS,COH          47           OUT:47,23.05
     4 SUB          D               26           OUT:26,11.06
     5 BETA_V5      BEFF            26           OUT:26,11.06
     6 MUL          BD              26           OUT:26,11.06
     7 ADD          AE              83           OUT:83,36.36
     8 SQUARE       YENH            83           OUT:83,36.36
     9 GAIN         G               255          OUT:255,128.9
    10 SRGB_ENCODE  OUT             255          OUT:255,128.9
control: OUT maxabs 0.0 mean 0.0 (zero-op tripwire)
self-diff: identical (first None)
```

Read: the output owes 255 to the format edges (decode/encode,
gain), 83 to the luma trunk, 47 to the splat field, 26 to the
detail chain — every op's causal share, measured not asserted.
Gate `tests/test_debugger.py` (CPU, ~1s): step-values byte-equal
to direct `run()` at every stop, 11/11 lesionable with finite
positive deltas, control exactly zero, self-diff identical,
perturbed-IN first-divergence at op 0, session-twice identical
tables, 7 fail-loud legs. Neighbors green: `test_asm.py`,
`test_census.py` (shared read paths, untouched).
Bugs caught while building (mechanisms): (1) hook-env snapshots
miss IN streams on paused runs (inputs live in run()'s internal
feeds, stranded by the pause exception) — pre-seed env from the
payload rule, shadowed streams still read latest via write
order. (2) Test compared intermediate-stop watches against
FINAL ref values — watches must expect MISSING for unproduced
streams (prefix-correct expectation, sources from census).
Gap named: O(n²) stepping (fresh bounded run per step) + no
served/CUDA stepping — fine for small lattice listings, stated.

## Teacher-side hooks (named, not built)

What teacher debugging would need: per-layer activation capture
(twin forward hooks, no_grad), prompt-override interventions
(the overrides analogue for teacher inputs), align-and-compare
vs banked teacher outputs with first-divergence reporting.
Named here so the shape is on record; none of it is built.

## Torch lesions (constitution, FROZEN pre-build)

Surface mirrors the listing semantics exactly (`chain/debugger.py`,
torch section — same module, not a second tool): `TorchSession`
over a fixed net + a reference `run()` returning named output
tensors. `lesion(module_path)` zeroes that LEAF module's outputs
via a forward hook (zeros_like — shapes/dtypes/kinds preserved,
only values change) and reruns; tabled as per-output
(maxabs, mean) deltas vs the reference run. Zero-module control
(empty hook set) must read exactly zero. Same session twice →
identical tables. Bad module paths refuse loud (no silent
skips — every leaf lesionable is gated, not claimed).
Validation targets, frozen: mscale64 + jattn64b2 BANKED weights
(read-only out-of-tree `~/student_runs/<tag>/weights.pt`,
`weights_only` load, eval + no_grad throughout — no training,
no weight writes, ever). Lesion tables banked per model
(`samples/marigold/lesion_<tag>.json`); shapes compared across
families with the open question stated, never assumed: do
lesion profiles differ the way failures do (MLP mixer vs joint
attention — uniform spread vs load-bearing blocks)?
Reference runs (frozen): mscale64 = full-frame MLP forward on
banked church fixture (stride math untouched — full frame, not
the stride sample); jattn64b2 = rgb-path encode+head on church
(Phase-8 shape: W64/B2/H1/s32/C4). CPU suffices (MLP/head
forwards are seconds on CPU — no GPU need established, none
used). Qwen and teacher weights never loaded (students only —
teacher lesion is the next round, explicitly not this one).

## Torch results (students, CPU seconds — `scripts/bank_student_lesions.py`)

mscale64 (church full-frame, 5 leaves, all fired; outputs ~3.6e5 —
blown-up like the arc's constant-attractor findings, so every
lesion ≈ total-kill and biases vanish relatively):

```text
module          fires worst_maxabs
0                   1 3.652e+05
1                   1 3.652e+05
2                   1 3.656e+05
3                   1 3.656e+05
4                   1 3.652e+05
control: exactly zero
```

jattn64b2 (rgb path, 22 leaves; unfired exactly {row, col,
dino_proj} — row/col BYPASSED (encode reads `.weight`
directly, never calls the modules), dino_proj genuinely unused
in rgb path; the fire-count tripwire tells them apart from
measured zeros):

```text
module            fires worst_maxabs
blocks.0.q/k/v/o    1   4.3-4.4e6
blocks.0.n1         1   3.3e6
blocks.0.m1/m2/n2   1   2.2-2.3e6
blocks.1.m1/m2/n2   1   2.6e6
blocks.1.q/k/v/o/n1 1   6.2-7.4e4
h1/h2/tok           1   4.4e6
row/col/dino_proj   0   0
control: exactly zero
```

Profiles compared (the open question, answered): YES, they
differ the way failures do. mscale = uniform total-kill (chain
MLP, no residuals — every layer load-bearing, outputs
astronomical). jattn = differentiated (block-0 attention path
~1e6, head h1/h2 + tok ~4.4e6 load-bearing, block-1 qkv/o/n1
~70× smaller than block-1 MLP — the second attention block is
near-dead weight; only h1/h2 carry the head). The mixer fails
as one fused rope; the attention stack fails in named places.
Gate `tests/test_debugger.py` extended in place (CPU ~9s):
lesion-every-module, control-zero ×2, live==banked both models
(cross-process determinism), session-twice identical,
container/root/bad-path fail-loud, dino_proj exact-zero,
unfired-set exact. Listing legs green unmodified (shared code
untouched behaviorally — proven by the unchanged legs, not
claimed). Neighbors green: `test_asm.py`, `test_census.py`.
Bugs caught while building (mechanisms): (1) mscale rows
printed identically at 4 sig figs — full precision shows
bias-scale differences (3.652e5 vs 3.656e5); the model IS blown
up, the printer hid it. (2) row/col zeros are hook-BYPASS
artifacts (encode reads `.weight`, never calls forward), NOT
causal findings — fire counts added to every row; the gate pins
the unfired set exactly. (3) Test helper `sess()` shadowed by a
loop variable of the same name (UnboundLocalError) — renamed.

## What teacher lesion will need

Per-layer hooks on the teacher twin (no_grad, eval), a
reference run returning named decisions (argmax-rank/fluency —
serving grades decisions, never exact strings, per house
contract), banked teacher outputs to diff against, and the same
fire-count tripwire (dead vs bypassed). Qwen-7B stays unloaded
until that round's constitution says otherwise.

## Teacher lesion Phase A — block sweep, causal table (constitution, FROZEN pre-run)

Target list frozen (exact paths, no "and neighbors" mid-round):
- 60 × `transformer_blocks.{0..59}` whole-block zero (diffusers
  QwenImageTransformer2DModel, 60×3072 — banked blueprint).
- VAE-decoder stages (9): `decoder.{conv_in, up_blocks.0..3,
  mid_block, norm_out, conv_out}` + `post_quant_conv` (grouped
  from the 108 banked vae_pairs — full-weight tuned, not LoRA).
- Top-5 tuned families by banked rel LoRA-delta, GROUP lesions
  (zero across all 60 blocks at once — instance-level would be
  300 runs, out of budget, stated): `attn.add_q_proj` (0.0558),
  `attn.to_add_out` (0.0492), `attn.add_v_proj` (0.0455),
  `attn.add_k_proj` (0.0452), `txt_mlp.net.0.proj` (0.0436).
Baseline first: unlesioned teacher AbsRel on the eval set,
banked before any lesion runs. Eval set: 8 Hypersim frames
(first 2 available of ai_018_001, ai_021_001, ai_022_001,
ai_023_005 — deterministic, no cherry-picking), GT-anchored
(masks & valid intersect — no teacher-mimicry circularity;
webcam fixtures excluded: no GT there, stated reason).
Metric: AbsRel-delta per lesion vs baseline, relative %
((les−base)/base); alignment lstsq (same doctrine as every
student round + Marigold's own lstsq-scale recipe — RANSAC
unjustified here, not run). Removal criterion frozen: delta <
5% nominates removable (below single-seed noise family; the
criterion judges, Phase C validates).
Harness (read-only sibling use): custom loader mirroring their
component path (4-bit NF4 transformer + LoRA trainables +
bf16 VAE + cached prompt embeds — never the text encoder) +
encode → ONE flow step (t=0.499, velocity) → decode →
channel-mean (mirroring QwenImageEncode/2509Step/Decode/
SelectChannel node-for-node) + seeded VAE sampling. PARITY GATE
first: custom-unlesioned == sibling-CLI output on 2 frames
(else stop — cross-harness deltas are incomparable, no rescue).
Lesions ride forward hooks (zeros_like block outputs);
peft/quant wrappers respected (fire-count tripwire carried
from student lesions — bypassed vs dead distinguished).
Budget: per-eval wall measured on 2 pilot blocks FIRST (early
idx 3 + late idx 56 — not 0/59: edges are atypical, stated),
then total = per-eval × 75 configs (1 baseline + 60 + 9 + 5);
proceed iff projected ≤ 12h else shrink the eval subset first
(documented fallback) or stop with pilot evidence (timeout
precedent). GPU solo (idle + VRAM stated at run); Qwen-7B never
co-loaded (VRAM math: ~10GB 4-bit DiT + ~2GB VAE + embeds +
activations « 24GB; text encoder and Qwen-7B stay on disk).
Weights read-only out-of-tree (lesion at runtime — a single
modified weight file voids the round; sha256 of the 7 files —
5 transformer shards + trainables + VAE safetensors — banked
pre AND post, gate compares).
Ruling: hooks only; no emitter/opcode/assembly changes (gap
hit = stop-finding); no new deps (torch/diffusers/peft live in
marigold-env); production weights untouched (teacher AND
Qwen-7B); no downloads (shard suffices).
Gate: new tests/test_teacher_lesion.py (CPU-fast legs +
banked-table integrity; heavy reruns NOT routine — checksums,
schema, baseline-recompute-from-banked, delta arithmetic
re-derived in-gate). Neighbors: anatomy + teacher gates green
(read-only reuse).
Spec: this section + baseline + per-block delta table +
removable nominations per the frozen criterion — nothing more.

### Phase-A results (TBD — pilots first, then the run)
