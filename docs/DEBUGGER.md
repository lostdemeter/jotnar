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
