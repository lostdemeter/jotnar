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

## Example table (TBD — flagship run lands here)

## Teacher-side hooks (named, not built)

What teacher debugging would need: per-layer activation capture
(twin forward hooks, no_grad), prompt-override interventions
(the overrides analogue for teacher inputs), align-and-compare
vs banked teacher outputs with first-divergence reporting.
Named here so the shape is on record; none of it is built.
