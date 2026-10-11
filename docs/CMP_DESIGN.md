# Comparison op (GT): hard routing design (v0.1, authored not built)

**Status:** design. Builds on v2's verdict (blend arithmetic perturbs
razor at any w>0: three ks rungs flat) and the negmine close (soft
routing suffices today). Hard routing removes the residue class
entirely; nothing currently green needs it.

## 1. Name + level

`GT` (greater-than), L1 compute (elementwise decision primitive, feeds
L2 control SELECT). Sibling: `STATIC` (exact verdict mask, precedent).

## 2. Geometric meaning

Lattice order as a verdict: for same-shape triple streams A, B, GT
answers per element "does A lie strictly above B in lattice order"
(the ARGMAX ordering: class pos>zero>neg, then exponent; ties are
NOT greater). Turns a soft weight into a hard gate: `w > thr` becomes
routable. Meaning, not mechanism: thresholded decisions.

## 3. Canonical form

- Operands: two triple streams, identical shapes (mismatch fails
  loud, SELECT-compatible: output I mask matches branch geometry).
- Output: I bool array (True where A strictly above B).
- Edge semantics: zeros equal (zero is never > zero); cross-class by
  class rank; within class by exponent; exact ties (same s,e,z) False.
- NaN: none exists (lattice has no NaN; state it).
- Failure modes: shape mismatch, float inputs (triples-only, core
  doctrine), non-triple streams -- all fail naming op+line.

## 4. Parameters

None (pure function of inputs). No frozen values, no fitting, no
CONFIG keys. Thresholds arrive as data (BETA-filled reference
streams, existing pattern).

## 5. Oracle mirror rule

Float twin: elementwise `>` on decoded float64 values. Agreement bar:
bit-exact vs twin on randomized shapes incl. ties/zeros/cross-class
(the ordering is total and deterministic; mirror rule is definitional
here, shared values never cross).

## 6. Required gates (before merge)

- Parity: bit-exact vs float twin (1000 randomized triples + edges).
- Behavioral: dualhead5 (GT-routed v2 bank) DOMINATES soft v2 on
  every metric (/tmp/dualhead5.log: 0.305 vs 0.293, glue 9 vs 6,
  holds 16/16 vs 11/16, unk tied 66). NOT v1-bit-identical: v1's
  purity came from full-drug H5, impossible under any routing with
  a null-mix bank (different H5, different LOGSKEW -- the original
  prediction here was wrong, corrected with mechanism).
- Negative/boundary: shape mismatch, float inputs, ties-False,
  zero==zero, all-names-op+line.
- C/shared-vector coverage: CPU ASM path + nonfpu backend
  (bit-exact, `test_emit_nonfpu.py`: mask bits + SELECT-over-mask);
  C/CUDA patterns filed as follow-up (NoPattern, loud).

## 7. Instances (first use: built, gated)

`lm_dualhead5.asm`: v2 blend with `M = GT(W, THR)` mask +
`SELECT(M, FLAT, SKEW)` replacing MUL/SUB/ADD blend. THR as IN data
(frozen per prompt, cmask class -- simpler than the BETA-fill first
proposed; no new CONFIG readers). Measured (/tmp/dualhead5.log):
FLIP + receipt clean + dominates soft on 3/4 metrics, ties UNK.
Razor-zero prediction falsified with mechanism (null-bank H5, not
blend: GT removes blend crumbs exactly as designed, residue was
never blend's). research/negmine5.py: hard routing + learned
background = strictly best native config (0.350/9/77 + FLIP).

## Non-goals

Full comparison family (LT/EQ/NE): GT + SELECT-negation covers
(`SELECT(NOT...)` needs boolean invert -- also filed, smaller);
float masks (refused: SELECT doctrine); in-place forms.
