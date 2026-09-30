# LANGUAGE.md — the assembly language, single reference (v1.0 Gate 2)

42 mnemonics. Every line of every program is `OUTS = MNEMONIC(args)` plus
four declarations (`CONFIG IN STATE RANGE`) and three composition forms
(`DEF CALL IMPORT`). This file is the whole language: the tutorial (write
your first listing in 15 minutes), the per-mnemonic contracts with examples,
and the contribution process (how a new mnemonic earns its place).

Run listings with `chain/asm.py:run_text(text, REGISTRY, payload, sigs=SIGS)`.
Fail-loud is the core doctrine: unknown mnemonics, arity mismatches,
use-before-def, layout conflicts, and shape mismatches all raise `AsmError`
naming the operation and line — never a silent wrong number.

## 1. Orientation (5 minutes)

```python
import sys; sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, ".")
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
text = open("programs/holo_flagship.asm").read()
feeds = ASM.run_text(text, REGISTRY, rgb_array, sigs=SIGS)  # U8 HWC array in
out = feeds["OUT"]  # U8 HWC array out
```

The flagship (`programs/holo_flagship.asm`, 11 instructions) is the model
listing: config up top, inputs declared, one structure per line. The
transformer block (`programs/xf_block.asm`, 19 instructions) is the second
model listing. Read both before writing your own.

## 2. Concepts

**Streams.** Every name is a stream: triples `(s,e,z)` (kind `T`), float
arrays (`F`), integer/bool arrays (`I`), or bytes (`U8`). An op with one
declared output produces ONE stream (even when the value is itself a
triples-tuple); N outputs take an N-tuple. Payloads: single-`IN` listings
take a bare value; multi-`IN` listings take `{name: value}` (unknown keys
and missing streams fail loud).

**Layouts `KIND:GEOM`.** Kinds: `F T I U8`. Geometry vocabulary is open:
`HW HWC SEQ HEADS HW2 SCALAR ...`. Declared on `IN` (`IN x AS T:SEQ`) and on
`DEF` formals; literals have layout `F:SCALAR`. Signatures unify `$VAR`
(whole layout), `$VAR^T` (transposed), `*` (anything, binds nothing).
Gradual typing: unannotated streams are `UNKNOWN` — they unify silently and
never fail (every suite passes with partial annotations = backward compat
proven). Concrete-vs-concrete mismatches fail naming op+line+stream.
A declared kind must also match the VALUE (`T`-declared holding floats
fails — triple-ops on float arrays used to compute garbage silently).

**Config.** `CONFIG key value` — frozen choices (e.g. `beta 0.5`,
`rope_base 10000.0`, `eps_rms_c 4514`). Scale overrides for multi-regime
listings (v1.1): MATMUL + BATCH_MATMUL honor `m_acc`, ADD + SUB honor
`m_cov` — integral 0..65535, same rule as RESCALE; absent keys take the
frozen values. Unknown keys don't fail; ops read what they need with
documented defaults.

**Literals.** Bare numbers in arg position are `F:SCALAR` floats. Shape
literals (axes, radii, counts) must be INTEGRAL — `1.5` fails loud, never
truncates.

## 3. Tutorial: your first listing in 15 minutes

Step 1 — arithmetic (minute 0-5). Save as `first.asm`:

```
IN a
IN b
OUT = ADD(a, b)
```

Run it:

```python
import numpy as np
import phi_core.lattice as S
a = S.encode(np.full(3, 0.25))
feeds = ASM.run_text(open("first.asm").read(), REGISTRY, {"a": a, "b": a}, sigs=SIGS)
v = S.decode(feeds["OUT"][0], feeds["OUT"][1])  # ~0.50 (lattice quantum ~3e-3)
```

Step 2 — reuse (minute 5-10). Add a procedure:

```
IN a
IN b
DEF twice(x) -> (y)
  T = ADD(x, x)
  y = ADD(T, x)
END
OUT = CALL twice(a)
```

`CALL` is a textual macro with per-call-site namespacing (the expanded
listing IS the program — inspectable, no call overhead). Recursion, nested
`DEF`, and `CONFIG/STATE/IN` inside `DEF` all fail loud.

Step 3 — activate (minute 10-15). Swap the last line for `OUT = SILU(a)`
or `OUT = GELU(a)` and compare against `torch.sigmoid` / `torch.gelu`.
You have now used the generate-your-own-gate loop: propose, run, compare.
Next: read `programs/xf_block.asm` with §4 open beside it.

## 4. Mnemonic reference (42)

Notation: `NAME(in -> out)` + layout sig + contract one-liner + example +
loud-failure. Arithmetic core (`ADD SUB MUL DIV`) refuses float inputs
(triples-only — silently computing garbage was a real bug class).

**Holo image path (flagship):**
- `SRGB_DECODE(U8:HWC -> F:HWC)` — gamma-2.2 decode. Ex: `LIN = SRGB_DECODE(rgb)`.
- `LUMA(F:HWC -> F:HW)` — Rec.709 luma (sum fingerprinted 1.0). Ex: `Y = LUMA(LIN)`.
- `SQRT(F:HW -> T:HW)` — `A=sqrt(Y)` via exponent halve (sign forced +1). Ex: `A = SQRT(Y)`.
- `SPLAT_BLUR(T:HW -> T:HW, T:HW)` — soft oriented bank + coherence. Returns `(AS, COH)`. Ex: `AS, COH = SPLAT_BLUR(A)`.
- `SUB/MUL/ADD/DIV($A,$A -> $A)` — triples arithmetic @ m_cov (DIV: zero-or, no guards). ADD/SUB honor CONFIG `m_cov`. Ex: `D = SUB(A, AS)`.
- `BETA_V5(T:HW,T:HW -> T:HW)` — learned beta gate `(D, COH)`. Ex: `BEFF = BETA_V5(D, COH)`.
- `BETA($A -> $A)` — CONFIG beta as triples at the reference shape. Ex: `B = BETA(A)`.
- `SQUARE($A -> $A)` — `tmul` + clip to [0,1]. Ex: `YENH = SQUARE(AE)`.
- `GAIN(F:HWC,F:HW,T:HW -> F:HWC)` — chroma-preserving gain. Ex: `G = GAIN(LIN, Y, YENH)`.
- `SRGB_ENCODE(F:HWC -> U8:HWC)` — gamma-encode + clip. Ex: `OUT = SRGB_ENCODE(G)`.
- `ISO_BLUR(T:HW -> T:HW)` — wide gaussian structure extraction. Ex: `S = ISO_BLUR(A)`.
- `GAUSS($A,F:SCALAR,F:SCALAR -> $A)` — gaussian `(a, radius, sigma)`, normalized. Ex: `S = GAUSS(A, 1, 0.8)`.
- `WARP(T:HW,F:HW2 -> T:HW)` — detail warp by float flow; `None` history passes through. Ex: `W = WARP(dprev, flow)`.
- `STATIC(F:HW2 -> I:HW)` — exact static mask (`|flow| == 0`). Ex: `M = STATIC(flow)`.
- `MIXDYAD(T:HW,T:HW,I:HW -> T:HW)` — static ? `(D+3W)/4` : `D`; `None` history -> `D`. Ex: `DM = MIXDYAD(D, W, M)`.

**Transformer (block):**
- `MATMUL/BATCH_MATMUL(*,* -> *)` — triples matmul @ m_acc (batch dims + B-broadcast). Inner dims must agree or fail WITH the transpose hint. Honor CONFIG `m_acc` (multi-regime listings). Ex: `Q = MATMUL(XN, wq)`.
- `SOFTMAX($A -> $A)` — row softmax to probability triples. CONTRACT: inputs ≤1.0 abs (`to_fixed` saturates above it at BIAS — the T-transformation doctrine; out-of-contract saturates to softmax-of-clipped, pinned by gate). Ex: `P = SOFTMAX(SCORES)`.
- `RMSNORM($X,$W -> $X)` — per-row RMSNorm+weight; `eps_rms_c` from CONFIG (default 4514 = `eps_c` in 2^-36 ambient counts; regime-dependent, per-model calibration is backlog). Ex: `XN = RMSNORM(x, rms_w1)`.
- `SILU($A -> $A)` — `x*sigmoid(x)`, 0-diff vs phi-core. Ex: `GS = SILU(GATE)`.
- `GELU($A -> $A)` — exact-form `x*Phi(x)` (EXPACT + PHI LUT, any range; exact asymptotes beyond ±16). Ex: `G = GELU(X)`.
- `ROTARY($X,* -> $X)` — RoPE pairs rotation; even last dim required; positions are int metadata; `rope_base` from CONFIG (default 10000.0). Ex: `QR = ROTARY(Q, pos)`.
- `TRANSPOSE($A -> $A^T)` — last-two-axes swap, exact. Ex: `KT = TRANSPOSE(KR)`.

**Exact moves (reshape family):**
- `RESHAPE2(*,s,s -> *)` / `RESHAPE3(*,s,s,s -> *)` — reshape to 2D/3D; element count must match. Ex: `H3 = RESHAPE3(X, 2, 3, 4)`.
- `PERMUTE3(*,s,s,s -> *)` — axis reorder, must be a permutation of (0,1,2) on 3D. Ex: `P = PERMUTE3(H3, 1, 0, 2)`.
- `CONCAT($A,$A,F:SCALAR -> $A)` — concatenate along an axis literal; non-axis dims must match. Ex: `C = CONCAT(cache, row, 0)`.
- `SLICE($A,s,s,s -> $A)` — `SLICE(X, AXIS, START, END)` half-open, bounds-checked (no silent clamp). Ex: `W = SLICE(X, 1, 1, 4)`.
- `GATHER(*,I:* -> *)` — row gather by int ids (bounds-checked). Ex: `R = GATHER(table, ids)`.

**Reductions / pooling / conv:**
- `ARGMAX($A,F:SCALAR -> I:*)` — exact lattice ordering WITHOUT decoding (class pos>zero>neg, then exponent); ties -> first; integral axis. Ex: `I = ARGMAX(X, 1)`.
- `CLIP($A,s,s -> $A)` — clip to `[LO,HI]` literals. Ex: `C = CLIP(X, -1, 1)`.
- `SIGMOID($A -> $A)` — EXPACT+LUT, any range, [0,1]. Ex: `S = SIGMOID(X)`.
- `RESCALE($A,F:SCALAR -> $A)` — THE only scale changer (fixed@m_cov -> `rescale_` to M2 int literal 0..65535). Needed where listings cross m_acc/m_cov. Ex: `Y = RESCALE(X, 33280)`.
- `PRELU($A,$A -> $A)` — exact PReLU (sign-bit select). Ex: `P = PRELU(X, slope)`.
- `POOLAVG(* -> *)` — global average to `(C,)`; HWC triples ONLY (SEQ fails loud). Mean truncates (1-LSB class). Ex: `V = POOLAVG(F)`.
- `DECONV(*,*,s,s -> *)` — transpose-conv `(x, W, stride, pad)`; no bias v1 (stated). Ex: `Y = DECONV(X, W, 2, 0)`.
- `INTERP(*,s,s -> *)` — bilinear `(x, sy, sx)`; NON-DYADIC scales are a LOWERING ERROR (fail loud, never approximate). Ex: `Y = INTERP(X, 1, 1)`.
- `CONV(*,* -> *)` — general kernel `(x, K float array)`, per-tap tmul + order-free accumulate. Ex: `Y = CONV(X, K)`.

**Control:**
- `SELECT(I:*,$A,$A -> $A)` — verdict-gated branch: per-element pick of A/B by bool mask; branches must match, mask must match. Ex: `Y = SELECT(mask, A, B)`.

## 5. Procedures and imports

```
DEF name(a AS T:HW, b) -> (y AS T:HW)
  ... (no CONFIG/STATE/IN/IMPORT/RANGE inside; no nested DEF)
END
OUT = CALL name(x, z)
IMPORT "lib.asm"        # basedir-relative splice; cycles fail loud
```

Formals may carry `AS` contracts: at each `CALL` site the caller's DECLARED
layout is checked (both known + concrete + unequal fails naming the CALL
site AND the DEF origin). `UNKNOWN` either side defers to post-expansion
`verify()` + runtime. `$VAR` formals are refused (no bindings at contract
time). `verify()` also reports the DEF interface table (all composition
contracts in one place).

Shared listings: put `DEF`s in a file, `IMPORT` it (see the IMPORT gate in
`test_asm.py`: spliced-file execution + cycle refusal). The `stdlib/` dir
ships two seed blocks (`attention.asm: attn_core`, `mlp.asm: swiglu_block`),
shared by `programs/xf_block.asm` — no copy-pasted prologues in shipped
programs (v1.0 Gate 3, dogfood-proven bit-exact in `asm-dogfood-xf`).
Search order (v1.0 Gate 4): bare names (`IMPORT "mlp.asm"`) resolve
STDLIB FIRST, then basedir-relative, so shared listings are addressable by
name from any directory; explicit paths (`./x.asm`, `sub/x.asm`) stay
relative-only and are never shadowed by stdlib. A basedir file with a
stdlib name loses to stdlib (gated) — name local files distinctly.

## 6. Loops and state

```
IN rgb
IN dprev
STATE dprev             # must be IN-declared AND assigned as OUT every iteration
dprev = ...             # carried; shapes fixed across iterations (dynamic refused)
```

`ASM.repeat(text, REGISTRY, payload, n, sigs=SIGS)` threads STATE: non-STATE
streams take LISTS of length n (strict, no broadcast); STATE takes one seed.
`grow=[...]` allows APPEND-ONLY growth along axis 0 (the KV pattern:
`cache = CONCAT(cache, row, 0)`); off-axis growth and shrinking fail loud.
Growth is logged per iteration. Data-dependent termination (`WHILE`) is out
of scope — stated (all current needs are bounded).

## 7. Static checks

- `ASM.verify(text, REGISTRY, SIGS)` — replays layout unification over
  DECLARED layouts with zero execution (flagship 13/13); returns
  `(errors, report)` with resolved/total coverage + DEF table.
- `chain/ranges.py:estimate(text, REGISTRY, SIGS)` — hull intervals vs
  `RANGE name lo hi` declarations + `M.json` coverage. ASYMMETRIC doctrine:
  saturation flags on ANY exceedance; underflow only whole-range-below
  (tiny values are often legitimate ~0). Stated limit: hulls cannot see
  precision loss INSIDE spanning ranges (pinned by `programs/tensor_disc.asm`
  + unit gates). Coverage binds LATTICE values only (`T:*` layouts).
- Runtime shape rules (no annotations needed): elementwise identical
  shapes; matmul inner dims (+ transpose hint); warp spatial match; argmax
  axis range; gather id bounds; SELECT branch+mask agreement. Strictness was
  EARNED (every suite passes with checks active — no legitimate broadcast
  broke). Per-op scale declarations (`@scale(m_acc, m_cov, eps_c)`) are the
  stated next step (phi-core ASM_HANDOFF.md §4 supplies the semantics).

## 8. Debugging

`run_text(..., trace=log)` records per-op `{line, op, in/out summaries,
sec}` WITHOUT perturbing values (bit-identical, gated).
`ASM.format_trace(log)` renders the panel (line, shapes, layouts, means,
per-op seconds + total). Summaries show SYMPTOMS (shapes/layouts/means
catch the broadcast/transpose classes in minutes); gates prove CAUSES.
The profiler falls out free (per-op wall time).

## 9. Contribution process

New structures (mnemonics AND authored patterns) clear ONE bar
(`docs/AUTHORING.md` — the submittable structure needs ALL of these):

1. Name + level (L0 substrate / L1 compute / L2 control / L3 compose).
2. Geometric meaning in one paragraph (what it MEANS, not what it does).
3. Canonical form: exact integer semantics, operands, edge semantics,
   failure modes.
4. Parameters: frozen (values + reasons) and/or fitted (pairs, objective,
   freeze file, must-beat rule). No live fitting.
5. Oracle mirror rule (float twin computes WHAT; never shared values).
6. Required gates: parity basis + ≥1 behavioral gate proving the MEANING
   + ≥1 negative/boundary gate + C/shared-vector coverage OR a stated
   backlog entry.
7. Instances: ≥1 gated use. Second use promotes helpers; third use
   REQUIRES promotion (the `select_mux` precedent).

**Promotion rule:** 2 uses = helper candidate, 3 uses = MUST promote to
phi-core (branch, claim check, no main push — shared-repo rules). The
extension drill (v1.0 Gate 5) is the timed form of this process: op + sig
+ gate + docs + listing use, wall time in `docs/VELOCITY.md` (GELU: ~3 min).

**Gate patterns to copy** (from `test_asm.py` / `test_xf_block.py`):
- 0-diff vs source fn with identical args (wrappers add NOTHING).
- Parity with declared basis (`peak=1.0`, bar 40dB) + fixture reported
  WITH the number + in-contract tripwire (prove the barred row is
  non-vacuous) + measured (not barred) out-of-contract row with mechanism.
- Negative gates: every failure mode fails naming op+line (`use-before-def`,
  bad axis, OOB ids, non-dyadic scale, import cycles).
- Tripwires: assert the seam was actually exercised (substitute invoked 2x;
  out-of-contract row truly out-of-contract) — a gate that passes without
  exercising the seam is green wallpaper.
- Mirrored-bug awareness: every blur/mux variant gets its OWN rotation
  gate; cross-check against the rule, never against the mirror.

## 10. Limits (stated, not missing)

No rank arithmetic (MATMUL is wildcard; phi-core asserts fail loud inside).
No dynamic shapes (fixed geometry refused upfront). No `WHILE`
(data-dependent termination needs a semantics + verdict-integration design
conversation first). No per-op `m` override in the estimator. No stdlib
search paths yet (Gates 3+4). No per-tile routing (backlog). C speed path
for new ops is deferred (correctness first, priced). `POS`/metadata rides
as `*`. `IMPORT`ed `IN`s stay `UNKNOWN` until post-expansion machinery.
