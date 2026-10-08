# YarnBall: the generic weight object (v1)

A YarnBall is a factored weight object plus its access discipline,
runnable as assembly data. Math and listing agree by construction;
gates pin both.

## Definition

```
YarnBall = (stores, tiers, ledger)          # data (assembler-side)
W = U·diag(s)·Vt                            # strands: Wx = Σ sᵢ(uᵢ·x)vᵢ
store = (key, value, gain)                  # WHERE x WHAT x HOW MUCH
```

- **address**: key match in receiver geometry (lexical-early for
  teachers, HN/HNB/E-space per path natively). Matched by projection;
  sharpened by key scale; conjunctions fuse pre-softmax (dualbank).
- **content**: value directions. Coarse content travels by contrast;
  precise content must be native/exact (readout rows). Provenance is
  part of the type, not metadata.
- **dose**: gain folded into value rows host-side (dose discipline:
  operating windows 1-2x everywhere measured; leakage is always
  partial-weight x large-dose).
- **tiers**: `exact` (bit-close ideas) | `assoc` (associative stores)
  | `opt` (operating-point refits) | `null` (zero value: background
  so softmax has somewhere to route non-targets = hold by
  construction) | `base` (native ball).
- **ledger**: one row per store (tier, support, key/value norms, dose,
  sha). A bank without ledger rows is anonymous yarn.

## Structures (stdlib, zero new mnemonics)

| DEF | file | form |
|---|---|---|
| `yarnball_apply(xa, Ua, Vc)` | `stdlib/yarnball.asm` | soft-gated bank (MATMUL+TSHIFT+SOFTMAX_WIDE+MATMUL) |
| `dualbank_apply(xh, xe, Uh, Ue, V)` | `stdlib/dualbank.asm` | two-channel log-space AND + apply, returns (y, P) |
| `dynaddr_match(x, key)` | `stdlib/dynaddr.asm` | address by content (MATMUL+ARGMAX+GATHER) |
| `implant_apply(x, u, v)` | `stdlib/dirstore.asm` | K=1 special case (rank-1 write) |

Data builders: `chain/engram.py` (`bank()` gains-folded pairs,
`yarnball_bank()` full ledger incl. nulls). Cost model:
`chain/read.py` (predict_db ±0.3dB; preview before emitting).

## Readout tiers (native LM)

| head | columns | carries | installs |
|---|---|---|---|
| skewed `wlog` | norms = prior (unk 6.98) | glue/UNK (0.337) | blocked (floor 8, any dose) |
| flat `wlogU` | unit (frozen `data/wlogU.npz`) | nothing alone (0.146) | flips at 4x, no holds |

Routed together (`programs/lm_dualhead2/3.asm`: blend by bank
weight): prior identical + top-1 installs + holds. Prior and
install are routed tiers, not competing objectives.

## Loops (products, not scripts)

- **negmine**: split -> collect flips -> mine keys -> append nulls ->
  re-gate. Addressing that learns (7->19 stores, receipt intact).
- **dose ladder**: operating window per install (1-2x typical).
- **key-scale ladder**: sharpen toward 0/1 weights (ks=32 locked
  natively; receipt is the tripwire).

## Gates

| gate | bar |
|---|---|
| `tests/test_yarnball.py` (MVYB) | parity exact; retrieve own; 299->24; stable holds |
| `tests/test_dualhead2.py` | FLIP + receipt (>0.9/<0.1) + split == skewed base + stable |
| `tests/test_scale2.py` | 2 FLIPs + identical prior + receipts + negmine loop |
| `tests/test_dualbank.py` | 8/8 invariance + exclusion, no tuning |
| siphon (mirror+listing) | target r1 + holds + own-retrieval, one ball |

## Limits (stated, not missing)

Single-vector installs through one skewed head: ~96% of targets
infeasible at any dose (full-vocab scan). Razor positions move under
any routing (blend arithmetic ~0.1-logit scale); stable never does --
grade on stable (margin>=1), track razor separately. Hard routing
needs a comparison op (language extension, filed). Training is out
of scope (no gradients anywhere); alignment growth is the known
ceiling (teacher flat ~2x + fluent; ours skewed 8.3x).
