# Authoring structures (the assembly level)

Discovered structures are parsed from models. Authored structures are
invented to fill a need -- the beta-field lineage (v1-v5), the router,
consensus, hashed priors were all authored, and they compose with
discovered ones indistinguishably BECAUSE they clear the same bar.
This file is that bar, plus the submission process. First gallery:
decide-helper (due), prior_mult (2 uses), per-tile routing (backlog).

## The bar (a structure is submittable iff it has ALL of these)

1. **Name + level** (L0 substrate / L1 compute / L2 control / L3 compose).
2. **Geometric meaning in one paragraph.** What it MEANS, not what it does
   (e.g. "caution: attenuate boost where a prior reports uncertainty").
   No meaning -> not a structure, just code.
3. **Canonical form**: exact integer semantics (ops, types, scales),
   operands in/out, edge semantics (replicate/zero/clamp), failure modes
   (what inputs break it, what it does then).
4. **Parameters**: frozen constants (stated values + reasons) and/or fitted
   ones (pairs, objective, freeze file, must-beat rule). No live fitting.
5. **Oracle mirror rule**: what the float twin computes (may be shared code
   shape, never shared values -- mirrors drift, cf #LIB-014 postscript).
6. **Required gates**: parity basis, at least one behavioral gate proving the
   MEANING (not just the math -- e.g. rescue-V proves corroboration, ratio
   0.504 proves caution), one negative or boundary gate, C/shared-vector
   coverage OR a stated backlog entry for it.
7. **Instances**: >=1 gated use. Second use promotes helpers; third use
   REQUIRES promotion (the select_mux precedent).

## Submission process

1. Propose (name, meaning, why now) -- one paragraph, get agreement.
2. Land numpy + oracle + gates in the workbench model (holo today).
3. Lower (C + file-exchange) or file the backlog entry with a priced reason.
4. Promote shared parts to phi-core with 0-diff gates.
5. Enter it in INVENTORY.md (status SINGLE until a second instance).

## Gallery (authored structures, with status)

- `decide` verdicts (chain/verdict.py): boundary float -> exact bool,
  stated inclusivity, refusal on unknown ops. 3 uses refactored 0-diff
  (motion static, depth median, router mode; suites prove it). STAGED for
  phi-core as branch `ai/verdict-helper` (module + gates, pushed, merge
  pending owner review) -- shared-repo coordination rules followed throughout
  (branch, claim check, no main push, base gates green).
- `prior` modulation (chain/prior.py): multiplicative caution from offline
  priors, two forms one family -- select(mask) for binary priors (depth),
  affine(field) with exact-ones-on-static for continuous ones (flow).
  Contract: unaffected subset takes exact identity; caution never invents.
  2 uses refactored 0-diff (suites prove it); no new C (composition of
  select_mux/tmul/binop/encode). STAGED for phi-core as branch
  `ai/prior-mult` (ported onto lattice+numpy_ops, gates green, pushed,
  merge pending alongside ai/verdict-helper).

- `stabilize` listing (programs/stabilize_mgd.asm): first GENERATIVE proof --
  designed on paper as a denoiser, falsified on flats (see #LIB-021), renamed
  to what it is (temporal stabilizer: flicker -26%, no smear, parity 64dB).
  Four new mnemonics (ISO_BLUR, WARP, STATIC, MIXDYAD), all thin wrappers.
  Test: test_stabilize.py. The paper-first loop closed in one round: predict,
  run, falsify-or-confirm, rename-or-ship.

## Assembly view (read a pipeline as structure invocations)

The holo flagship today, at assembly level (cf chain/ for the machine code):

```
Y
|> SQRT                          # S-amplitude (I=|A|^2 domain)
|> SPLAT_BANK[V,H,D1,D2,iso]     # S12 (fused or soft)
|> TENSOR_COH                     # S11 (discriminant @ m_acc)
|> CONSENSUS[flow?]               # S-motion (confirm/veto, never originate)
|> BETA_FIELD[v5]                 # S13 (fitted table x learned gate)
|> PRIOR_MULT[depth?,flow?]       # S14 (multiplicative caution)
|> TEMPORAL_IIR[state?]           # S19 (dyadic memory, first-frame identity)
|> ROUTE[still/temporal]          # S15 (meta-selection per frame)
|> SQUARE |> GAIN                 # S-amplitude back + chroma preserve
```

Every line names a structure, every structure has a spec, every spec has
gates. New pipelines are new listings. THAT is the abstraction level.
