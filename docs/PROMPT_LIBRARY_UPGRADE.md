# PROMPT: phi-core library upgrade round (run in a fresh instance)

Copy everything below the line into a new chat. It is self-contained.

---

You are working on the phi-core geometric AI program. Two repos matter:

- LIBRARY (where your changes go): `/home/thorin/Documents/OpenCode/phi-core`
  The shared integer substrate: phi-lattice codec (`phi_core/lattice.py`),
  opcode spec (`IR.md`), calibration, C/SIMD/CUDA lowerings (`c_core/`),
  repo stamper (`phi_core/stamp.py`). Has its own `FRAMEWORK_NOTES.md`,
  `ROADMAP.md`, `tests/`. Check `git log` first; it moves.
- PROVING GROUND (test consumer, must stay green):
  `/home/thorin/Documents/OpenCode/holographic_enhancement`
  A true-amplitude holographic image enhancer built on phi-core (sqrt-amplitude
  datapath, 5-way splat bank, fitted beta-field controller v5, DAV2 depth
  composition). It exercises every library pattern and holds the evidence for
  everything below. Do not break it; use it to prove your changes.

## Coordination (shared repo: other AIs work on phi-core — read this first)

Multiple AI workers land on phi-core. Your job includes not overwriting
their hard-earned work. Rules, no exceptions:

1. NEVER work on `main`. Session start is:
   `git fetch origin && git checkout -b ai/<short-topic> origin/main`
   (e.g. `ai/trap-v2`). All work happens on that branch.
2. Before touching anything: `git log --oneline -10 origin/main` plus
   ROADMAP.md and FRAMEWORK_NOTES.md. If another recent commit claims or
   touches your area, STOP and ask the user which takes precedence. Do not
   silently re-do, revert, or "improve" another worker's landed commit.
3. Verify the base is green before starting (library gates:
   `tests/test_lattice.py`, `make -C c_core check trap`). A red base is
   reported, not built on — say so and wait for direction.
4. Stage explicitly (`git add <paths>`), never blind `git add -A` here;
   before each commit inspect `git status`, `git diff --stat`, `git diff`.
   Commit per item with evidence-stating messages. Never commit secrets.
5. NEVER push to `main`, NEVER force-push anything. Push only your
   workstream branch (`git push origin ai/<topic>`). Merging to main is the
   user's call.
6. Session end: `git fetch origin && git rebase origin/main`, re-run ALL
   gates (library + proving ground). If the rebase conflicts: resolve
   carefully, re-gate, and REPORT every conflicted hunk in your final
   summary. Never leave the tree dirty: commit or stash with a clear message.
7. The proving ground repo (holographic_enhancement) is single-worker; its
   normal workflow applies there. This section is about phi-core.

## Mandatory reading, in order (do not skip)

1. `/home/thorin/Documents/OpenCode/phi-core/IR.md` — the opcode spec, type
   system, doctrine. This is law.
2. `/home/thorin/Documents/OpenCode/phi-core/FRAMEWORK_NOTES.md` — existing
   improvement notes. Extend, don't duplicate.
3. `/home/thorin/Documents/OpenCode/holographic_enhancement/docs/PROGRESS.md`
   — what was built and measured.
4. `/home/thorin/Documents/OpenCode/holographic_enhancement/docs/LIBRARY_NOTES.md`
   — #LIB-001..015: every item below has its evidence here. Read all of it.

## Philosophy in six lines (violating these fails review)

- Values are `sign x PHI^(exp/512)` triples or `fix @ m` int64; float exists
  ONLY at encode/display boundaries and offline (calibration, fitting, LUTs).
- Two tensor types, closed op set; `rescale` is the only scale changer.
- Integer sums are order-free, so lowerings are bit-exact BY PROOF, gated by
  file-exchange parity (exact equality, never dB for lowerings).
- dB parity always declares its basis (units, peak, reference, rounding).
- Fit offline on deterministic pairs, freeze to git, must-beat-to-rewrite;
  runtime never fits. Loaders return the WHOLE frozen dict (#LIB-012).
- Reuse count, not taste, promotes helpers (3 uses). Bank evidence with numbers.

## The work (priority order; each ships with its gate or it doesn't ship)

### 1. Fit/freeze harness (3 hand-rolled copies: calibrate, fit_ctrl, fit_v5)
Build the framework helper the promotion rule demands: deterministic pairs
registry, scoring fn, must-beat-to-rewrite, pairs-hash gate, schema-completion
writes. Port ONE existing fitter onto it (suggest fit_v5: smallest, best
understood) with 0-diff behavior (same frozen values re-derived).
Gate: ported fitter reproduces its CTRL.json values; old script removed;
proving-ground suites green.

### 2. Trap v2 (the current one greps prose)
`make trap` fires on English words ("float" in a block comment, "torch" inside
a docstring) because it greps text. Twice observed. Rebuild it AST-based:
strip comments/strings, check real imports and float constructors in hot-path
modules, with an allowlist file. Gate: old trap's true positives still fire
(stash a float in a C file and a torch import in a hot module and show both
caught); the two historical false positives pass. Apply to phi-core AND the
proving ground.

### 3. Mirror discipline (parity is blind to mirrored bugs)
`_soft_blend` mapped diagonal weights to swapped kernels in BOTH numpy and
oracle: parity green, rotation 5x broken (#LIB-014 postscript). A stale oracle
line did the same earlier (agreement 0.44). Rule: every blur/mux variant
declares a symmetry manifest (orientation mapping + rotation gate); orientation
mappings cross-check against the bucket rule, never against the mirror.
Deliverable: manifest format documented in IR.md (short section) + manifests
for the proving ground's three blur paths; rotation gates already exist as
reference (test_ctrl/test_v4/test_v5).

### 4. Frozen-file schema validation (the #LIB-012 pattern, free with #1)
`load_ctrl` once dropped new keys while both sides ran different defaults and
parity stayed green. Make it a helper: validate frozen file against an
expected-key schema at load, fail loud on gap. Gate: a test that drops a key
from a fixture file and shows the loud failure; proving-ground cache-parity
gates keep passing.

### 5. Static range estimator (catch underflow at wiring time)
A discriminant sum (measured 9e-6) underflowed m_cov's fixed floor (~1.6e-5)
and was found by a failing gate. All inputs for a static check exist (kernel
norms + calibration maxima). Build the estimator: given op list + scales +
maxima, flag values that underflow floors or overflow caps before anything
runs. Gate: it flags the historical discriminant case from its frozen
inputs; zero false positives on the proving ground's current chain.

### 6. Exchange-driver codegen (hand-packed structs isn't a method)
Every C file-exchange driver hand-packs planes (padding/layout bug class).
Generate drivers from op signatures. Start with ONE: regenerate an existing
driver (suggest conv_exchange) and prove byte-identical outputs on its
existing vectors. Gate: generated driver passes test_c_conv.py unmodified.

### 7. Stamper artifact classes (we nearly git-ignored our calibration)
`.gitignore` once excluded `M.json`; frozen evidence (M.json, CTRL.json,
gate-depended priors) must be TRACKED while regenerable LUTs stay ignored.
Encode TRACKED-vs-IGNORED classes in `phi_core/stamp.py` so new repos get it
right. Gate: stamp a scratch repo, verify the classes; no change to existing
repos' behavior.

## Global constraints

- Every proving-ground suite stays ALL OK (run all):
  `test_core.py test_parity.py test_splat.py test_ctrl.py test_depth.py
  test_v4.py test_v5.py test_c.py` (needs gcc), plus `showcase.py` (all modes GO).
- Library's own gates stay green: `tests/test_lattice.py`, `make -C c_core check trap`.
- No float in hot paths (trap v2 is the judge), no new params without frozen
  files, no veto term in any objective without a reachability probe (#LIB-015).
- Keep notes as you go (FRAMEWORK_NOTES.md in the library; the proving
  ground's LIBRARY_NOTES pattern is the template: numbered, evidence-backed,
  disposition stated). Name lessons after the incident, not the abstraction.
- Commit locally per item with messages stating the evidence
  (e.g. "Trap v2: catches stashed float+torch, passes historical false positives").
- If any item's premise proves wrong, say so with the measurement and stop
  that item (rejected proposals with numbers beat silent scope creep).

## Definition of done

Items 1-4 landed, gated, committed; 5-7 landed or scoped out with a stated
reason and a priced backlog entry. Proving ground fully green. A short report:
per item, what changed, the gate output, and what you'd do next.
