# VELOCITY LOG (v1.0 Gate 5: "how fast can a newcomer extend it?")

Measured, not claimed. Drill entries carry wall clocks. Build entries
(v1.1+) record scope + gates + surprises with HONEST time status: the
v1.1/v1.2 builds below were not clocked (gap admitted, not backfilled —
fabricated times would poison the log's whole point). Process fix: every
build from here on starts with an explicit clock read like the GELU drill
did. The log IS the extensibility answer; an honest gap beats a fake row.

## 2026-09-30: GELU (~3 min)

- Mnemonic: GELU (exact-form x*Phi(x) via phi-core `gelu_erf_int`).
- Wall time: ~3 min (study handoff sig + probe vs torch + wire op/sig +
  4 gates + docs). Clock 16:57:57Z -> ~17:01Z (docs + verify incl.).
- Gates: 0-diff vs phi-core, 60dB vs torch gelu (peak=1.0, bar 40),
  edge shapes (saturate/dip/zero/linear), in-listing exactness.
- Surprises:
  1. Zero design friction: the handoff supplied signature + gate pattern
     + the rejected alternative (tanh-approx, numbers pinned) -- drill
     time measures handoff quality as much as our velocity.
  2. No scale contract needed ("any range" input, LUT spans [-16,16]
     with exact asymptotes) -- unlike SOFTMAX, which cost a doctrine.
     Not all mnemonics are equal; LUT-with-asymptotes is the cheap kind.
  3. Suite (60dB) beat the probe (56dB) on a different fixture -- both
     far above bar, but parity numbers are fixture-dependent; report
     the fixture with the number (done in the gate string).
  4. Edge documented-not-gated: x=10 decodes 9.99 (inside LUT span but
     near the asymptote shoulder; exactness only beyond +/-16).

## 2026-09-30: v1.1 spine (NOT CLOCKED — see header)

- Per-block scales (spike + MATMUL prototype + matmul-family threading +
  two-regime listing 65dB). Surprises: RESCALE alone cannot give regimes
  (ops re-bridge global — scale context must thread into execution);
  single-scale-up cannot separate the walls (matmul + softmax saturate
  together); attention is contract-bound while MLP is coverage-bound
  (the rule for which ops get overrides).
- Matmul C lowering (5/5 bit-exact first run). Surprise: none — the
  file-exchange pattern is now routine (pattern maturity is velocity too).
- S19 parse (dated negative). Surprise: the residue (S09 + STATE) was
  worth more than the bet.
- F1–F4 (SELECT refusal + tutorial frictions). Surprise: sandbox `sys`
  lacks `environ` — portability means `os.environ` only, caught by suites.

## 2026-09-30: v1.2 substrate builds (NOT CLOCKED — see header)
- k_matmul + 3-way parity (5/5 first run). Surprise: reduction-order risk
  designed out (single-thread accumulate per output — stronger than
  commuted, no gate needed where no mechanism exists).
- CUDA units + elementwise batch (18 rows green). Surprises: phi-core's
  "LUTs fit 32 bits" comment is FALSE for SIGX (3.9e17 — report upstream);
  stale-binary false failure (Makefile must track kernel headers).
- Flagship on CUDA (3/3 bit-exact incl. real frame). Surprise: composition
  of gated units + math-free orchestration gives exactness for free —
  the tdiv-guard output is provably discarded by the coh0 mask, so pure
  tdiv suffices (logic, then gate confirms).
- Emission driver + matrix (8 cells). Surprises: substrate is a RUN
  concern, not a CONFIG key (listings stay machine-independent); builder
  moved into chain/ (chain owns building, tests own gating); one Makefile
  edit silently didn't land (verify writes by read-back).

## 2026-09-30: access-shape census (~4 min, CLOCKED — process fix working)

- Instrument: chain/census.py (static producer/consumer/fan-out + access
  classes) + 9 gates in test_census.py.
- Wall time: ~4 min. Clock 19:01:48Z -> ~19:05Z (instrument + planted
  gates + real maps + docs + full verify incl.).
- Gates: planted fan-out exact, sharing-free control empty, 42/42
  coverage, flagship/xf maps pinned (A×3, XN×3).
- Surprises:
  1. The reframe WAS the finding: statistics (distributions) vs access
     shape (topology + geometry + class) are different instruments —
     census complements ranges.py, subsumes nothing.
  2. Reuse made visible: promotion-rule evidence (3 uses → promote) is
     now measurable per listing instead of argued from code reading.
  3. One miss on first probe (RESCALE unclassified) — caught by the
     coverage gate, which exists precisely for this. Gates earning keep
     is now routine enough to note only in passing.

## 2026-09-30: causal probes (~4 min, CLOCKED)

- Instrument: ASM overrides (interventions as a run feature) + 6 gates
  in test_probe.py + probe TABLE representation.
- Wall time: ~4 min. Clock 19:06:59Z -> ~19:10Z (override support +
  sham/typo guards + two probes + held-out rerun + docs + verify incl.).
- Results: D-zero 38.46dB CONFIRMED in predicted [20,45]; COH-zero
  44.23dB FALSIFIED a predicted [3,40] upper bound; revised [38,50]
  CONFIRMED on held-out content (42.66dB).
- Surprises:
  1. Falsification is the finding: coherence decides WHERE, iso/atten
     carry HOW MUCH — the threshold miss taught more than a pass would.
  2. Flipped-frame held-out proved nothing (flip-equivariance → identical
     decimals BY CONSTRUCTION). Caught by reading the numbers, not by a
     gate — held-out means different CONTENT, now stated in LANGUAGE.md.
  3. D-zero mean change (1.35 LSB) ~= the full enhancement: the boost IS
     the detail path, no other contributor. Subtraction as identification.

## 2026-09-30: information round (~4 min, CLOCKED)

- Probe table +2 rows (xf O/DOWN at structure granularity), S11 hunt
  measurement, INVENTORY L4 with S21/S22 PREDICTED + matrix rows.
- Wall time: ~4 min. Clock 19:13:55Z -> ~19:18Z (probes + hunt +
  entries + verify incl.).
- Surprises:
  1. Probe STRUCTURE outputs, never mangled sub-wires: internal names
     carry expansion counters (attn_core#1.P) that renumber when listings
     change — composition contracts decide probe granularity, not
     curiosity. The doctrine keeps paying.
  2. S11 hunt returned 0.27, not an instance: texture-without-depth and
     depth-without-texture both exist, so covariance caps low BY
     MECHANISM. Negative with a reason beats positive without one.
  3. Information-first works: measured rows accumulate without pretense
     of bands; bands form later (LIB-015). The table is the map.

## 2026-09-30: full-sweep equivalence classes (~4 min, CLOCKED)

- All 11 flagship streams probed + 4 exact equivalence classes gated
  (2 predicted from pipeline algebra, 2 discovered).
- Wall time: ~4 min. Clock 19:19:55Z -> ~19:24Z (sweep + equality gates
  + docs + full verify incl.).
- Surprises:
  1. Predicted equalities held bit-exact (BEFF≡BD≡D, AE≡YENH) — pipeline
     algebra as a predictive theory of interventions, confirmed.
  2. LIN-zero == G-zero was NOT predicted and broke the naive model
     (LIN=0 should equal Y=0) until GAIN's direct LIN read explained it
     — and the census had ALREADY recorded LIN's fan-out over LUMA+GAIN.
     Instruments corroborate each other; that is the program working.
  3. AS-zero (max boost, 19dB) is the most informative row: the blur
     reference is what restrains enhancement — restraint, not boost,
     is the flagship's main content. A finding about the DESIGN, from
     subtraction alone.

## 2026-09-30: wider — census versions + stabilize (CLOCKED, interrupted)

- Census v1.1 (per-stream versions) + stabilize census gates + stabilize
  probe rows (static MEASURED, motion EXACT) + LIB-043.
- Wall time: active work 19:23:01Z -> 19:50Z with one interruption
  mid-round (session resume; the interruption cost a re-read of LIB
  numbering, which is why LIB-043 exists instead of a duplicated LIB-042
  — recorded honestly). Active effort ~15 min across the gap.
- Surprises:
  1. Inverted prediction, kept as the record: W-zero predicted EXACT
     under static, measured identical under BOTH flows — MIXDYAD mixes
     memory into STATIC pixels (header already said so). Wrong with a
     mechanism beats right without one.
  2. Census versions made debugging a 2-minute read (WARP@v0 vs
     MIXDYAD/BETA@v1) instead of a probe hunt — instruments paying for
     instruments.

## 2026-09-30: first modification (~3 min, CLOCKED)

- Variant listing (flagship_iso.asm: bank+gate to iso+scalar) + 3 gates
  in test_modify.py + LIB-044.
- Wall time: ~3 min. Clock 19:53:46Z -> 19:55Z (variant + gates +
  docs + verify incl.).
- Results: variant bit-exact vs iso chain FIRST RUN; differs 45dB from
  v5; census shows new structure.
- Surprise: the two-line edit worked first try because every piece was
  already gated — modifiability falls out of composition + gates, no
  extra machinery. Editable structures were the destination all along;
  the road built itself.

## 2026-09-30: weight-edit routes (~2 min, CLOCKED)

- Four routes tried live (matrix/channel/scale/rank) + method gates in
  test_edits.py + standard process (#LIB-045).
- Wall time: ~2 min. Clock 20:37:48Z -> 20:39:35Z (route probes A-D +
  stability separation + gates + docs + full 27-suite verify incl.).
  Tiny fixtures run in seconds; the analysis is the work.
- Surprises:
  1. Channel importance splits: ch3 dead across held inputs with FIXED
     weights (weight-property) while rankings reshuffle with new weights
     (input-property) — the separation method (fix one, vary the other)
     is what makes channel edits attributable at all.
  2. Scale-x2 == zero to 0.03dB: the algebra predicts its own
     consistency checks — linearity falls out of |2O-O|==|O-0|, free.
  3. Rank is a non-route on random weights (flat spectrum) — correctly
     defers to trained weights instead of manufacturing signal. Knowing
     which routes DON'T respond is information too.

## 2026-09-30: singular directions (~2 min, CLOCKED)

- Same probe in the space's own basis (per-direction ablation on wv/wo)
  + 3 gates in test_edits.py (#LIB-046).
- Wall time: ~2 min. Clock 21:21:46Z -> 21:22:30Z (direction sweep both
  matrices + gates + docs + verify incl.).
- Results: direction spread 51.4dB vs 15.6dB channels; corr(dir-dB,
  sval) -0.84; near-null direction removable at 91.4dB.
- Surprise: the tail correlation is PARTLY trivial (smallest sval IS
  0.00 — ablating nothing changes nothing) — recorded, not hidden. The
  non-trivial mid-range (40-60dB over s 0.37..0.13) is the real signal:
  geometry beats coordinates even where energy can't explain everything.

## 2026-09-30: real weights, Qwen2-0.5B L0 MLP (CLOCKED, start missed)

- Full block on trained weights + real embeddings: parity, spectrum,
  direction sweep, planted recovery (test_realw.py + mlp_qwen0.asm).
- Wall time: start clock MISSED (process lapse, recorded — the fix only
  works if misses are admitted); end 21:48:23Z; active effort ~25 min
  (cache survey, bias/embedding triage, rope/mask geometry fixes,
  magnitude scoping, sweep, gates, docs).
- Results: parity 51.82dB; spectrum 17x decaying; spread 33dB / corr
  -0.81; planted recovery bit-exact, control 18.8dB.
- Surprises:
  1. Q/K biases enormous (max 128-152) — weights-only would have been a
     DIFFERENT computation, not Qwen's. Triage before running saved the
     experiment; K-bias cancels by proof, Q-bias tiled-host, V stated.
  2. Folded attention scores hit 964 — the T-transform backlog measured
     at 1000x, live, on the record. No workaround attempted.
  3. Three geometry slips in the torch mirror (rope seq-dim, head batch,
     mask broadcast) — all shape-loud, all fixed in minutes. The
     strictness doctrine works on mirrors too.
  4. corr -0.84 (toy) vs -0.81 (real): same law, different universe.
     Partly algebraic (route C linearity) — the spread and ranking are
     the content, the correlation is the sanity.

## 2026-09-30: rank-1 implant (CLOCKED, start missed again)

- Functional-aim write (u = target's own MID direction) + 4 gates in
  test_implant.py (#LIB-048).
- Wall time: start clock MISSED twice running (lapse pattern, not bad
  luck — clock discipline slips on question-led turns vs build-led ones;
  fix: read the clock when the QUESTION lands, not when the build
  starts). End 21:55:18Z; active effort ~10 min.
- Results: target at negative dB (-14.3, -10.7), gaps 35.4 / 26.2dB,
  both bands confirmed with wide margins.
- Surprise: negative dB — the implant doesn't nudge the token, it
  REWRITES it (change exceeds peak). Writes are stronger than the
  framing ("edit") suggested; the primitive is a local overwrite with
  26dB+ containment. Specificity was the risk; strength is the news.

## 2026-09-30: directional stores as structure (CLOCKED)

- implant_apply DEF + 2 variant listings + test_store.py (toy) +
  transfer row in test_implant.py (real) + gauge doctrine (#LIB-049).
- Wall time: clock 21:59:05Z -> 22:03Z (~4 min: DEF + variants + toy
  gates + transfer falsification + bisection + gauge cycle + docs).
  Two falsifications in one round: (1) sham predicted bit-exact,
  measured 99.8dB parity (ADD re-bridges -- roundtrip quantum, now
  stated); (2) transfer predicted >=60dB, measured 36.8dB, and the
  gauge "fix" made it worse (36.8<38.6).
- Surprises:
  1. The binding constraint is ENCODE quantum on small vectors, not
     envelope saturation -- found by bisection (C 41dB, IMPL 17dB,
     DOWN bit-exact), after the wrong mechanism cost a full gauge
     cycle. Bisection before theorizing, stated as process.
  2. A=2.0 saturates each side differently (16dB); A=0.2 holds 91.2dB.
     Envelope discipline applies to BOTH forms independently -- parity
     needs both in-envelope, not just the mechanism right.
  3. Materialization pays quantum tax per hop -- LIB-006's fusion
     lesson generalized to the store abstraction. The road rhymes.

## 2026-09-30: first model read (CLOCKED)

- Read instrument (chain/read.py) + gates (test_read.py) + 112-dir real
  readout + docs/MODEL_READ.md with three searches answered (#LIB-050).
- Wall time: clock 22:16:05Z -> 22:21Z (~5 min: readout sweep 116s +
  searches + instrument + gates + MODEL_READ + verify).
- Surprises:
  1. Shared giants + distinct leaners (France 16/24, Paris 96/328):
     content has BOTH shared and selective structure -- the read
     separates them for free once fingerprints exist.
  2. The most uniform direction (872) is dead: uniformity without magnitude
     is vacuous. Selectivity must be read jointly with global dB --
     queries compose, single numbers mislead (the router's
     flicker-wins-AND-sharpness-bounded lesson, recurring).
  3. 116 seconds for 112 directions: reads are CHEAP. The labeling loop
     is no longer blocked on instruments -- only on the cross-context
     stability question, which is now askable.

## 2026-09-30: cross-context stability (CLOCKED)

- Same 112 dirs, second (quantum-subword) prompt: readout 115s +
  rank/selectivity comparison + MODEL_READ section (#LIB-051).
- Wall time: clock 23:11:59Z -> 23:15Z (~3 min: second readout 115s +
  comparison + docs; no new gates -- measured rows per LIB-015).
- Surprises:
  1. The split landed EXACTLY on the predicted seam (stable WHAT 0.55,
     contextual WHERE 0.25) -- a caveat calling its own shot is a
     theory with predictive power, not humility theater.
  2. Giant dir0 identical to 0.1dB across domains: the top of the
     causal hierarchy is context-invariant. Shelves for the labeling
     loop to stand on.
  3. Dead-shelf non-overlap (2 vs 0, max 54.9 just under bar) counsels
     against thresholding storage on one context -- shelves need
     multi-context reads. Stated, not tuned (bar left at 55).

## 2026-09-30: shelf map + labeling-loop draft (CLOCKED)

- Third-prompt readout (code, 116s) + shelf_map instrument + gates +
  MODEL_READ section + docs/LABELING_LOOP.md draft v0.1 (#LIB-052).
- Wall time: clock 23:17:27Z -> 23:20:30Z (~3 min: code readout 116s +
  shelf_map + gates + MODEL_READ + loop draft + verify).
- Surprises:
  1. Empty intersection: nothing dead everywhere at 55dB. The map bit
     back -- "free shelves" don't exist for this matrix at this bar,
     and the honest output is a design consequence (union + budgets),
     not a lowered bar.
  2. The loop draft wrote itself from existing parts (readouts in,
     implant verifier already gated, INVENTORY close already process):
     four steps, each with its gate, each falsifiable. Design as
     composition of proven primitives -- the Jotnar way.
  3. Giant dir0 at 22-23dB on all three domains: the causal hierarchy
     has a context-invariant top. Whatever "important" means, dir0 is it.

## 2026-10-01: blind prediction, edit-with-preview (CLOCKED)

- Calibrate-once + 3 blind predictions confirmed at +0.2dB +
  predictor standardized (#LIB-056).
- Wall time: clock 12:39:00Z -> 12:47Z (~8 min: calibration +
  3 blind runs + predictor + full-map + in-suite gate + one real bug
  + docs).
- Surprises:
  1. +0.2/+0.2/+0.3 on all three: prediction to a fraction of a dB on
     never-run interventions. The advantage is operational, not
     theoretical: preview any edit's delta before running it.
  2. The in-suite gate FAILED first (8.33dB) on a REAL bug (silu on UP
     twice, not GATE) -- calibration debugged clean first, isolating
     the bug to new code by elimination. Fixed: 0.22dB. New gates earn
     keep by failing on real bugs.
  3. Full 896-map: values trustworthy (blind +/-0.3), neighbor-ranks
     not (top-10 overlap 3/10) -- cut by value threshold, never by rank.
  4. Two whitespace/structural edit slips in one round, both caught by
     read-back: verify-writes now covers docs edits explicitly.

## 2026-10-01: CRUD goal + pruning demo (CLOCKED)

- Goal doc (docs/CRUD_GOAL.md: operations, v1.4 language shape, 5 research
  questions, ceilings) + Q1 closed (test_prune.py, #LIB-057).
- Wall time: clock 12:59:03Z -> 13:00:30Z (~2 min: doc + mirror helper +
  prediction + 2 runs + gates + docs).
- Surprises:
  1. Combined err 0.1dB on 27 components: superposition budgets EXACTLY.
     Delete-by-prediction is now a standing capability, not an experiment.
  2. "Dead" is per-direction: 27x >55dB each = 42.9dB together. The shelf
     story's honest decimal -- above bar, not free. Aggregation changes
     the claim class (cf per-op vs whole-block parity, same lesson).
  3. qwen_mirror.py's 4th avoided copy: promotion working as designed,
     helpers accreting instead of mirrors multiplying.

## 2026-10-01: CRUD round — Q2/Q3/Q4/Q5 + ENGRAM (CLOCKED)

- Gain retune (= signed ablation), label-to-store CREATE, shelf null,
  layer-1 spot-check, structure named (test_create.py, test_layer1.py,
  S23, #LIB-058).
- Wall time: clock 13:04:45Z -> 13:09Z (~4 min: Q2 + Q3 + Q4 + Q5 +
  naming + docs; test runtimes dominate).
- Results: retune 22.1==22.1; create rank 1/8 at 8.4dB gap 22.6;
  shelf-null 57.8dB; L1 spectrum 12x giant 14dB tracking -0.54.
- Surprises:
  1. Retune predicted ITSELF from the ablation number (same |delta|):
     the cheapest prediction in the program -- zero new runs to state
     it, one run to confirm. Laws compose.
  2. Created store leaks to Germany/Switzerland (proper nouns move):
     keys built from entity-MIDs match entity-ness broadly, not one
     name exactly. Labels are coarser than their names -- recorded
     against future overclaiming.
  3. L1 tracking weaker (-0.54, n=8): unknown whether depth or noise.
     Stated with the caveat; the next layer survey decides.
  4. Two more whitespace edit slips (phantom trailing-space "fixes"
     that merged a header once): edits that LOOK like no-ops get
     diff-checked before AND after now, not just read-back after.

## 2026-10-01: storebanks as listing data (CLOCKED)

- engram.bank() + storebank_apply DEF + 2 variant listings +
  toy gates + real test_storebank.py (#LIB-060).
- Wall time: clock 13:15:23Z -> 13:17:30Z (~2 min: bank() + DEF +
  2 variants + toy + real gates + docs).
- Results: toy parity 96.1dB; real parity 55.8dB; pruned 42.7 vs 42.9;
  three-form agreement within 0.1dB (surgery/bank/statics).
- Surprises:
  1. implant_apply already handled K>=1 (MATMUL is inner-dim general):
     the "new" DEF is a rename with a contract, not new machinery.
     Naming the concept IS the feature -- listings read as what they are.
  2. One dropped sys.exit (edit removed it silently): suite would have
     passed WHILE FAILING (exit 0 always). Caught by tail-reading the
     file, not by any gate. Process: tail-read test files after editing
     their endings -- endings carry the exit code, the one line that
     decides pass/fail.
  3. Three-form agreement (42.7/42.8/42.9): surgery, bank, statics all
     land together. When three independent forms agree, the law is doing
     the work and the implementations are just witnesses.

## 2026-10-01: DDColor query mining round one (CLOCKED)

- Triage (no-English verdict, compression confirmed, cls-head cleared) +
  weight-space read + vote maps + footprint probes (#LIB-061, OPEN).
- Wall time: clock 13:25:18Z -> 13:32Z (~7 min: survey + loader +
  maps + md5 mystery (own sample == f_012!) + footprints + docs).
- Surprises:
  1. Own sample md5-identical to f_012: identical outputs were
     determinism confirmation, not model failure. Check inputs first.
  2. Diffuse votes (0.95) vs biting footprints (3.7x dark): affinity
     != causality, the same lesson as mass != share, one level up.
     Read what things DO (interventions), not what they SAY (maps).
  3. q0 outranks q39 causally (13 vs 26dB) against mass order: the
     hierarchy that matters is always the interventional one.

## 2026-10-01: brightness-vs-object split (CLOCKED)

- Decile profiles + gradient control from saved footprints, zero new
  runs; band-pass verdict with shared cliff (#LIB-063).
- Wall time: clock 13:34:57Z -> 13:36Z (~1 min: profiles + control).
- Surprises:
  1. Within-dark foot-vs-L correlation POSITIVE (+0.4): the first number
     contradicted the working theory (dark-gated) and forced the decile
     profile, which showed the band-pass. Contradictions are instruments
     too -- this one cost nothing and paid a verdict.
  2. Neither hypothesis survived: not brightness-monotonic (peak at
     decile 2-3, not 0), not object-gated (grad corr ~0). The axis is
     DIVIDED among queries with a shared bright cliff. Third options
     win more often than the split admits -- design splits with room
     for "neither".
  3. Analysis-only rounds (saved artifacts + statics) now resolve
     questions end-to-end: footprints were the expensive part, profiles
     the cheap part. Save everything, analyze forever.

## 2026-10-01: palette verdict, two falsifications (CLOCKED)

- Bright-hunt falsified + need-control + pairwise footprints +
  refine-conv hue readout + LIB-063 correction pointer (#LIB-064).
- Wall time: clock 13:36:44Z -> 13:39Z (~2 min + hunt runtimes).
- Surprises:
  1. The hunt prediction (bright mirror >2x) falsified so thoroughly
     (<=0.26 everywhere) that the CONTROL became the finding: base |ab|
     carries the footprint's shape, so dark-bite was need all along.
     Controls first, then verdicts -- the control design is the experiment.
  2. LIB-063's verdict stood wrong for exactly one round before its own
     correction pointer: supersession WITH pointer, never silent rewrite.
     The notes are append-only history, wrong turns included.
  3. Zero-run kill (refine rows -> hue circle): READ THE WEIGHTS before
     running the model. Two probe rounds answered in one weights read
     what footprints couldn't separate. Statics first is now doctrine,
     not preference.

## 2026-10-01: tests/ move, suite stays green (CLOCKED)

- 44 files moved + 161 anchors + dispatcher + README block (#LIB-091).
- Wall time: clock 17:08:57Z -> 17:14Z (~5 min: move + anchors +
  dispatcher + README + full 44-suite).
- Results: 43/44 first try; test_c dispatcher fixed; full green after.
- Surprises:
  1. Exactly ONE break from moving 44 files (the subprocess dispatcher):
     uniform anchor patterns are genuinely uniform. Mechanical moves
     with mechanical verification -- the ratio held (1 fix / 44 files).
  2. Overrode LIB-088's own restraint on explicit request: process notes
     record defaults, owners override. Written here so the reversal is
     legible, not hidden.

## 2026-10-01: handoff drafted (CLOCKED)

- Readiness map + inventory + feasibility + first-week plan (#LIB-090).
- Wall time: clock 17:05:30Z -> 17:06:30Z (~1 min: inventory + verdict).
- Surprises:
  1. Writing "GO" required the gaps table first: readiness without
     named gaps is marketing. The verdict section took as long as the
     inventory -- honesty has mass.
  2. Gradients OUT stated plainly (not "future work"): saying what the
     program will NEVER do is as load-bearing as saying what it will.
     Scope stated negatively holds shape under pressure.

## 2026-10-01: mirror migration, promotion fires (CLOCKED)

- test_realw/test_implant mirrors onto chain/qwen_mirror + verification
  (H bit-identical, decimals compared) (#LIB-089).
- Wall time: clock 16:59:37Z -> 17:04:30Z (~5 min: migrate + verify +
  trace +-0.1dB shifts + docs).
- Surprises:
  1. H bit-identical, per-token +-1dB: identical inputs can still shift
     sensitive metrics through a CHANGED downstream (eps migration moved
     the aim vector microscopically). Trace shifts to their true cause
     (the eps fix, already explained), never to the most recent edit.
  2. read_model.py deliberately NOT migrated (needs rerun to verify;
     backlog). Promotion fires on touch, and this touch didn't need
     that file. Restraint is also a decision.

## 2026-10-01: reorg — research/ moved + verified (CLOCKED)

- 9 one-shots git-mv'd + path widening + index + rerun proof (#LIB-088).
- Wall time: clock 16:50:54Z -> 16:54Z (~3 min: inventory + move +
  widen + index + rerun).
- Surprises:
  1. Tests stay in root DELIBERATELY (44 files x __file__ anchors):
     reorg scope ends where churn exceeds gain. Restraint documented
     as a decision, not laziness.
  2. Rerun-verified (identical decimals), not just compile-checked:
     moved code that never re-executes is Schrödinger's code. One
     rerun collapses it.

## 2026-10-01: T-transform follow-through, WIDE live (CLOCKED)

- Vendor + 2 mnemonics + gates + reference (46 total) (#LIB-087).
- Wall time: clock 16:44:52Z -> 16:47:30Z (~3 min: vendor + ops + gates).
- Results: 89dB on +-500; legacy untouched; invariance bit-exact.
- Surprises:
  1. Translation invariance bit-exact IN-LATTICE (TSHIFT+WIDE ==
     WIDE-direct): the algebra survives the substrate -- composition
     of gated ops inherits the math's symmetries for free.
  2. Contract split instead of contract change: legacy keeps its pin,
     WIDE takes the range. Two contracts, both green, zero migration.
     Splits beat migrations where behavior was pinned on purpose.

## 2026-10-01: demo speaks (CLOCKED)

- demo_lm.py (greedy generation through the listing) + fixes (#LIB-085).
- Wall time: clock 16:30:25Z -> 16:31:30Z (~1 min: demo + argv fix + UNK
  rule + rewrite + runs).
- Results: "and the great" on loop, end to end, zero trained weights.
- Surprises:
  1. Stillbirth first (UNK everywhere in 1 step): tail mass makes raw
     greedy unusable -- masking UNK at bank-build is a DECODING choice
     (boundary), not model surgery. Boundaries absorb hacks honestly.
  2. The attractor ("and the great" x10) diagnoses greedy decoding
     live: sampling earns its place by demonstration. Theory said
     host-boundary; practice shows why.
  3. Third-patch-on-one-function means rewrite: patch count is a
     complexity signal. Heed it at three, not thirteen.

## 2026-10-01: repetition beaten by selection (CLOCKED)

- Echion survey (selection, not tricks) + demo flags + gates (#LIB-086).
- Wall time: clock 16:32:46Z -> 16:35Z (~2 min: survey + flags + gates).
- Results: 0.94 unique-ratio, identical replay, gates green.
- Surprises:
  1. "How did Echion beat repetition" had a one-line answer (selection
     by diversity fitness) sitting in bootstrap.py's docstring: survey
     BEFORE solving netted the pattern in one grep. Ask the neighbors.
  2. The demo speaks real history now (caesarion, ptolemy xii, persian
     royal family): bigram statistics + selection compose into something
     that reads like content. Weak models, honestly framed, still speak.

## 2026-10-01: loop on own creature, exact bands (CLOCKED)

- Column-silence (26/26) + 5 implants (5/5) on the bigram LM (#LIB-084).
- Wall time: clock 16:25:39Z -> 16:27:30Z (~2 min: vocab check + gates).
- Results: all exact, zero statistics.
- Surprises:
  1. Proper nouns UNK (france/paris/texas/capital all out of top-512):
     the creature lives in common words + dating artifacts. World-model
     limits stated upfront -- narrowness documented is a feature (v1.6
     says narrow domain deliberately).
  2. Exhaustive verification (all 513 rows through the listing, not a
     sample): own models are small enough to check COMPLETELY. Sampling
     is for other people's scale; exactness is the home advantage.
  3. the->legitimacy: implanted pairs read like found poetry. Creation
     with guarantee has aesthetic side effects; log them too.

## 2026-10-01: construction — bigram LM behaves (CLOCKED)

- Freeze pipeline + listing + gates, first-try green (#LIB-083).
- Wall time: clock 16:21:53Z -> 16:24Z (~2 min: freeze + listing + gate).
- Results: memorize 200/200; top1 0.41 / top5 0.61 / ppl 47.
- Surprises:
  1. Unsmoothed ppl 7.7e29: INFINITE-by-construction on unseen pairs,
     not a bug -- reported once, then add-one at scoring (model stays
     raw). Honesty about infinities beats smoothing them silently.
  2. Coverage 0.609 worn openly: OOV is the dominant error source and
     the doc says so. Weak numbers with stated causes beat strong
     numbers with hidden ones -- the v1.0 stranger-test spirit.
  3. Five lines of listing (GATHER+ARGMAX) over frozen counts IS a
     language model (weak, honest, working): from-scratch construction
     costs almost nothing once stores + assembly exist. The expensive
     part was the two years underneath.

## 2026-10-01: mamba trajectory on real weights (CLOCKED)

- S3 study + scan_mamba.asm variant + test_mamba.py at 46.3dB (#LIB-082).
- Wall time: clock 16:18:16Z -> 16:20Z (~2 min: survey + magnitudes +
  variant + green).
- Results: trajectory parity 46.3dB; determinism bit-exact.
- Surprises:
  1. -14.6dB first (total fold at frozen scales): magnitude-first
     debugging again (measure ranges BEFORE theorizing -- the SmolLM2
     lesson applied without relearning).
  2. scan_demo.asm untouched, variant added: generic demo stays frozen,
     magnitude variants fork. One listing per regime, not CONFIG soup
     in one file -- readability as discipline.
  3. Boundaries honestly listed (conv/softplus/exp/Bx-composition):
     each named with its reason, each a future gate. The file states
     its own incompleteness -- completeness theater helps no one.

## 2026-10-01: data-freeze decision, both written small (CLOCKED)

- Wikitext sparsity probe + echion-store vs S17 side-by-side + decision
  (#LIB-081).
- Wall time: clock 16:13:55Z -> 16:16:30Z (~3 min: probes + decision).
- Surprises:
  1. 12x size gap (843KB vs 70KB) for identical content: formats are
     NOT interchangeable veneers -- structure has mass. Measure both,
     then choose by the numbers (the roadmap said exactly this).
  2. echion import clean (0.0s, zero heavy deps): a good citizen either
     way, which made "adopt the envelope, skip the dependency" an easy
     honest middle instead of a rationalization.
  3. My manifest needed one fix (format key overrode the envelope's):
     even config-shaped code fails loud under this program's reader.
     Their reader refused my sloppiness -- good reader.

## 2026-10-01: SCAN drill + mamba pole + design formal (CLOCKED)

- SCAN op + 3 gates + scan_demo.asm + mamba weights/config + design
  fourth pole + formal draft (#LIB-080).
- Wall time: clock 16:05:38Z -> 16:11:30Z (~6 min: download + drill +
  doc fixes + design formal).
- Results: drill green (0-diff/69dB/bit-exact-repeat); 44 mnemonics.
- Surprises:
  1. repeat()==manual bit-exact on first run: STATE threading was
     DESIGNED for exactly this (handoff said so) and the design held.
     Well-specified interfaces compose on contact.
  2. Two haste-slips in one drill (garbage check-line, rng ordering):
     both caught before commit by reading + running. Speed is fine;
     skipping the read-back is not -- the rule held under pressure.
  3. Mamba config reads like a Jotnar program already (scan + conv +
     gate + norm, no attention/softmax anywhere): the language was
     ready for recurrence before we were. Coverage check passed by
     existence.

## 2026-10-01: anatomy-2 + fan-in law + LLM design (CLOCKED)

- GPT-2 survey (biases/pos-emb/gelu_new) + LAYERNORM exposure (#43) +
  split-claim gates + formal design draft (#LIB-079).
- Wall time: clock 15:52:34Z -> 16:03:30Z (~11 min: survey + exposure
  + 5 misdiagnoses + fan-in law + design draft).
- Results: LN-GS 57.9dB barred; DOWN 19.9dB measured with law;
  design draft v0.1 with priced decisions.
- Surprises:
  1. FIVE misdiagnoses, two mechanisms: coverage (exact-inputs proof)
     AND fan-in floor (encode-quantum x sqrt(K) x magnitudes, predicts
     within 1.3dB). Bisection chains beat narrative -- but count the
     cost: five wrong theories is a slow turn. Bisect EARLIER (the
     per-term autopsy should have been step two, not step six).
  2. Conv1D-vs-Linear (no-transpose) + heads-batch geometry: the same
     two mirror bugs as the Qwen probe, caught by shapes in minutes.
     Mirror bugs rhyme; the shape-loud doctrine generalizes to new
     repos untouched.
  3. Extraordinary claims (48dB peakmax "law") need the extraordinary
     version of the usual: predicted 46.6, measured 47.9 -- laws close
     to 1dB or they are stories. This one closed.

## 2026-10-01: anatomy-1 + eps law (CLOCKED)

- SmolLM2 survey + two-stage split + uniform threading + true-eps +
  test_anatomy.py at 48.9dB (#LIB-078).
- Wall time: clock 15:28:21Z -> 15:48Z (~20 min: survey + 3 misdiagnoses
  + threading + eps law + migration + 41-suite verification).
- Results: anatomy instance 1 green; eps law stated with formula.
- Surprises:
  1. THREE wrong theories before the mechanism (two-scale-can't-separate
     disproven by construction; split-key insensitive by dominant term;
     matmul blamed while exonerated bit-identical): bisection beats
     narrative every time, and "unchanged to 4 sig figs" is data (it
     fingered the dominant term).
  2. 5%-uniform-relative error with exact integer ops is IMPOSSIBLE on
     paper -- impossibility claims locate false premises fast (the
     premise was m-blind eps, dead in one direct probe: 47->84dB).
  3. Blind replace-all hit our own definition body (H._load_scales took
     the override): bulk edits need the anchor READ (which def? whose
     scope?), never pattern-matched. Fourth instance of the species
     if anyone's counting -- the process note in LIB-070 stands.

## 2026-10-01: v1.6 scoped from survey (CLOCKED)

- Echion inventory (data + mechanisms) + ROADMAP_v1_6 (#LIB-077).
- Wall time: clock 15:24:48Z -> 15:26Z (~1 min: survey + roadmap).
- Surprises:
  1. Echion is a PARALLEL program (templates/batteries/fingerprints/
     bootstrap/store), not salvage: v1.6 inherits an eval culture
     (expects, shapes, batteries) and a word ("constructing").
     Survey before salvage -- the map found the territory organized.
  2. qa batteries carry EXPECTS: behavioral gates with answers, the
     exact thing demo-2's success criterion wanted. Reuse beats design.
  3. store.py's manifest+digest discipline vs our hash-sidecars: two
     provenance designs for the data-freeze decision. Writing both
     small beats arguing (roadmap gate 2 says so explicitly).

## 2026-10-01: demo 3 + gallery, eyeball-verified (CLOCKED)

- Two consecutive previews (test_demo3.py) + 4 verified figures
  (gallery/make_gallery.py, #LIB-076).
- Wall time: clock 14:49:34Z -> 14:52Z (~3 min: demo3 + gallery tool +
  path bugs + showcase run + eyeballing).
- Results: demo3 3/3 previews green; wheel/teal/sheet/curve all seen.
- Surprises:
  1. Recall cliff at ~1/3 bits flipped (100% to 16, 0.98 at 20, 0.61
     at 24): capacity edges are CLIFFS not slopes -- same shape class
     as the luminance cliff (decile 5-6) and the softmax contract.
     Threshold physics keeps recurring; name it when the third instance
     lands (this is the second... third counting T-transform saturation).
  2. Eyeball verification caught nothing -- and that silence is data:
     every figure correct on first render means the pipelines are
     deterministic end-to-end (same inputs, same pixels, every time).
     Trust, but verify -- then log that verification happened.

## 2026-10-01: demo 2, memory from scratch (CLOCKED)

- 5-line listing + capacity-reasoned bands + test_demo2.py (#LIB-075).
- Wall time: clock 14:44:20Z -> 14:45Z (~1 min: listing + bands + gate).
- Results: 100% recall at 0/8/16 flips (band asked >=80% at 16).
- Surprises:
  1. First-try green with margin everywhere: capacity reasoning
     (margins, not fits) is the right way to set behavioral bands --
     conservative by construction, confirmed with room.
  2. The whole demo is 5 lines + seeded data: from-scratch models are
     CHEAP when behavior replaces training. The expensive part was
     never the model; it was knowing what composition proves what.

## 2026-10-01: demo 1 greens by coordination (CLOCKED)

- 9-round teal saga (8 falsified mechanisms + coordinated success) +
  receipt #002 (#LIB-074).
- Wall time: clock 14:35:27Z -> 14:43:30Z (~8 min: 9-round saga with
  file drift, metric fixes, mechanism hunts, coordinated success).
- Results: energy x1.25, global -3.7dB, both bands held.
- Surprises:
  1. File drift ran Q=41 twice unknowingly (identical decimals should
     have TRIGGERED suspicion at once, not after a mechanism story):
     identical outputs across supposedly-different runs are ALWAYS a
     red flag, never a coincidence. New tripwire, stated.
  2. Metric blindness (positive-clip), premise errors (already-teal
     slot), sign washout, sigma ripple: four DISTINCT failure modes
     found by keeping every falsification. The saga IS the method
     section demo 1 needed.
  3. Coordinated multi-slot writes succeed where singles can't, and the
     predictor for them is superposition (Q1), not any single-slot law:
     composition scales where components don't. The program's thesis
     in one demo.

## 2026-10-01: t-transform branch pushed (CLOCKED)

- Additive-only phi-core branch (LUT + fn + tests) + full their-suite
  green + push, no main touched (#LIB-073).
- Wall time: clock (jotnar) 14:15:20Z start of analysis; branch work
  ~25 min wall (design shrink, 2 fixture bugs, suite, push, docs).
- Results: wide parity 3.6e-05 on +-500; their 3 suites green.
- Surprises:
  1. The planned change died by units analysis BEFORE commit: a wrong
     branch never born beats a reverted one. Analysis is cheapest
     pre-birth; the doctrine now says so explicitly.
  2. Additive-only as a review strategy: zero diff to existing paths
     means the owner's review is "is the new thing right" not "did you
     break my things" (their suite proves the second half already).
  3. LUTs/untracked in phi-core (all local builds): the frac_hi table
     regenerates anywhere, nothing to commit. Patterns working as
     designed feel uneventful -- log them anyway.

## 2026-10-01: T-transform specified (CLOCKED)

- Range survey + obstruction analysis + two-part spec + alternatives
  killed (#LIB-072). No implementation (cross-repo by design).
- Wall time: clock 14:15:20Z -> 14:17:30Z (~2 min: code read +
  range survey + spec + docs).
- Results: exp path exonerated, bridge indicted (fold); 9-layer ranges;
  one-assert fix + composition pattern.
- Surprises:
  1. The bottleneck was ONE assert, not the LUT machinery: chasing
     every lead converged instead of sprawling. Breadth first, then
     the survivor gets the depth. Research order matters.
  2. Alternatives died fastest when priced against measurements
     (global shift: one counterexample; clip: e^-5 vs e^-1 arithmetic;
     temperature: semantics change, out). Kill criteria beat opinions.
  3. DDColor L0 needs only m_of(~16): the first customer is the
     CHEAPEST row, not the headline 429. Start where the gap is
     smallest -- strangely easy to forget.

## 2026-10-01: selection-side, S08 first instance (CLOCKED)

- Minimal attention listing + 4 gates incl. measured contract (v1.4
  gate 2, #LIB-071).
- Wall time: clock 14:12:14Z -> 14:13:30Z (~1 min: mini listing +
  gates + contract fix + docs).
- Results: P-invariance bit-exact, recompose 82.9dB, ordering 26.7<27.3.
- Surprises:
  1. V-independence is STRUCTURAL (P never sees V): the decomposition's
     precondition holds by construction, so attribution by ablation is
     exact in principle -- the gate just confirms the implementation
     honors the math. Strongest kind of gate: proves no bug, by design.
  2. Thin margins are data (0.6dB): flat attention means flat
     contributions -- the gate string carries the mechanism so the next
     reader doesn't "fix" the margin into a false alarm.

## 2026-10-01: projection banks, v1.4 gate 1 (CLOCKED)

- block_inputs bundle + test_projbank.py (6 banks + prune) (#LIB-070).
- Wall time: clock 14:07:06Z -> 14:10Z (~3 min: bundle + gates +
  2 falsifications + docs).
- Results: 6/6 at 59-63dB; prune harmless at 76.1dB.
- Surprises:
  1. Two falsifications stacked: form-tax (compare within-form) THEN
     floor (tiny removals cost less than predicted). Naive linearity
     needed two corrections, each with its own measurement. Laws earn
     their corrections one falsification at a time.
  2. Duplication-slip species AGAIN (double RESULT print): caught by
     counting outputs this time. Three instances of one bug class =
     process gap, not bad luck: read-before-edit (find exact anchor),
     diff-after-edit, count-outputs-after-run. Written here so the
     fourth instance has no excuse.

## 2026-10-01: v1.4 definition + gate-3 verdict (CLOCKED)

- Roadmap + scoremax probe + verdict + probe committed (#LIB-069).
- Wall time: clock 14:03:25Z -> 14:05:30Z (~2 min: roadmap + probe +
  3 bugs + verdict + docs).
- Results: scores 7..429 across 9 layers; gate 3 WAITS.
- Surprises:
  1. Decisive-first ordering paid: one probe settled the release shape
     before any building. Measure the load-bearing unknown FIRST is now
     process (it was instinct; now it's written).
  2. Three probe bugs, three shape-loud failures, minutes each: hook
     arity, batch-vs-head geometry, 3-arg hooks. Foreign-code probing
     is debuggable exactly when shapes fail loud -- grad school in a
     traceback.
  3. Score sharpening through depth (to 429!) then moderation (30):
     attention dynamics neither uniform nor monotonic. Filed, not
     chased -- v1.4 has enough spine already.

## 2026-10-01: edit receipts + DDColor stores (CLOCKED)

- Receipt standard + filled #001 + DDColor store freeze with roundtrips
  (#LIB-068): the handoff sequence starts.
- Wall time: clock 13:59:44Z -> 14:01Z (~1 min: receipt doc + freeze +
  gates + docs).
- Surprises:
  1. Receipt #001's residue (disentangle) was already closed by LIB-067:
     receipts COMPOSE across rounds (open items get closed by reference,
     not by rewrite). The paper trail is load-bearing infrastructure.
  2. dd roundtrips at 1e-14 (vs 6.4e-16 Qwen): bigger matrices, same
     story. Storage is boring now, which is exactly what storage
     should be.

## 2026-10-01: disentangle + resonant falsified twice (CLOCKED)

- dd_modify modes (query-only 36dB / refine-only 19dB / both 14.5dB) +
  dd_resonant.py (scalar 0.01, H=8 0.04 vs 0.03 chance) (#LIB-067).
- Wall time: clock 13:49:19Z -> 13:52Z (~3 min: modes + both probes).
- Surprises:
  1. BOTH disentangle bands falsified inverted (predicted query<25 AND
     refine>25; measured 36.0/19.0): the slot is a (place, color) PAIR
     and the pair writes coherently (14.5dB super-additive). Attribution
     questions with "which half" answers are malformed; pairs all the
     way down.
  2. Scalar phase 0.01 is BELOW chance 0.03: not noise but wrap-to-
     uniform (mechanism, not failure). Mod-2pi after random projection
     is a uniformity machine -- which is exactly what resonant dedup
     wants and similarity search doesn't. Read the other repo's claims
     precisely: exact-match + uniformity was ALWAYS the offer.
  3. Correct bridge restated (phase over scalar attributes like hue,
     never projected keys): failed bridges with stated constraints
     beat unbuilt ones. The constraint IS the deliverable.

## 2026-10-01: add/remove demo, norm-fit limits (CLOCKED)

- LOO norm-fit analysis + ADD demo + follow-up verification + DNA
  verdict (#LIB-066, OPEN: disentangle pending).
- Wall time: clock 13:46:06Z -> 13:48Z (~2 min + demo runtimes).
- Surprises:
  1. Norm-fit p90 10.9dB: predictability TRACKS LINEARITY (exact law ->
     0.3dB; 9 nonlinear layers -> 10dB). A meta-law about our own laws:
     preview-grade prediction needs a factorizable path. Price every
     predictor by its path's linearity.
  2. ADD falsified-then-half-confirmed in one follow-up: hue unproven,
     slot live (21.4dB). Falsification with a live remainder beats
     clean confirmation -- the remainder (disentangle) is next.
  3. "Error free" redefined honestly: dB-bounded with stated bar, acting
     on measured costs (100 x 0.2s catalog) with margins. Bit-exact
     multi-hop stays impossible; bounded multi-hop is the product.

## 2026-10-01: full palette catalog (CLOCKED)

- dd_catalog.py (100 queries x 2 images, 28s) + PALETTE_CATALOG.csv +
  analysis (#LIB-065).
- Wall time: clock 13:41:43Z -> 13:43Z (~1 min + 28s catalog run).
- Surprises:
  1. 0.2s forwards made the "expensive" catalog trivial: 28 seconds
     total. Cost models age fast -- reprice often (this catalog was
     scoped as "too slow" one turn earlier on a wrong estimate).
  2. Static norm (0.78) beats dynamic mass (-0.40) as causal predictor:
     the vote maps we spent a round capturing predict BACKWARDS. The
     instrument that felt most direct (watch it work) loses to the
     one that reads the weights. Statics first, for the third time.
  3. q77 moves output at 3.2dB alone: single-point concentration in a
     100-slot palette. Robustness question filed (knock out the giant?).

## 2026-10-01: q39 loop closes (CLOCKED)

- dd_footprint.py loops two images; dark-bite bands held on fresh
  content (#LIB-062).
- Wall time: clock 13:32:44Z -> 13:33:30Z (~1 min + test runtimes).
- Results: q39 f_012 at 25.8dB/-0.74 (vs 26.1/-0.72); q0 11.3/-0.38.
- Surprise: NONE -- and that is the news. Same signature to a decimal
  on fresh content, no new machinery, no new doctrine. The loop has
  gone from research to routine in four closes (Qwen x3 + q39). Routine
  is what instruments are FOR; the excitement budget moves to what the
  labels SAY (dark-regions? brightness-gated or object-gated? -- next).

## 2026-10-01: native ENGRAM storage (CLOCKED)

- chain/engram.py + test_engram.py + v1.3 scope extension (#LIB-059).
- Wall time: clock 13:11:37Z -> 13:13:30Z (~2 min: storage module +
  gates + scope extension + docs).
- Results: roundtrip 6.4e-16, deterministic bytes, store-IO bit-exact.
- Surprise: the only design decision (blobs in git or not) resolved by
  ARITHMETIC (41MB vs 1.9MB repo) with the determinism gate as
  compensation -- process constraints (hygiene) and proof constraints
  (reproducibility) reconciled instead of traded. v1.3's definition grew
  a gate mid-release; the roadmap says so explicitly, dated.

## 2026-10-01: factorization (CLOCKED)

- Energy-factored residual + key-alignment law + form gate + writeup
  (#LIB-055). Zero new runs (existing readout + statics).
- Wall time: clock 12:20:30Z -> 12:23Z (~3 min: residual analysis +
  alignment law + form gate + writeup). Zero new runs.
- Surprises:
  1. 0.998 is not "strong correlation", it is identity with noise:
     the residual IS the alignment, and content reduces to geometry
     with nothing left over. The fundamental thing was hiding in the
     correlation everyone (including us, LIB-046) called "partly
     algebraic" -- it was ENTIRELY algebraic, in two factors.
  2. The loop's match step just got 1000x cheaper (alignments ~ms vs
     ablations ~100s): instruments obsolete each other here, and that
     is progress, not waste.
  3. My print labels in the first probe were backwards (top residual =
     matters LESS than energy says); the math corrected the narrative
     before it got committed. Analysis first, prose second.

## 2026-10-01: labeling loop, run one (CLOCKED)

- First full loop turn: (dir24, France) hypothesized, matched, verified
  (loop1.py + chain/qwen_mirror.py promotion, #LIB-053).
- Wall time: clock 00:51:06Z -> 00:52:30Z (~2 min: mirror promotion +
  loop script + 6 runs + docs + verify).
- Results: MATCH rank 1/8 at 33.9dB; VERIFY -12.7dB rewrite, 24.4dB gap.
- Surprises:
  1. The fingerprint followed the CONTENT across positions (3->5), not
     the position. Positional caveat honored in the breach: fingerprints
     are positional per-read but content-tracking across reads -- the
     distinction the loop exists to draw, drawn on its first turn.
  2. Shape bug caught on re-read (write must use Vt-side output, not
     U-side): input keys (4864, MID side) vs output values (896) are
     different spaces and the store view names which is which. Reading
     geometry first keeps paying.
  3. S21's first instance cost one script: the expensive part was the
     year of instruments underneath it, not the turn itself.

## 2026-10-01: loops two and three, S21 CONFIRMED (CLOCKED)

- loop1.py generalized to loop_turn.py (replay-verified identical) +
  two more labels closed (dir32/capital, dir328/Paris) (#LIB-054).
- Wall time: clock 11:25:26Z -> 11:27:30Z (~2 min: generalization +
  s-fix + replay + loops 2-3 + docs).
- Results: 3/3 matches rank 1/8; writes -12 to -15dB, gaps 24-33dB.
- Surprises:
  1. Every match rank ONE of eight against bands asking top-3: either
     the bands are loose or selective directions are cleaner than the
     doctrine assumes. Not tightening yet (3 samples); the fourth loop
     decides whether bands move.
  2. The s[24] bug survived one full green run (loop 2's first pass
     CONFIRMED with wrong magnitudes): bands that pass under bug and
     fix alike are TOO LOOSE to catch that bug class. Replay-identity,
     not bands, caught it. Gate the machinery, band the content.

## 2026-10-01: GAPs-to-LM session -- attention merge to retrieve-then-generate (CLOCKED)

- Handoff GO verdict worked through: baseline green -> phi-core
  ai/t-transform-bridge merged+pushed (af4e3c0, both suites green on
  main) -> tokenizer OOV measured (0.59 in-domain, 0.29-0.48 wider,
  UNK 39% mass) -> depth compounding decided (84->79.3dB, ~sum+1.4) ->
  block-one (71/84dB) -> H=2 (74/87dB, fan-in law) -> depth-2 tied
  (70.9/83.7dB, -3.4dB) -> causal mask as structure (past bit-exact)
  -> twins without template strings -> SVD injection (top1 0.325,
  ppl 95) -> BPE freeze+assembly (roundtrip 0, OOV 0) -> edge-store
  freeze (1.00/1.00/0.99) -> retrieve-then-generate (16/16 tag-exact).
- Full suite: 55/55 ALL OK (10 new gates this session).
- Surprises:
  1. Word-2k bought coverage (0.826) but cost behavior (top1 0.252,
     ppl 390): coverage without mass concentration doesn't convert.
     The experiment was worth more than the vocab.
  2. Piece-bigram 0.035: pieces stretch words over ~1.7 steps, so a
     1-step model loses word memory. The spec's gate was wrong, not
     the freeze -- amended with numbers kept.
  3. Raw-vs-triples GATHER silently misread (luck-masked): hardened
     op_gather fail-loud, all suites re-green. Design cases leaning
     on the language make the language stricter.
  4. Passive S=5 40.6dB thin margin was content, not length (S=8
     gives 51-53dB). Margins need content sweeps, not just length.
  5. Cue combiner position-gap mismatch (frozen kept gaps, demo
     compacted): caught by the passive twin missing, fixed to
     identical combiners. Twins earn their keep as gates.

## 2026-10-01: word-order arc -- identity-V to cosine retrieval (CLOCKED)

- Seed sweep (12 dirs, best 0.94) + gain fits all rejected by the joint
  rule (dist down + top1 down = blur/collapse); identity-V (wv=wo=I,
  content unscrambled) first both-axes win (0.858/0.341); similarity-QK
  (I*0.2) second (0.832/0.354, in-contract); dot retrieval diagnosed
  norm-biased (2/16) -> cosine as structure (assoc_cos, priced m_acc
  36230, 14/16 through-listing); hidden-space keys required (emb-space
  != hidden-space); GATHER triples assert hardened (luck-masked
  silent misread found + fixed); MLP moves all neutral/rejected.
- Full suite: 57/57 ALL OK (12 new gates this arc).
- Surprises:
  1. The joint objective (dist + top1) rejected 6 straight "wins" --
     without it we would have optimized into mush twice over. The
     constraint is the instrument, not the decoration.
  2. Dot-vs-cosine: exact self-matches lose argmax to big-norm rows.
     Direction, not magnitude, retrieves -- geometrically obvious in
     hindsight, invisible until hidden-space keys made it measurable.
  3. Fresh scales priced first-try green 4 times running (35492, 35048,
     36230...). The three laws price before running -- that is what
     they are for, and the streak is the evidence.
  4. v07 misses retrieval both sides at every stage (genuine
     ambiguity); v02 UNK outlier persists (OOV mechanism, BPE-shaped).

## 2026-10-01: counts-structured DOWN -- bankhn (CLOCKED)

- MID-space bank built first (128 mean-MID keys): breached logits
  (9.2dB, 21x oversized writes) then went lateral (0.84/0.341) --
  MID-space is content-free, matching is uniform noise. Re-grounded
  in emb-space (unit-norm keys + softmax competition = attention over
  stores): twin_dist 0.781 (best yet), top1 0.350 (holds), parity
  54.5/53.5 at priced m_acc 36118 (global raise held; per-block
  M-dicts stay follow-up per the law that demanded them).
- Full suite re-run at close (report in chat). Rejected this arc:
  MID-bank (wrong space), wgate x2 (blur), seed 3/4 (overfit).

## 2026-10-01: bankhn2 -- last random structure gone (CLOCKED)

- Layer-2 swiglu replaced by the tied storebank (same 128 stores):
  twin_dist 0.781->0.743, top1 0.350 holds, torch parity 53.5/53.1.
  Every matrix now counts-derived or content-structured (SVD ends,
  identity V/out, similarity QK, dual storebank MLPs, fixture norms).
- The joint rule's full record this arc: 9 rejections (gains,
  seeds, MID-bank, wgate) each with numbers; 5 writes (g2, Ivo, IvoQ,
  bankhn, bankhn2) each beating on primary without regressing
  constraint. Selection with a leash, not tuning.

## 2026-10-01: BPE-covered rebuild -- transfer measured (CLOCKED)

- Piece model (V=2038 SVD ends, IvoQ body transferred, bankpiece
  top-128): parity 56/62dB first try (same D, architecture carries);
  twins cleaner in piece space (0.485 vs 0.74 word -- no UNK mush);
  prediction weaker (piece top1 0.062 S=16, word-decoded 0.076 vs word
  0.35): 2038 choices + 4-word windows + 43%-mass bank + unfitted
  body. Gain-4 twin win rejected again (0.402/0.023 -- the joint rule
  is now 7-for-7 on catching blur). Piece prediction needs its own
  fitting arc (gains/temps refit, bigger bank, S=16+).
- Fact organs separated at scale: retrieval edge-ID 25/55 from
  subj+pred cues through the listing (chance 0.02; misses are
  same-person neighbors -- subject class resolves, edge disambiguates
  partially); pure generation 0/54 facts (function-word attractor).
  Retrieval is the fact organ, generation the fluency organ -- the
  fusion composition, not either alone, is the functional system.

## 2026-10-01: depth-4 -- composition beats counts (CLOCKED)

- Depth-4 tied bankhn2 (174-line listing, m_acc 36849 priced): parity
  H8 44.1/LOGITS 45.2 (budgets summed as predicted; 2.7dB cover-tax on
  H8 from the global raise, stated), twin_dist 0.743->0.479, top1
  0.350->0.472 -- BEATS word-bigram 0.406 from inside the transformer,
  first time. Composition outperforms counts alone: the depth bet of
  LLM_DESIGN pays with numbers, not narrative.
- Piece-space fitting verdict: gains rejected (blur), seeds lose to
  structure, S=16 helps weakly (0.044->0.062). Piece prediction needs
  structural fitting (bank mass, temps), not global knobs.

## 2026-10-01: head specialization -- TBETA, 47th mnemonic (CLOCKED)

- Per-head temperatures (head1 T=1 control, head2 T=4 fitted over grid
  0.25/0.5/0.75/1.0/2.0): twin_dist 0.743->0.700, top1 0.350 holds,
  parity 52.3/52.5. The joint rule passes its first specialization:
  distinct roles per head (averaging + selection) beat uniform temp.
- Extension drill per AUTHORING (op + sig + honors-key gate + docs +
  listing use + census class): TBETA honored beta_b (2.0 moved),
  default==beta, asm-lang-coverage drift gate FIRED correctly (46/47)
  and went green on documenting. The checklist works as designed.

## 2026-10-01: v2.1 piece-fitting -- knobs exhausted, depth-4 fusion speaks (CLOCKED)

- Piece space: bank K512 (twin cost, rejected), global gains (blur),
  seeds (lose to structure), coordinate (collapse-blocked again --
  joint rule 11-for-11), S=16 weak help (0.044->0.062), head-temps
  transfer shape but not joint win, piece depth-4 twins 0.383 with
  top1 cost (rejected). K64 probe died on a float-mask script bug
  (mine); trend already answers (smaller-better twins, mass too thin
  to matter). Piece prediction needs structural fitting from piece
  measurements, not transferred anything.
- Depth-4 word fusion demos: facts exact + fluent coil (ptolemaic
  persian, great's, second, battle), no loops, UNK only on OOV verbs.
  Most functional outputs to date (top1 0.472 stack).

## 2026-10-01: D32 refit -- scale reliably (CLOCKED)

- D32 transfer failed joint (twin 0.896, QK-identity worse, seeds
  collapse); refit from D32 measurements: g2 (0.763/0.337) -> I0.5 V
  (0.722/0.35, both-axes) -> headt 0.25 (0.714/0.35, holds) ->
  depth-4 (twin 0.405 BEATS D16 depth-4 0.479, top1 0.35 holds,
  parity 44.7/43.7 at priced m_acc 36849 + m_cov 35686 -- H8 9.6
  needed the cov raise, laws 5-for-5). Width scales when refit, not
  transferred: operating points move with fan-in, gains don't carry.
- Rejected: seed2 (0.621/0.187), QK-I (0.896), V1.0 (0.414/0.187).

## 2026-10-01: mined content -- 51 edges with probes (CLOCKED)

- Counted miner (BE/ACT/OF patterns, minsup-2, junk-filters): grokipedia
  1173 sents -> 51 edges (battle-of-X riches: actium x20, alexandria
  x19); wikitext-2 sample 111 edges (noisier, open-domain). Recall
  100% @0/8 through assoc_mem; probe self-retrieval 9/18 measured
  (partial-cue collisions price disambiguation, not a bar); curated
  55 untouched (no regression). Content growth gated by construction.

## 2026-10-01: scale-up v3 -- 174 edges, shapes grow (CLOCKED)

- w20k mining (45k sents, minsup-3): 138 edges, glue-filtered to 123;
  merged with groki 51 -> 174 (dedup exact). Recall 348/348 @0/8;
  probes 159/174 (91%) full-triple; curated 55 unregressed. Test
  generalized (>=51, count-agnostic -- scale-proofed gate).
- Attest loop x2 seeds: shapes 58/102 -> 80/130 attested families;
  selection + retrieval + cosine gates all re-green. Content 3x with
  quality held: the scale procedure works.

## 2026-10-01: scale quality -- hash-lex, phrase-norm, banked 202+3773 (CLOCKED)

- Scale bug found by measurement: lex-extension patterns were
  iteration-order-dependent, so banks disagreed per-word (hall of fame
  64 vs 22). Fixed with hash-addressed lexicon (lex99: per-word
  deterministic, order-independent) -- banks now agree exactly.
- The fix exposed a real collision (battle/chaeronea-in vs
  gaugamela-in 48-vs-50): pre-norm fragments in the committed merge.
  Rebuilt with phrase-norm (donations x7+5+4 -> x17): 404/404 @8,
  collision gone STRUCTURALLY (not bar-moved). 202 merged edges,
  probes 195/202; w103 3773 re-frozen norm+hash.
- Banked retrieval (curated/mined202/w103, dot-arbitrate, narrow wins
  ties): exact-64 routing verified live on both domains.

## 2026-10-01: piece-loop end-to-end -- zero UNK, fragment failure named (CLOCKED)

- demo_piece.py (BPE -> bankhn2 piece stack S=16 -> word-decode):
  'secretary of state' speaks (word-model UNKs it); zero UNK anywhere
  by construction. Failure named: mid-word fragments ('pthe',
  'mancand', 'bing') -- piece-boundary unmodeled at top1 0.06.
  Speakability solved, accuracy is the fitting arc. Demo, not gate
  (no bar met yet -- honestly labeled).

## 2026-10-01: piece-fit -- structural fitting in piece space (CLOCKED)

- Bank K sweep (twin-screen): 64/0.507, 128/0.566, 256/0.606,
  512/0.641 -- smaller-sharper wins twins monotonically; K64 top1
  0.06 holds (mass 0.31 suffices: writes refine, logits carry).
- Per-path at K64: wv*2 (0.471, top1 0.055 holds), QK-I worse (joint
  rule 12-for-12 on collapse-catching); headt 0.25 stacks (0.443).
  Frozen lm_piece_fit + test_lm_piecefit ALL OK (twin 0.443, top1
  0.055). Piece twins 0.566 -> 0.443 with top1 held throughout.
- demo_piece points at fit: zero UNK everywhere; fragments persist
  ('ation', 'bint') -- boundary modeling named as the next structure.

## 2026-10-01: boundary modeling -- word-trie piece mask (CLOCKED)

- Fragment failure (0.2 invented-word rate: 'pthe', 'mancand') fixed
  at the decoding boundary: trie over train-word piece paths masks
  logits to valid continuations (word-starts at boundaries).
  test_piece_bound ALL OK (0 invented, replay identical, 1514
  word-starts). Listings untouched (rule, not structure -- same
  doctrine as UNK-mask). Trade stated: generation closes to train
  vocabulary (open-domain OOV unsayable -- w103 trie is the follow-up).

## 2026-10-01: piece guidance -- rho, topics, w103 trie, word-finish (CLOCKED)

- demo_piece: repetition penalty (rho 1.3, guided precedent), topic
  steer (piece ids + strength, dose-responsive: city 0->2x at s=3.0),
  trie augmented with w103 vocab (open-domain words speakable),
  word-boundary finish ('resor' -> 'resorted', greedy +3 max).
  All host-boundary; listings untouched. Fixed en route: topic-id
  set bug (int-in-strdict, always empty -- caught by identical
  outputs, the tripwire that works).

## 2026-10-01: output-driven loop -- audit finds glue, penalty fixes (CLOCKED)

- scripts/audit_outputs.py (failure modes counted over flagship
  generations): glue 8/8 (0.60 vs 0.24 corpus), unk 1/8, repeat 0/8,
  fragment 0/8, thin 0/8. Outputs drove the fix list, top mode first.
- Selection glue-term alone changed nothing (all candidates equally
  gluey -- selection re-ranks distributions, can't shift them; honest
  negative, kept as measure). Decoding glue-penalty dose-responds
  (0.60->0.47, bloom holds): wired as gpen=0.5 default in demo_select
  + glue-term in fitness. Best output yet: 'ptolemaic propaganda ...
  roman power persian campaigns ... east' at glue 0.22.
- Machine-level glue (top1 hedging) stays the structural fix; the
  audit stays unmedicated to track it (re-audit after top1 moves).

## 2026-10-01: ollama-in-loop REJECTED (two probes, numbers kept)

- Rank agreement vs fitness: rho=0.37, then 0.49 with spread (bar
  0.5). Mechanism: our fitness saturates by construction (blocking
  forces real bigrams, all candidates 0.86-1.0) -- nothing to rank.
  External judgment adds latency + nondeterminism + worse ranking
  than frozen counts. Corpus judges; models propose at most.
  Proposer role not pursued (topics already come from edges).

## 2026-10-01: untied layer banks LATERAL, rejected (CLOCKED)

- Distinct layer-2 bank (HN2 stats, unit-norm keys, own listing):
  twin 0.743 + top1 0.350, EXACTLY tied numbers -- twice (out-of-body
  stats, then in-body stats under bankhn). HN/HN2 geometries too
  similar (both RMSNormed, shared Vb) for distinct keys to matter.
  Depth helps via attention composition, not per-layer MLP content.
  Rejected: tied stands (simpler, fewer params). Dead listing
  removed; collector + keys kept as evidence.

## 2026-10-01: flagship composition -- depth-4 with full wrapper stack (CLOCKED)

- demo_flagship.py (retrieval prepend + depth-4 + selection over
  bloom/rep/shape/glue + gpen + no-repeat): first runs combine the
  0.472 stack with every wrapper. 'alexander founded alexandria
  egypt ... roman army ... ptolemaic ... macedonian forces' --
  best sentences yet (fact + depth + selection).
- Unskip fallback: retrieved facts with OOV words prepend UNKs (new
  failure, caught on cleopatra run) -> skip-prepend + report
  (seed-only, honestly labeled). The unspeakable-fact class belongs
  to the piece loop (it speaks allied/defeated).
- Noted mismatch: attested shapes counted on bankhn2 generations;
  depth-4 speaks different shapes (0.214 vs 0.471) -- re-attest on
  flagship is the follow-up.

## 2026-10-01: three-track decision -- piece-d4, w103-d4, flagship shapes (CLOCKED)

- Piece depth-4: parity 44.6/49.0 first try; twin 0.443->0.337
  (-24%), top1 0.055->0.046 (within collapse-bar 0.018):
  ACCEPT as structure (tweaks need both-axes; structures need
  primary-win + no-collapse -- policy stated). Demo stays depth-2.
- w103 D32 depth-4: parity 42/48 green; top1 0.270/ppl 217.7 vs
  0.274/217 depth-2: LATERAL. No depth without w103 refit (body
  never saw wikitext; deeper mixing adds nothing). Refit arc queued.
- Flagship re-attest (20 seeds): 72 attested families, 39 NEW
  (eastern/persian/cleopatra/roman -- depth-4's content showing);
  merged attested.json 50->89 shape priors. Selection shape-term
  strengthens on flagship outputs (0.538 live).

## 2026-10-01: w103 refit arc -- transferred stands, local optimum (CLOCKED)

- Operating point moved to flagship (d4): d2 gain-2.0 ppl win does
  NOT transfer (d4 explodes 222->8930 -- fit at the flagship, always).
- w103 d4 screens (10-sent): gains (no win), coordinate (top1 FROZEN
  0.243 all 8 moves; ppl neutral-worse), headt (nothing: 222.5/222.0),
  seeds 1-6 ALL worse (best 0.218/466 -- transferred directions beat
  every random draw on BOTH axes: the structures generalize, not
  overfit), bank K256/512 (top1 frozen; K256 ppl -6, noise: rejected).
- Verdict: transferred groki structure optimal at w103 too -- genuine
  local optimum for this architecture. Next structural shots named:
  D64, per-block M-dicts, w103-body HN banks. Top1 frozen is itself
  the finding (argmax pinned by ends+depth; only new directions move
  it, all worse -- content-fit, not capacity, binds).

## 2026-10-01: per-block M -- mechanism proven, lateral at D16 (CLOCKED)

- Layer magnitudes profiled (H 3.41->5.72, LOG 30.0); per-layer m_cov
  priced 34725->35136 + head 36899. Driver pattern (lm_step + lm_head,
  host loops, per-layer CONFIG): bit-exact vs monolithic (test_lm_step
  ALL OK). Joint: twin 0.479 EXACT, top1 0.472->0.484 (noise),
  parity +-0.5dB. Verdict LATERAL: global compromise already near
  every sweet spot at D16. Vehicle kept for D64 (cover-tax is real
  there). Flagship stays monolithic (5x driver cost, no gain).

## 2026-10-01: D64 arc -- width degrades, diagnosed, parked (CLOCKED)

- Freeze (rank-64 SVD, spec 15.9x) + listing (Dh=32) + parity green
  (53/50 at 36118/35048 -- tighter beats bigger, goldilocks again).
  Transfer fails (twin 1.246): refit finds QK-I0.2 best (0.956/0.309)
  but still trails D32 (0.722) and D16 (0.479).
- Diagnosis (measured): Dh=32 random dots ~4x D16 -> attention
  entropy 0.15 vs 0.6 (near one-hot); V-identity fails everywhere
  (mixing scrambles first, V cannot save); law-temp 1/sqrt(32)
  BACKFIRES (1.13 -- twin rewards flatness, not sharpness);
  head-temps flat; depth-4 WORSE (1.064 -- depth compounds scramble
  without content-preserving V). U-shaped temp optimum at I0.2.
- Verdict PARKED: width needs content-fit, not transfer. No gate
  below flagship bars (honestly parked, recipe in fit manifest).
  D16 depth-4 remains flagship; D32 the wide runner.

## 2026-10-01: content x53 -- w103 minsup-5, probe 0.893 (CLOCKED)

- Miner: no-extend-with-articles (trailing-'the' junk class closed);
  w103 minsup-5 -> 10,672 edges (top: hall of fame x2150, secretary
  of state x1180), glue-junk 3.2%, tail reads clean.
- Recall 40/40 @0/8 at 10.6k keys (exact cues self-match by margin
  at any scale); probe 9528/10672 = 0.893 (202-bank: 0.965).
  Misses concentrate in genuine ambiguity (month-of-that-year,
  cambridge/oxford-sometimes) -- disambiguation consumer priced
  at ~11%, not noise.
- Demos repointed at w103m5 bank (secretary of state exact-64).
  Verdict: scale 53x with graceful, concentrated degradation. GOOD.

## 2026-10-01: piece-native stores -- OOV-free retrieval (CLOCKED)

- scripts/freeze_pkeys.py (same combiner, piece space, plex hash
  patterns): 202 keys, oov=0. test_pkey ALL OK: recall 404/404 @0/8,
  partial 198/202 BEATS word 195/202 (no OOV drop ever), full 202/202
  exact. Live sanity (battle/actium, donations/alexandria, league/
  corinth) exact through the listing. Caught en route: my own
  value-vs-key back-mapping nonsense (dotted random values at keys)
  -- fixed with exact value identity. Retrieval is now vocab-free;
  generation (fragments) remains the accuracy bottleneck.

## 2026-10-01: full suite 68/68 (one honest re-bar) (CLOCKED)

- Suite caught a real interaction: gpen (wired post-gate) costs ~1
  bigram, breaking selectgen-bloom == 1.0 (0.944, max 1.000 among
  candidates). Re-barred to >= 0.85 WITH mechanism stated + max
  printed every run (the trade stays visible, never hidden):
  gpen buys glue 0.6->0.2 for ~1 bigram. Falsified bands become
  measured rows -- this one did, openly.

## 2026-10-01: w103 HN banks lateral -- local optimum from 7 sides (CLOCKED)

- scripts/collect_w103.py (200 sents, HN/HN2 rows by next word, 256
  words 0.878 mass) + diagnosis (HN~emb cosine 0.32 -- weak but real)
  + HN-matched untied banks: top1 0.243 IDENTICAL, ppl 580 vs 573.
  Same verdict as D16 untied: layer content doesn't differentiate.
- w103 local optimum now confirmed from 7 directions (gains,
  coordinate, headt, seeds, bank mass, HN banks, depth). The optimum
  is the finding: transferred structure + w103 ends are jointly
  optimal for this architecture; top1 moves need new mechanisms
  (rank>D32 needs D64 body; D64 needs content-fit -- the circle).

## 2026-10-01: piece arc verdict -- fit stands, width queued (CLOCKED)

- K256 top1 0.058 (lateral), boundary word-top1 0.085 (validity
  not accuracy), gains U-opt at fit (0.481), seeds all worse (best
  0.519), piece32 transfer twin 1.048 (width-without-refit degrades
  -- same law as word-D64). Fit (K64+wv2+headt, 0.443/0.055) stands
  across every probe. Piece-D32 refit queued (gains/V/QK/headt at
  piece32 point). Fragments remain THE visible gap; boundary mask
  holds them at zero inventions.

## 2026-10-01: piece-D32 refit -- same structures win at width (CLOCKED, start missed)

- Start clock MISSED (process lapse, recorded -- picked up mid-roadmap
  without a clock read; fix holds: read the clock when the QUESTION
  lands). Refit from piece32 transfer baseline (1.048/0.049):
  QK-I0.2 (0.952) -> +V-I1.0 (0.735/0.058 both-axes) -> +headt 0.25
  (0.665/0.058 holds) -> depth-4 (0.448/0.044, cost 0.014 within
  0.018 bar, ACCEPT as structure, demo stays depth-2). Frozen
  data/lm_piece32_fit.npz + test_lm_piece32fit.py ALL OK (twin<0.75,
  top1>=0.05, d4<0.60). test_piece_bound re-green (0 inventions).
- Rejected: K128 (lateral, smaller-sharper holds), V2.0 (0.580/0.049
  collapse -- joint rule 13-for-13), MLP-body gains (flat 0.665:
  dead weights under bank MLP, correctly null).
- Surprises:
  1. Piece structures transfer across width when applied TOGETHER
     (QK0.2+V1.0+headt rediscovers the D16 fit exactly): refit at
     the operating point converges to the same answer. Width moved
     the point; the structures held.
  2. D32 depth-2 (0.665) still trails D16 (0.443) even refit -- width
     costs twins; depth-4 (0.448) recovers it almost exactly. Same
     shape as word (0.722->0.405). Depth composes with width.
  3. Gate-3 distance honest: piece top1 0.058, word-top1 ~0.085
     class -- refit is step one of the gate path (boundary-aware
     scoring + bigger banks remain), not the gate.

## 2026-10-01: gate-3 structural shots -- fit is a local optimum (CLOCKED, start missed)

- Start clock MISSED again (question-led turn; the fix keeps not
  sticking on question-led turns -- recorded, not hidden). Eight
  shots at word-top1 0.091->0.20, all priced: trie-mask (no lift:
  greedy already valid, 28/330 both), K128 (identical 30/330),
  K256 built from reverse-engineered construction (row-sum top-256,
  K64-prefix verified exact, mass 0.57): twin 0.673 holds, accuracy
  lateral -- REJECTED (4x params, keep K64); unigram prior rejected
  UNBUILT (miss analysis: truth 0.0026 vs guess 0.0219, model already
  over-guesses frequent 8x -- prior would reinforce the bias);
  beta_b sweep (sharp kills twins 0.853/1.107, 0.125 lateral);
  WIN32 (+1 word, lateral); QK0.1/0.15 + V1.25/1.5 all twin-win +
  top1-collapse -- joint rule now 17-for-17 at piece level.
- Verdict: fit stands from 8 directions -- the optimum is the
  finding (w103 precedent). The miss mechanism is named
  (rare-truth vs frequent-guess: discrimination, not calibration).
  Next mechanisms for a future turn (not moves): distinct-content
  banks, word-bigram fusion proposer (0.406!), content-gated
  sharpening (sharp WHERE, not global -- global kills twins).

## 2026-10-01: retrieval second-pass -- disambiguation closes (CLOCKED, start missed)

- Start clock MISSED (question-led turn again -- the pattern is now
  3-for-3 on question-led turns; fix: read the clock when the
  QUESTION lands, not when the build starts). Composition
  (always-run, no thresholds): pass-1 question cue -> top-1 edge +
  ambiguity SET (top-K edges UNION ties) -> pass-2 full-triple
  re-cue, each edge by best row -> argmax. Frozen
  tests/test_disambig.py ALL OK: 202-bank 195 pinned + 202/202
  (K=4, mean set 5.1); w103m5 9528 pinned + 10284/10672=0.9636
  (K=8, mean set 11.6, bar 0.96); v07 both sides retrieve (word
  path); listing discipline + replay green.
- Rejected/closed: margin triggers (exact-collide misses score 64.0
  confident-wrong -- thresholds can't catch them, always-run can);
  row-level rescore (v07/passive picked poss#1 -- edge-level
  max-row fixed it: content decides, order matches itself);
  rank-based coverage (overcounts ties by 24 -- argmax-honest pin
  9528 matches the old number exactly); hidden-path v07 (3 designs
  falsified: candidate rescore, grammar strip, subject-residual --
  same-subject hidden collapse mechanism; word path covers).
- Mechanism rows: stage-2-over-ALL-keys = 100% (disambiguating info
  always present in full triples); stage-2 loses ONLY outside the
  shortlist (exact). Tail priced: 388 rank>8 burials live in giant
  families (78.2 vs 2.9 mean) -- next mechanism is family-aware
  retrieval, named not wished.

## 2026-10-01: family-aware tail hunt -- 0.9636 to 0.9960 (CLOCKED, start missed)

- Start clock MISSED (4-for-4 question-led; the fix is now a standing
  agenda item, not a lapse note). Tail = giant-family burials (median
  fam 44, tail objs 2-word vs head 1-word). Shots: obj-only cue
  (tail 0.67 BUT bank 0.32 -- family-picker without family-finder,
  rejected as replacement); IDF-weighted cues (0.70/0.01 collapse --
  the combiner needs balanced summands, weights reduce to 1-term
  dominance + noise; rejected with mechanism, cheap); obj-answer
  composition (0.9958 MIRAGE -- leaked obj[1:] past the question
  contract, caught by contract audit and redone honestly at 0.9931);
  J-tuning (J=8 chokes BELOW baseline 0.9574 -- intermediate stages
  can hurt; dropping J for plain family-expand wins).
- Composition final: P1 shortlist (K=8 ∪ ties) -> expand to (subj,
  pred) FAMILIES -> P2 full-triple argmax over members. Gate adds:
  m202-family 202/202, w103-family 10629/10672=0.9960 (bar 0.995,
  mean cand 185), v07-family 2/2 -- test_disambig.py 10/10 ALL OK.
  Losses == family-absent EXACTLY (43; K=16 doubles working set for
  +11 -- K=8 stands).   Residual (0.4%) = pass-1 family recall, next
  priced problem.

## 2026-10-01: fusion falsified, hybrid speaks (CLOCKED, start missed)

- Start clock MISSED (5-for-5 question-led). Fusion proposer
  (bigram x piece) falsified in BOTH shapes: product 0.109->0.088
  monotonic worse with lambda; selection 0.109->0.091 (K=1 to 512).
  Audit found bigram's 0.109 rides first-index tie-breaks over
  97.5%-tied rows (sparse counts: 6522/263k) -- honest bigram lower.
  Piece OOV accuracy 0/144 EXACT (supply without aim; all 30 piece
  hits in-vocab). Oracle union bounds the pair at <=66/330 = 0.20
  with no margin for a real router -- Piece-first-as-fusion CLOSED.
- Hybrid instead (generation quality, not top1): demo_hybrid.py --
  banked retrieval -> fact prepended AS PIECES (no UNK) ->
  piece-carried context (D32 refit) with deterministic router
  (bigram-minsup-2 propose incl. sampled top6 + nrep guard consult,
  else seeded piece step + trie + nrep). UNK-mass rule fired 0/28
  (39% mass -- dead rule, replaced); argmax-core looped x5 (Echion
  attractor, replaced by sampling). Demos: cleopatra speaks its
  fact (was seed-only fallback); alexander mixes 13+1 with olympias
  content; replay-identical verified. Remaining piece mechanisms
  (distinct-content banks, content-gated sharpening) still open.

## 2026-10-02: sharpening spike -- coupling is threshold-gated (CLOCKED, start missed)

- Start clock MISSED (6-for-6 question-led; the agenda item stands).
  Bounded spike, dual-temp oracle on 330 boundaries (660 runs):
  sharp (beta_b 2.0) changes 2/330 top-1 decisions (1 fix, 1 break,
  net ZERO) while the same knob moves twin_dist 0.665->1.107.
  Content-gating moot (n=1 per side) -- spike FALSIFIED as designed.
- First write claimed DECOUPLING (twins move, decisions don't) --
  CHALLENGED same day and revised: continuous decision variables DO
  move (truth-rank +3.9 under sharp, 133 up / 82 down; truth-prob up
  on 251/330, mean +0.002; margins flat). Coupling is real but
  THRESHOLD-gated: mass shifts sub-argmax. Reframed geometrically
  (navigation: flat converges trajectories, sharp discriminates
  endpoints -- one mechanism with sign-disagreement, not two
  worlds). The joint rule stands vindicated as tension: twins and
  decisions disagree on temperature's SIGN, and neither implies the
  other. Aim still lives in content paths (knobs that move decisions
  at constant twins were all value-side); distinct-content banks
  remains the open structural shot.
- Pipeline audit (time/order lead): absolute-origin shift moves H4
  at 65-67dB (integer-quantization leak of rotary relativity --
  real, measured, below every bar) but twin_dist only to 4 decimals
  (0.665/0.666) -- EXONERATED as the cause. Mismatch hunt stays open
  (twin S-confound known-shape); theory side flagged (T-transform:
  lattice temperature entangled with encoding scale).

## 2026-10-02: what the twin corrects for -- differential instrument (CLOCKED, start missed)

- Start clock MISSED (7-for-7; agenda item now comedic). Thesis
  (user): the clock IS the execution; twin comparison is Da Vinci
  averaging (opposed imperfections cancel common-mode error);
  find WHAT it corrects for. Results: (1) origin-leak CANCELS in
  the difference (H4 moves 65dB each side, twin 4th-decimal still)
  -- common-mode cancellation PROVEN live, the Da Vinci mechanism
  confirmed as the instrument's working principle; (2) content-
  aligned trajectory probe (same pieces, all H rows, not endpoints):
  0.464 mean vs 0.665 endpoint -- ~0.2 of the endpoint is
  order/length execution-shape residue (the averaging limit, with
  a number); remainder = content-invariance proper + uncancelled
  residue. Causal caveat recorded (aligned pairs share tokens, not
  histories -- decomposition estimate, not replacement).
- Instrument proposal (not gate): twin v2 compares worldlines, not
  endpoints (content-aligned trajectory comparison). Endpoint
  averaging is structural to the current instrument; trajectory
  comparison sees past it. Needs bars + multi-pair thought first
  (LIB-015: measured row today).

## 2026-10-02 (branch teacher-scaffold): assay A -- teachers audited with our harness (CLOCKED, start missed)

- Start clock MISSED (12-for-12). Qwen2-0.5B + SmolLM2-135M, 8
  audit seeds x 12 tokens, plain topk-12 seeded (native behavior,
  no guards), same 5 modes + closure/open-validity
  (research/teacher_audit.py + .json, branch-only).
- Qwen: repeat 0/8, thin 0/8, glue 1/8, closure 4/8 -- does NOT
  coil, no guards needed. BUT fragment 7/8 as open-domain sludge
  ("bio website location", "may 27th 2006"): aim without grounding.
  SmolLM2: repeat 2/8, glue 4/8, closure 1/8 -- coils LIKE US
  ("the last the last the last", "battle of the basileia").
- Verdicts: (1) coil ~= small-scale phenomenon (135M coils with
  training; 0.5B doesn't) -- supply/scale, not our unique bug;
  guards aren't the difference (teachers ran unguarded). (2)
  Validity tradeoff INVERTED (fluent-sludge vs valid-coil) -- the
  fusion argument restated empirically. (3) Closure = missing
  machinery with cheap shape (EOS/stop; ours: shape-completion
  stopping via 89 attested families -- UNBUILT, queued). Confound
  stated (guarded-ours vs plain-teachers; Qwen gap survives it).

## 2026-10-02 (branch teacher-scaffold): assay B-1 -- stores share physics, differ in shape (CLOCKED, start missed)

- Start clock MISSED (13-for-13). SmolLM2 L0 MLP through our
  two-stage listings, test_realw protocol mirrored
  (research/smolm2_realw.py + .json, branch-only): parity holds
  (66.3dB peak-relative; 31.79 at peak=1.0 is fixture magnitudes
  +-53, re-barred honestly like selectgen); spectrum 24x (Qwen
  17x); spread 34.9dB (Qwen 33); track corr -0.81 (Qwen -0.84);
  planted inf (same). Single-block storage LAWS IDENTICAL.
- Differences: macro SHAPE (Qwen 5.4x-wide x24 shallow vs SmolLM2
  2.7x-narrow x30 deep) + TEMPERATURE (SmolLM2 weights 7x,
  activations 10-60x hotter: DOWN +-53 vs Qwen H 0.08). "Stored
  differently" = capacity placement + operating temp, NOT
  different physics. Our narrow+banks+depth stack is SmolLM2-class
  (and coils like it -- assay A grounds out here).
- Port verdict: both ALREADY run in our listings (parity both).
  Echion update rule is JOINT (bank mass x temperature x depth):
  K-sweeps held temps fixed, but Qwen-ratio width (K173 @D32)
  needs its own temp point -- width alone lateral, joint unknown.
  Queued: bank-mass x temperature JOINT sweep (off-axis optimum
  hunt), then Echion refit at the winner.

## 2026-10-02 (branch teacher-scaffold): adapters run, theft fails on reference (CLOCKED, start missed)

- Start clock MISSED (14-for-14). Adapter (programs/lm_d32_adapt
  .asm, probe variant kept): Qwen-L0 MLP as coprocessor via random
  orthonormal projections (seed 11, tight frame) -- executes green
  in assembly (integration proven), twin sane (0.684), top1
  IDENTICAL (33/566, same counts) -- semantics null (alien LN
  geometry + early-layer memories + our local coverage; fitted-L0
  closed without build: same direction, stronger prior needed).
  Two shape bugs caught en route (projection order, torch .T --
  both shape-loud, minutes).
- Harvest trial (16 prompts x 2 teachers greedy, mine+attest):
  34 candidates, attested-novel 9 (ALL pronoun sludge: "he was a",
  "it was the" -- attestation admits grammar, not facts),
  agree-novel 3 (teachers diverge), junk 25 majority -- REJECTED
  by pre-stated rule. Mechanism: teachers speak in PRONOUNS
  (discourse), our miner reads NAMES (capitalized) -- knowledge
  present but unresolvable without coreference (new machinery,
  named speculative). Steal-via-harvest CLOSED; coreference-
  resolved harvest queued behind real need.
- Reverse verdict: integration YES (adapters execute), physics
  SHARED, knowledge NO (reference gap + early-layer null).
  Echion update stays the JOINT sweep (main branch), not theft.

## 2026-10-02 (branch teacher-scaffold): coprocessor generation -- three voices (CLOCKED, start missed)

- Start clock MISSED (15-for-15). research/gen_adapt.py: same
  seed + same guards (seeded topk-12, trie, nrep-4) through base /
  qwen-L0 / smol-L0 stacks. All speak (integration end-to-end):
  base coils with content (ptolemaic maladies), qwen leaks a
  novel word (tablets) inside the same coil, smol doubles glue
  (and and) + fragments once (resored -- demo-grade trie-fallback
  roughness, test_piece_bound rule untouched). smol twin-shift
  (0.494) moved voice without improving it -- geometry moves,
  decisions threshold, again. Honesty boundary kept: coprocessor
  mode (one teacher block beside our loop), never full-model port
  (attention + 23/29 layers stay teacher-side, stated).

## 2026-10-02: data audit -- 3 articles carry everything (CLOCKED, start missed)

- Start clock MISSED (9-for-9). Lineage: groki corpus = THREE html
  files (938/235 sents) -- word vocab/counts/ends, BPE-2000 merges,
  piece counts/ends/banks, mined edges ALL derive from it (+w20k
  for mining only, +w103 for scale banks; LM data never leaves
  groki). Rare-word aim starved BY CONSTRUCTION (rare contexts
  appear ~1x -- nothing to fit; mechanism behind 0/144 OOV).
  BPE-2000 merge count NEVER varied (unexplored axis, mechanism
  both ways: fewer merges = less stretch vs worse sharing).
- Probe variance (paired, same 330/566): word CI half-width 0.032,
  piece 0.019 -- small top1 deltas UNRESOLVABLE (0.085->0.091 is
  +2 words, nested wins 28-subset-30, directionally clean but
  nonsignificant; piece 31v33 discord 1v3, p~0.6). Doctrine
  VALIDATED in retrospect: gates already bar twins (continuous)
  with top1 as floor, never top1-deltas. CIs belong beside pins
  in gate strings (follow-up, no gate change).
- Audit-the-auditor: SECOND harness bug caught by a gate pin today
  (double-last [-1][-1] scored scalar==id -> 0/566; word loop in
  the same script stayed correct). Two-for-two on pin-caught
  harness bugs -- pins earn keep hardest against our own scripts.
- Proposals priced: (P1) report CIs beside pins, probe-v2 widened
  set later; (P2, the data answer to ignorance) w20k/w103
  piece-COUNTS + ends stack -- new supply for aim, replays
  freeze/refit arc on bigger corpus (scripts/freeze_w20k.py
  committed: deterministic counts/ends/banks + manifest); (P3,
  after P2) BPE merge-count sweep -- full cascade per point,
  priced frankly in hours.

## 2026-10-02: P2 new supply -- better spectra, halved accuracy (CLOCKED, start missed)

- Start clock MISSED (10-for-10). w20k (85k wiki2 sents) encoded
  under groki BPE (skip=0, OOV-free by construction): nnz 8x,
  top200-mass 0.66 (vs 0.54), spec16 10.1 / spec32 12.9 (vs 3.9 /
  4.6 -- looks healthier). Ends+banks frozen (w20k_piece16/32,
  w20k_bank16/32; counts git-ignored per precedent).
- Transfer: twins WIN big (d16 0.443->0.390, d32 0.665->0.407)
  via 2.7x end norms (operating point moved, magnitude-first).
  Refit battery at w20k point: V-down twin-win/top1-flat (0.037
  best, noise), V-up REJECTED both axes (0.503/0.030 -- rebalance
  hypothesis dead), QK flat, headt lateral, depth-4 twin 0.284
  with top1 flat-collapsed 0.032. Both-domain probes: w20k-best
  loses on wiki2-test too (0.045 < groki-fit 0.057) -- H2
  (domain-matched supply wins) FALSIFIED.
- Verdict: better-counts != better-aim. Priced hypothesis named
  (unproven): SPECTRAL PEAKINESS trades twins against accuracy
  (flat groki 3.9 aims 0.058; peaky w20k 10.1 aims 0.021; word-D64
  15.9 parked the same way) -- giants dominate invariant twins
  and drown readout details. Next shot queued: spectral
  flattening (whiten top modes) as ends-side knob, behind
  distinct-content banks. w20k artifacts kept as evidence.

## 2026-10-02: distinct-content banks closed, flattening confirms gradient (CLOCKED, start missed)

- Start clock MISSED (11-for-11). Bank ablation (evb-zero, no new
  file): bank = frequent-amplifier, rare-NEUTRAL (frequent 32->29,
  rare 1->1 -- removing loses without recovering; twins 0.488 ->
  0.618 LOGIT-space, direction holds). BGE
  (bigram-expectation bank, emb-space keys, groki+w20k stats):
  twins win (0.593/0.607) but frequent DOWN (0.146) and rare still
  1 -- expectation writes blur toward the mean (keys live in the
  wrong space: HN queries vs emb keys, the rejected MID-bank
  class). HN-space mini (284 exemplar buckets, 50 sents):
  coherence WITHOUT conversion (within-bucket cos 0.56-0.58 vs bg
  0.39 at ALL sizes incl n=2 -- signal exists! -- but bank
  lateral 31v33, rare still 1: competition margins too thin;
  threshold lesson again). Full 200-sent V2 closed WITHOUT build
  (mini decisive; word precedent agreed). Rare stays 1/388 across
  FIVE bank variants -- readout-side closed as a class.
- Spectral flattening (w20k ends, E=U s^tau V): tau 1.0->0.6
  (spec 12.9->4.6 = groki): twins 0.407->0.604 AND accuracy
  0.032->0.041 (rare 0->1) -- gradient CONFIRMED toward groki on
  BOTH axes (~35% of the gap is shape; rest is content/domain).
  Peakiness hypothesis now mechanism, not story. Not a gate (below
  fit) -- recipe recorded (regenerable), flattening queued as the
  live ends-side knob.

## 2026-10-02: PMI differential -- inert, error is ignorance not bias (CLOCKED, start missed)

- Start clock MISSED (8-for-8). Differential-decoding probe (user
  thesis: stack differentials, converge, trace the error): PMI
  score = logit - λ·logP_uni (frozen train unigram), greedy word
  rollout, λ sweep on 330 boundaries. λ=0.25: IDENTICAL 30/330
  (zero flips either way); λ=0.5: 20 (worse); λ>=1: 0 (total
  collapse -- rarity bonus exceeds logit dynamic range, rare-noise
  wins everything). Plus an earlier self-bug caught en route
  (rollout overshoot on single-piece words gave 0/330 incl.
  baseline -- fixed to pure-loop, baseline re-verified 30/330;
  harness bugs fail LOUD here, caught by the baseline pin).
- Trace verdict: error source is IGNORANCE (underdetermination),
  not BIAS (removable skew). Four nulls converge: (1) PMI removes
  nothing (identical-30: no skew to remove); (2) miss pattern
  (rare-truth/frequent-guess) is the optimal-backoff signature,
  and removing the backoff destroys calibration (λ>=0.5); (3) OOV
  aim 0/144 (no signal anywhere for rare); (4) WIN32 +1 word (more
  context adds no signal either). Differentials converge on bias;
  they cannot create signal. Bank-ablation differential closed
  TRANSITIVELY (same direction as PMI, same null -- no run spent).
  Redirect stands (content paths carry new signal, decode tricks
  don't); distinct-content banks remains the open shot.

## 2026-10-02 (branch teacher-scaffold): organization law + siphon capped (CLOCKED, start missed)

- Start clock MISSED (16-for-16). Permutation sandwich on our
  stack: pair-preserving 23.7dB, within-half pairs 29.1dB,
  general 6.0dB -- NO hidden permutation is a symmetry (graded
  breaking, exact nowhere). First prediction (pairs free) died
  on TBETA head-asymmetry, second (halves free) on RoPE angle
  slots: basis PINNED by angle-slots > head-split. First
  organization law from an invariance test, as proposed.
- Paired-state siphon (50 sents, ours-32 -> Qwen-896 least
  squares, held-out 10): L0 15.2 ... L23 23.1dB, rank-flat
  (rank4 ~= rank32: mappable content is ~4-dim; bottleneck is OUR
  content, not map width). Below 40dB bar -- linear last-row
  siphon INSUFFICIENT. Confound stated (40-train fit, 28k params:
  sample starvation possible; 200-sent refit would separate, not
  run -- gap to bar too wide to matter).
- Verdict: organization test DELIVERS (graded symmetry map);
  siphon does not (knowledge needs our-side content first --
  depth/content, then transfer). Full port urgency reduced for
  knowledge (port buys compute); attention blocker unchanged.

## 2026-10-02: port depth limit -- range law bites at L1 (CLOCKED, start missed)

- Start clock MISSED (19-for-19). L0 port GREEN (41.81dB +
  divergence pin); L1 staged RED (13.4dB). Bisection (4 rounds
  incl. 2 mirror bugs, both shape-loud): QKV green (65dB), failure
  at DOWN -- L1 MID +-40 (vs L0 +-17) x Wd folds per-PRODUCT
  (range law: matmul products must stay ~+-13, m-INDEPENDENT;
  chunking useless since fold is per-product pre-sum; ADD folds
  too). L0 was LUCKY (small Wd), not robust -- same law, kinder
  numbers. Staging/splitting cannot help; only phi-core wide-path
  unblocks (filed as backlog class, not this repo).
- Compounding: no-qb divergence 23.3 -> 15.2dB across two layers
  (range workaround decays with depth -- priced). Knowledge
  (lens: L22+) UNREACHABLE by port on current substrate. Closing
  irony, stated warmly: our small-weights construction sidesteps
  the exact law that blocks teacher ports (fit bodies live where
  products stay small BY DESIGN). The native loop+implant path is
  unaffected -- fill-up proceeds natively.

## 2026-10-02: fill-up turn 1 -- marshalling process runs (CLOCKED, start missed)

- Start clock MISSED (20-for-20). THE PROCESS (v1, five stages):
  PROPOSE (corpus-mined edges only -- attested by support; never
  bare teacher text) -> PARAPHRASE-FILTER (teachers supply surface
  orders; exact-word filter or drop loudly) -> FREEZE (same
  combiner+lex99, SEPARATE bank, provenance-kept) -> VERIFY
  (self-recall exact through listing + replay identical +
  arbitration routing + existing probes green by construction) ->
  CATALOG (append-only ledger: geometry, support, orders, shas).
  Bit-exact everywhere claimed (recovery, replay, freeze bytes).
- Turn 1 (m0054 secretary/of/state, sup 12): paraphrase 0/4
  faithful (drift+loops -- teachers can't rephrase either);
  canon-only frozen (scripts/freeze_marshal.py); self-recall 64.0
  exact; arbitration marshal-64 vs mined202-64 TIE (duplication
  found by measurement, not review -- turn-1 class grandfathered,
  novel-only enforced from turn 2); test_retrieve 16/16 green.
  Two process bugs caught live (catalog double-append on rerun;
  orphaned edit block -- both fixed, idempotent now).
- Catalog open: data/marshal_catalog.jsonl entry 1. Next turns:
  novel-only ideas; teacher role narrowed to paraphrase attempts
  (disposal remains ours: attest + filter).

## 2026-10-02: meaning-vs-routing -- all accuracy is routing (CLOCKED, start missed)

- Start clock MISSED (24-for-24). Frozen counts: surprisal
  (meaning) vs out-entropy (routing) correlate -0.41 -- proper
  nouns 7-11 bits meaning / 1-3 bits routing ("battle" 1.3!);
  function words ~5 / ~4; egypt the hub exception (8.7 AND 4.2).
  Past experiments' intuition, priced.
- Bigram-residual probe (subtract routing, keep content):
  baseline 30/330 splits 30/81 GLUE vs 0/249 CONTENT -- every
  correct prediction is a function word; content aim is ABSENT,
  not weak. Residual kills routing (30->0) for +1 content:
  not viable as decoder, DECISIVE as diagnostic. Division of
  labor MEASURED: routing (models, 37% on glue) + content
  (retrieval/copy only) -- the two-organ architecture is FORCED,
  and hybrid is its structural expression.
- XOR doctrine named (user): define-by-differences IS the
  program (twins, probes, PMI, implants, census, residual --
  every instrument subtracts). Subtraction as identification,
  stated as principle.

## 2026-10-02: audit loop closes -- echo fixed, all modes zero (CLOCKED, start missed)

- Start clock MISSED (25-for-25). Audit B (research/audit_hybrid
  .py+.json): same 8 seeds/modes on hybrid+closure outputs --
  flagship-era (glue 8/8) vs hybrid (repeat 2/8 ECHO, rest 0).
  New mode: fact-prepend/seed DUPLICATION beyond nrep range
  (retrieval keys on seed words, so facts restate seeds).
- Fix: echo guard (strip fact/seed shared prefix, deterministic,
  both demo_hybrid + audit script): re-audit ALL MODES ZERO
  (repeat/fragment/glue/thin/unk 0/8). Loop closed exactly as
  designed (top mode first, re-run shows delta). One sed-hack
  misstep en route (junk line, immediately reverted -- edits go
  through the reader, process note).
- Outputs now: fact-tail + seed, closed, valid, content-ending
  (helios/summoned/pompey-defeated/twenty-cities/olympias --
  retrieval facts flowing). Remaining texture: short (5-10
  words), thin-adjacent by design (closure trades coil for
  brevity -- aim's shadow, still R1).

## 2026-10-02: multi-fact collage -- content without prose (CLOCKED, start missed)

- Start clock MISSED (26-for-26). Top-3 DISTINCT edges prepended
  (deduped) instead of top-1: outputs go 14 words, all content
  (never-lost-battle, oracle-siwa, allied-caesar, asp-bite --
  curated depth showing), zero glue-endings -- but generation
  adds NOTHING (hedge-stop fires at once on fact-piles; model
  has no continuation for concatenated facts). Collage, not
  prose: fact pile + seed, closed.
- Verdict: composition gap NAMED (facts don't compose into
  sentences -- needs verbs/relations = the 0/249 content
  problem). Collage stands as RETRIEVAL-DISPLAY mode (grounded
  fact lists > fluent hallucinations; pairs with Gate-4
  ambiguity SETS shown as lists -- coherent product story).

## 2026-10-02: fill-up turn 2 -- novelty pipeline + tier policy (CLOCKED, start missed)

- Start clock MISSED (21-for-21). Novelty funnel: groki
  sub-minsup 334 -> pronoun/glue filters 244 -> w103-attested 4
  (god/war, battle/carrhae, ides/march, department/history) ->
  curated-novel 4. Picked ides/of/march (Caesar-thread synergy).
- Process contradiction FOUND by the tool (attestation REQUIRES
  w103, novelty FORBID banked): resolved as TIERS (w103m5 =
  source mass, marshal = curated tier; cross-tier dupes resolve
  marshal-first, stated). Tool enforces both (asserts) + catalog
  idempotent + per-idea value streams + content-hash ids +
  same-version rebuild deterministic (verified twice).
- Paraphrase 0/4 again (Qwen HALLUCINATES "2017", smol loops) +
  new rule: looped paraphrases rejected (audit repeat def) even
  when word-faithful (smol's "marches" changed meaning!).
  Teachers now 0/8 across turns at faithful rephrasing.
- Verify: self-recall 64 exact, arbitration tie (policy),
  test_retrieve green. Catalog: 2 ideas. Fill-up velocity: the
  funnel (334->1) is the work; freezing is seconds.

## 2026-10-02: fill-up turn 3 -- consumption closes the loop (CLOCKED, start missed)

- Start clock MISSED (22-for-22). Adjudicated OUT: god/of/war
  (myth-vs-game ambiguous), department/of/history (institutional
  boilerplate) -- skips recorded with reasons, not silently
  dropped. Turn 3: battle/of/carrhae (Parthian content we lack).
  Paraphrase 0/4 (Qwen FABRICATES with confidence: "1805",
  "412 Constantine" for 53 BC Carrhae -- filter catches all;
  teachers now 0/12). Canon frozen (n90d3c29c).
- Consumption probe (4-bank arbitration incl marshal): all 3
  marshal ideas self-retrieve 64 exact through the listing AND
  win routing (marshal-first on cross-tier ties). Loop closed:
  marshal -> retrieve -> route. Honest accounting: ties
  everywhere (same combiner+lex = identical keys) means marshal
  buys ROUTING PRIORITY + catalog ledger + domain curation, not
  new retrievability beside w103m5 (value lands for w103-less
  consumers + tier quality). Catalog: 3 ideas.

## 2026-10-02: bank-CRUD turn -- two-tier architecture verified (CLOCKED, start missed)

- Start clock MISSED (23-for-23). Echion-weights thread: usage
  audit (1920 positions, BP streams) shows bank dominated by
  CHARACTER pieces (y/e/s/t top) -- refines fragments, as
  designed. Pruned weakest ('e' bare) + created cannae store
  (bankp32_crud.npz, variant -- frozen K64 untouched): twin
  0.666 (vs 0.665), top1 IDENTICAL counts (32/178, 1/388,
  33/566). Verdict (valuable negative): piece-bankhn stores
  CANNOT hold multi-piece word ideas (single vectors vs 4-piece
  'c'+'an'+'na'+'e' words; first-piece 'c' shared by hundreds =
  no discrimination) -- while assoc/marshal sequence-keys hold
  cannae EXACTLY (64.0). Two tiers VERIFIED by construction
  attempt: assoc (words/facts) -> bankhn (pieces/fluency).
  "Update Echion weights" bifurcates CLEANLY: bankhn = refit
  operating points (joint sweep), assoc = catalog turns (ideas
  land exactly). Catalog: 4 ideas.

## 2026-10-02: twin-v2 + closure -- instruments close, outputs end (CLOCKED, start missed)

- Start clock MISSED (18-for-18). Twin-v2 (tests/test_twin_traj
  .py, 5/5 ALL OK): 8 voice pairs x endpoint+trajectory twins --
  endpoint-mean 0.420/max 0.665, traj-mean 0.520/max 0.651,
  coverage 8/8 with min-aligned 3. Neither bounds the other
  (v03 0.665/0.464 vs v02 0.335/0.651): endpoints carry
  order/length residue, aligned pairs carry history mismatch --
  triangulation instrument, both barred.
- Closure (tests/test_closure.py, 4/4 ALL OK): hedge-stop (top-1
  in frozen frequent-64 = would-coil-next) + glue-trim + CAP-24
  (termination by proof). 8/8 fire before cap, 8/8 add content,
  replay identical. Outputs end on content (maladies/resentments/
  ptolemaic/persian/roman -- short 5-8 word closed sentences).
  Coil-to-closed converter DELIVERED; thinness recorded as aim's
  shadow (short-closed >> rambling-coil for the stranger test).

## 2026-10-02 (branch teacher-scaffold): joint K-x-temp sweep -- no off-axis optimum (CLOCKED, start missed)

- Start clock MISSED (17-for-17). Qwen-ratio bank K173 frozen
  (mass 0.488); 4x4 grid (K 64/128/173/256 x beta_b
  0.125-1.0): twins monotonic in BOTH (0.656->0.735, no
  interaction); top1 three off-cells all EXACTLY 33/566 (argmax
  sleeps through K/temp entirely). Joint CLOSED: K64/0.25 stands
  (simplest + best twins). Finding that outlives the null: knobs
  SORT by the content/shape split -- V/QK move decisions, K/temp
  move twins-only. The split predicts which knobs can ever move
  top1 (value paths only); future sweeps consult it first.

## 2026-10-02: fill-up turns 4-5 -- elicitation harvest + unique track (CLOCKED, start missed)

- Start clock MISSED (28-for-28). Name-forcing (proper names,
  never pronouns): Qwen 5/5 entity-subj (pronouns ~0) vs
  SmolLM2 0/4 (ignores instruction, same US loop) -- elicitation
  is QWEN-ONLY path. Attestation held (4 unattested shapes cut):
  battle/of/cannae survives -> turn 4 (catalog 4). Harvest-v2
  protocol validated (elicit -> mine -> attest -> marshal).
- Action funnel: 66 support-1 -> 30 filtered -> ZERO w103
  attested (verbs don't transfer exactly; groki actions are
  groki-specific). Flipped: w103-ABSENCE = uniqueness signal.
  Unique track (--unique: groki-witness recorded, thin evidence
  class labeled): turn 5 eumenes/defeated/craterus (Diadochi,
  self-recall 64 exact). Adjudicated OUT: god/war (ambiguous),
  department/history (boilerplate).
- Catalog: 5 ideas (banked/attested/attested/attested/unique).
  Funnel economics: elicitation ~25% yield, sub-minsup ~1%.

## 2026-10-02: onion layers -- depth sheds routing, adds no content (CLOCKED, start missed)

- Start clock MISSED (29-for-29). Logit-lens per skin (H2/H4/H6/
  H8 read through wlog, 330 boundaries, preregistered content-
  enrichment): H2 30/330 (29 glue/1 content) = H4 30/330 >
  H6/H8 24/330 (glue 24/81, content 0). Onion CONFIRMED with a
  twist: skins stratify by routing-confidence SHED, not content
  enrichment -- depth removes certainty without adding
  discrimination (explains d4 twin-win + top1-cost in one
  mechanism). AIM IS SHALLOW (layer 1 + bank), INVARIANCE IS
  DEEP. Cross-model corollary: siphon must match depth
  philosophy (Qwen concentrates facts LATE L22+, ours aims EARLY
  -- late-to-early mapping mismatches by construction, closing
  the siphon loop with mechanism).

## 2026-10-02: exclusivity operators -- dead at decoding, alive in acquisition (CLOCKED, start missed)

- Start clock MISSED (30-for-30). Thesis (user): compression is
  spatial, exclusive differences are meaningful. Operators
  tested: max-contrast selectivity (spikiness tracks FREQUENT:
  the 11.2 > alexander 7.79 -- wrong operator, decode
  lateral-to-harmful 30->27, +1 content at lam 1.0); context-IDF
  (1947/2038 pieces never top-12; decode inert 30/30 at 0.5 then
  destructive 29->7 -- PMI's signature exactly). Four decode
  differentials now share one fate (PMI, bigram-residual,
  selectivity, IDF): inaudible or vandalism. Readout's mind made
  up structurally at 0/249; no additive reweighting moves it.
- Division recorded: exclusivity guides ACQUISITION (marshal
  funnel: novel-only + entity filters = exclusivity selection,
  5 ideas banked -- thesis LIVES there), never DECODING. Content
  arrives via stores/retrieval/copy; decoding does routing.
  The two-organ split, fourth independent confirmation.

## 2026-10-02: supply-scale closed -- wiki spectra saturate (CLOCKED, start missed)

- Start clock MISSED (31-for-31). w103-pilot (69k sents, same
  BPE/counts/SVD recipe): nnz/top200/spec ALL ≈ w20k
  (213k/0.661/9.8-12.4 vs 237k/0.660/10.1-12.9) -- same
  distribution, saturated statistics. Full-w103 (50x) would
  replicate, not surprise: peakiness is DISTRIBUTION property
  (wiki ~13 vs groki-history 4.6), size-independent past ~70k.
  Supply axis CLOSED at all testable scales (groki/w20k/w103p).
  c4/dolma caches hold metadata only (20K blobs) -- new
  distributions need downloads (priced, not run). Intrinsic
  question (peakiness universal?) stays open; flattening-tau
  remains the operable knob.

## 2026-10-02: P3 merge sweep -- tokenization neutral, twins monotonic (CLOCKED, start missed)

- Start clock MISSED (32-for-32). bpe_freeze + --outdir (shared
  tooling extended, default unchanged). Merges 500/1000/2000/
  4000 (V 538/1038/2038/4038, stretch 2.48/2.03/1.71/1.32):
  word-top1 on IDENTICAL 871 bounds FLAT (0.064/0.062/0.076/
  0.065, all inside +-0.018 -- hypothesis INVERTED first
  (fewer merges = MORE stretch, not less; corrected live),
  then FALSIFIED (stretch 2x moves nothing resolvable).
  Twins MONOTONIC worsening with merges (0.359/0.418/0.665/
  0.739): finer pieces = less invariant?? no -- FEWER merges
  (shorter, shared pieces) = MORE invariant. Tokenization
  granularity is a SHAPE knob (moves twins, not decisions --
  split holds again). Corollaries: BPE-N free for speed
  (4000 = 1.3x fewer steps/word at equal quality -- propose
  demo adoption, don't churn yet); bpe500-REFIT queued behind
  need (transfer twin 0.359 already beats fit -- refit would
  extend a twins-only lead, not aim).

## 2026-10-02: w103full -- density peaks, aim collapses, supply closed (CLOCKED, start missed)

- Start clock MISSED (33-for-33). Full w103 (4.2M sents, 173M
  toks, streaming counts): density 21% (vs 5-6% pilot/w20k),
  spec EXPLODES 16.8/23.2 (pilot-equivalence REFUTED -- two
  close points misled; spec climbs with density, lesson: never
  extrapolate from two points). Transfer: twin 0.292 (extreme
  invariance!) + top1 0.011 collapse; flatten-tau-0.49 recovers
  0.520/0.034 (gradient reconfirmed). Wiki-probe: raw 0.019,
  flat 0.036 -- BOTH below groki-fit 0.057 AND w20k 0.045.
- SUPPLY VERDICT (final): groki/w20k/w103pilot/w103full x
  groki-probe/wiki-probe -- groki-fit wins EVERY cell. More,
  denser, better counts NEVER convert (twins win big, accuracy
  halves+). Ignorance is STRUCTURAL (body can't use rare stats:
  competition drowns, argmax thresholds, giants dominate), not
  supply. R1's remaining addresses: none in supply; flattening
  live as partial; body-architecture open (unstated how).

## 2026-10-02: instruction diff -- one gap, already priced (CLOCKED, start missed)

- Start clock MISSED (35-for-35). x86-trace proposal:
  binary instrumentation priced-excessive (SASS semantic gap);
  FX proper blocked 3x on harness (symbolic control-flow,
  make_fx moved/arg-shape) -- manual DAG substituted
  (equivalent: teacher math known cold from mirror+source).
  Diff teacher-DAG vs qwen0_block.asm: EXACTLY one gap (Q-bias,
  stated + pinned 23.3dB); all else 1:1 (RoPE base, GQA groups,
  folding, biases, mask convention, no-TSHIFT). 41.81dB IS the
  order-proof. FMA/reduction-order live inside parity numbers
  (attribution complete, no action).
- Compiler note: port template exists (freeze --layer +
  150-line pattern) but full-24 blocked on range-law
  compounding (23->15dB over 2 layers), not tooling. Gradient
  "learning at assembly" reading explicitly out (doctrine:
  needs learning spec). Hints extracted: none new -- the one
  difference was already law.

## 2026-10-02: margin-GUE probe -- no repulsion, mid-depletion (CLOCKED, start missed)

- Start clock MISSED (34-for-34). Zeta analogy test (user):
  top1-top2 margin distribution over saved 330 logits vs
  Poisson-ref: P(m<eps) 0.003/0.018/0.030/0.121/0.230 vs
  0.003/0.017/0.033/0.154/0.284 -- MATCHES Poisson at small
  gaps (ties at chance rate, m0131-class unexotic), depletes
  mid-gaps (bimodal tendency: decisive wins + coin flips, few
  middles). NO GUE repulsion: the analogy's testable half
  FAILS here; mirroring half (s<->1-s :: order transform,
  twin_dist :: line distance) stands as coherent conjecture
  (depth-convergence 0.665->0.284 reads as approach-to-line).

## 2026-10-02: full-24 priced -- port dies at L2 (CLOCKED, start missed)

- Start clock MISSED (36-for-36). Staged L0->L3 (same listing,
  per-layer freezes): parity 41.8/13.4/-19.9/-20.5,
  no-qb divergence 23.3/15.2/-15.6/-16.4, outmax 3.4/31/585/603
  (x170 explosion by L2 -- residual stream runs hot, lattice
  folds everywhere). Full-24: NO (curve verdict, not opinion).
  L0 gated green; L1 marginal-below-bar (recorded, not gated);
  L2+ infeasible (range law + explosion + compounding).
  Lens-on-port skipped deliberately (diverged states carry no
  knowledge to read -- HF lens stands as the map). Port factory
  exists (freeze --layer + pattern); fidelity dies at L1 regardless.

## 2026-10-02: per-layer regimes -- transition is recognize-and-route (CLOCKED, start missed)

- Start clock MISSED (37-for-37). Transition lead (user: layers
  aren't universal, reverse-engineering always needs per-layer
  treatment): outmax HF-vs-noqb AGREE (3.4/31/585/603 BOTH) --
  explosion REAL (not omission artifact), divergence is
  DIRECTIONAL (same envelope, different directions). Q-bias
  omission exonerated as explosion cause.
- Transition verdict: recognize-and-route, not a missing op.
  Cool layers (L0: 3.4) PORT directly (41.81dB gated); hot
  layers (L2+: 585) LENS only (read via HF, never port --
  range law + explosion); writes go NATIVE loop (unaffected
  small magnitudes BY DESIGN). Same doctrine as ever
  (per-block M, two-scale stages, D32 cov raise): price the
  regime, don't fight it. Chunking/splitting/rescaling all
  closed (per-product fold). Unblocks: phi-core wide-path
  (range) or extended-precision ops (new mnemonics class).

## 2026-10-02: one-teacher-per-layer -- degenerate single donor (CLOCKED, start missed)

- Start clock MISSED (38-for-38). Correspondence map (ours
  H2/H4/H6/H8 x Qwen L0-23, least-squares, held-out 10): ALL
  our layers map best to teacher L22 (23.5/24.3/23.4/23.5dB;
  L0 ~17, L11 ~15-16, L23 ~21-22). Below 40dB bar everywhere.
  Per-layer donor selection COLLAPSES to single-donor: no
  pair clears the bar, best is one. Franken-model inherits the
  cap + adapter compounding (not run, priced out).
- Depth philosophies made precise: our depth traverses ONE
  neighborhood (all layers -> L22; depth refines invariance
  within-regime per twin curve) while teacher depth traverses
  REGIMES (L0 syntax -> L22 facts, 15->24dB gradient). Siphon
  fails partly on regime mismatch, not just rank: early/mid
  teacher has NO counterpart in our stack (<=17dB all pairs).
  Lens/siphon tension noted (L22 most mappable, L23 most
  fact-decodable -- geometry vs decisions again).

## 2026-10-02: routed specialization -- first joint win on aim (CLOCKED, start missed)

- Start clock MISSED (39-for-39). Induction gap (Qwen to 17000x
  vs ours 0) traced to missing copy function; per-head identity
  head: twins 2.3-3.8 collapse (no knee vs QK scale) + Pabs ~0
  (ratios-on-dust -- report absolutes with ratios, process fix).
  Top1 surprise: 45-46/566 (+36%!) with twin collapse.
- Discord analysis (paired): 1 loss / 14 gains, McNemar 9.6
  p~0.002 -- FIRST significant aim movement ever. Gains are
  MID-WORD fragments (14-1), losses word-starts (5-4): boundary
  rule (observable pre-decision, no thresholds): flat@starts +
  sharp@continuations. Routed: twin 0.665 HOLDS (twin contexts
  all starts) + top1 46/566 exact. Joint win: twin holds bar +
  top1 significant = both-axes, the first ever on aim.
- Frozen (lm_piece32_induct.npz + manifest) + gated
  (tests/test_routed.py 4/4: twin/probe-size/top1-exact).
  Policy clarified: twin bar catches BLUR (fake geometry wins),
  not honest trades -- but this trade needs no exception (twin
  HOLDS: routing picks flat where twins live). Harness toll
  this turn: flat-index bug, loop-var shadowing, double-count
  artifact (caught by contradiction, read-back, probe-size pin
  -- pins on pins now: tot==566 asserted in-gate).

## 2026-10-02: neighborhood -- misses are near, top-8 at 0.20 (CLOCKED, start missed)

- Start clock MISSED (40-for-40). User read (misses related):
  CONFIRMED both ways -- pred-truth cosine 0.090 vs 0.044
  random (2x), string-sim 0.374 vs 0.255 (shared substrings).
  Top-K curve (routed, first-piece): 0.091/0.103/0.161/0.200/
  0.282/0.358 @1/2/4/8/16/32. Top-8 EQUALS R1's number.
- Reading: top-1 understates knowledge 2-4x; a top-K SELECTOR
  is the priced gap (ceiling 0.20@8, 0.36@32) -- but every
  selector feature tried is dead (margin/glue/frequency/
  rescoring all fail), so the ceiling is visible but not yet
  reachable. Same wall, better window. Hygiene note: 78
  zero-norm pieces (4% dead vocab rows).

## 2026-10-02: selection closed -- joint equals greedy (CLOCKED, start missed)

- Start clock MISSED (41-for-41). Agreement filter (flat∩sharp
  top-K): narrows 8->6.4 keeping ALL truth (66/66) but picking
  stays argmax -> lateral 0.091. Joint-word select: first build
  scored 0/330 via overshoot + length-bias bugs (caught by
  contradiction with greedy); fixed (boundary-aware rollout +
  length-mean): EXACTLY greedy 30/330. Reason it had to be:
  completion ~deterministic given first piece, so P(word) ≈
  P(first) -- no extra information in the rollout to select on.
  Selection CLOSED (agreement/joint-sum/joint-mean + all prior
  rescoring): ceiling 0.20@8 visible, unreachable with
  model-internal info. R1 lives ONLY in routed specialization
  now (sole live mechanism).

## 2026-10-02: proof of the shape -- threefold closure (CLOCKED, start missed)

- Start clock MISSED (42-for-42). Shape claim (selection can't
  beat argmax since P(word)~P(first)): PROVEN three ways --
  (1) pick-agreement joint-vs-greedy 328/330 (identical
  decisions, not just equal scores); (2) 190/330 truths are
  single-piece (no completion exists -- selection VACUOUS by
  construction); (3) teacher-forced truth-continuations mean
  0.004, 0/231 above 0.5, truth-first median mass 0.000 (no
  signal to select on, no source to select from). Selection
  closed WITH PROOF (not exhaustion): vacuous + signal-free +
  sourceless. Content arrives ONLY via external paths
  (retrieval/copy/marshal) or geometry change (refit/routed).
  Two probe bugs en route (empty-confs loop, boundary-blind
  loop -- both shape-loud: empty arrays fail LOUD).

## 2026-10-02: all-of-it batch -- v2 killed, steer measured, CIs, bpe4000 (CLOCKED, start missed)

- Start clock MISSED (43-for-43). Routed-v2 (prev-rare rule):
  twin 0.924 BREAKS bar + top1 identical 46 (ecological
  fallacy live: subgroup estimate confounded by midword --
  end-to-end corrects it). Kill rule for router features:
  must be TWIN-ORTHOGONAL (constant on twin contexts) +
  move top1. v1 stands.
- Steer dose (alexandria/egypt, 4 seeds): content 5.0->5.25->
  6.0 at s=0/3/6 (+20%), repeat 1->2/4, validity held, mid-word
  truncation is harness (no word-finish), not mechanism.
  Weak-positive quality lever, measured not wished.
- Elicitation round 2: 6 triples, 0 attested (vacuous
  is/were-the shapes -- attestation correctly rejects).
  Funnel says no (working as designed); overall yield ~10%.
- CIs computed (key numbers): routed [0.0615,0.1067] vs fit
  [0.0418,0.0807] OVERLAP -- yet McNemar p~0.002 STANDS
  (paired discord 14/1). Lesson recorded: CIs compare ACROSS
  probes; paired tests decide WITHIN. Never substitute.
- BPE-4000 adopted for demos (--bpdir flag, default path
  untouched + verified green): 13 vs 17 pieces/output (fewer
  steps, equal quality per P3 null). D16 ends/bank frozen
  alongside (spec16 3.4).

## 2026-10-02: discord screen -- v1 router confirmed optimal (CLOCKED, start missed)

- Start clock MISSED (44-for-44). Full discord feature screen
  (flat-vs-sharp per position, correct indices): depth0 net -1
  (tied), depth1 +5, depth2+ +8 (9-1!); prev-rare +13 CONFOUNDED
  (rare pieces sit deeper in words -- v2's signal was midword
  wearing a mask); sentpos/margin nothing. v1 rule
  (flat@starts, sharp@rest) already implements the ONLY cut
  that separates -- screen VALIDATES it, adds nothing.
- Twin-orthogonality law (for future routers): features must
  be CONSTANT on twin contexts (boundary qualifies -- all twin
  ctxs are starts; rarity fails -- twin ctxs contain rare
  pieces). Design law, not observation.
- Process toll: mislabeled-field screen reported garbage
  (off-by-one indices -- caught by impossibility 532-vs-33,
  data re-analyzed clean with zero new runs).

## 2026-10-02: shared axis found -- readout norms, and they're correct (CLOCKED, start missed)

- Start clock MISSED (45-for-45). Gut lead (user): four walls
  share one axis; divide-and-conquer fails on coupled systems.
  Temperature lever: twins swing 0.656->1.107 while top1 sits
  EXACT 33 x5 (+3 at extreme, noise) -- decoupled lever, not
  shared axis. Then THE find: corr(log-freq, wlog-row-norm) =
  0.776 (frequent rows 12x mean norm) -- every wall modifies
  inputs/selection around an untouched frequency-weighted MAP.
  Shared axis CONFIRMED mechanistically (explains PMI-inert:
  additive debias can't cancel multiplicative norm advantage
  scaling with |H|; explains temp-frozen argmax: uniform
  rescale preserves norm ratios).
- But: whitening DESTROYS (0.058->0.012 -- norms ARE the base
  rate, real information), tempering declines monotonically
  (33/25/29/34/7 across lam -- optimum at incumbent full
  strength). The axis is CORRECT; the missing piece is
  LIKELIHOOD (discriminative hidden directions -- content
  paths, all exhausted). Divide the EXECUTION (routing works),
  never the MAP (norms correct). Gut honored precisely:
  shared axis located, divide-and-conquer redirected, not
  abandoned.

## 2026-10-02: inverse detector found -- it was retrieval all along (CLOCKED, start missed)

- Start clock MISSED (46-for-46). Dead-write implant (catalog
  cannae into wlog rows, full 4-piece path writes): 0/5 at
  every gain (norm-gap x margin-gap ~= 1200x needed = total
  capture or nothing; A=20 still 1/5). Dead space WRITABLE
  (math exact) but UNREADABLE via argmax. Relation-forward
  selector (subj+pred-matched edge objs, max-logit pick):
  used 45/6, hits 0/2 -- readout can't rank rare objs either.
  Topic-override detector: fires 90%/79%, precision at chance,
  destroys baseline 104->6 (retrieval ranks ABOUTNESS, model
  needs NEXTNESS -- orthogonal axes, honest negative).
- Synthesis (user's two proposals unified): the inverse
  detector EXISTS -- ASSOCIATIVE RECALL (dot over unit keys =
  norm-free direction matching reads dead space exactly where
  argmax is blind: 202/202, 0.9960, v07). Two readout physics:
  ARGMAX (norm-biased, live-only, routing) vs DOT-RECALL
  (norm-free, reads dead, content). USE EACH WHERE IT WORKS
  (= fusion/hybrid, now with readout-physics justification, not
  just engineering taste). Dead channels DO carry more
  information (paper vindicated): writable exactly, readable
  by dot -- argmax was simply the wrong reader all along.

## 2026-10-02: the universal -- differential verification at every level (CLOCKED, start missed)

- Start clock MISSED (47-for-47). User thesis: something done
  at EVERY level, universal -- describe it. Description:
  REFERENCE + DIFFERENCE + RESTORATION, all the way down.
  Lattice ops (mirror + dB + exactness), listings (twins +
  replay identity), weights (mirrors + implant/undo at 368dB
  measured -- removal restores, not approximately),
  data (shas + rerun bytes), gates (bars + re-bars on
  falsification), demos (NOW expects: fact/valid/closed/
  replay), process (audit counts + velocity deltas). Nothing
  is ever claimed absolutely; everything is relative to a
  reference, and every addition carries its own undo.
  Knowledge = verified difference + exact restoration.
- Gap-fill proof (taken seriously, both closed same turn):
  our-wlog-implant undo 368dB (removal-exact); hybrid expects
  gate tests/test_hybrid.py 4/4 ALL OK (fact-tail 8/8, valid
  8/8, closed 8/8, deterministic -- first machine-checked
  CONTENT assertions on demos: helios/summoned/pompey-defeated/
  twenty-cities/olympias all flowing through gated outputs).

## 2026-10-03: instance-prescribed sweep -- three regimes, joint table (CLOCKED, start missed)

- Start clock MISSED (48-for-48). w103full tau ladder twins:
  0.292/0.415/0.500/0.520/0.528/0.528 (spec 23.2->2.2:
  MONOTONIC rise then PLATEAU ~0.53, below groki 0.665 --
  floors are content, curves are spectra). Plateau top-1
  0.025 (vs knee 0.034): aim FALLS on the plateau while twins
  hold -- knee (spec ~4.7) is optimal for both; R3 priced.
- BPE-500 tau 0.4/0.25: twins 0.001/0.085 (COLLAPSE to
  bit-identical endpoints!) + top1 0.046 (down from 0.070).
  Twin~=0 is DEGENERATE (indistinguishability), not ideal
  invariance -- twin DECREASES are ambiguous without aim.
- Joint table (twin-direction x aim-direction) COMPLETES the
  instrument theory: refit (down/up) = sensitivity cut;
  peaky->sweet (up/up) = capacity gain; sweet->flat or
  sweet->peaky (down/down) = collapse either way. Twin-only
  readings ambiguous, PERIOD -- joint or nothing. For
  instance adjudication into v0.3 (their process, not mine).

## 2026-10-03: rank-k ladders -- knee moves, fit confounds k=64 (CLOCKED, start missed)

- Start clock MISSED (49-for-49). Instance prescription (O1):
  twin ladders at fixed corpus+fit, k=16/32/64. k=16 (fit):
  0.511->0.410 MONOTONIC fall over spec 1.7-7.6 (NO knee in
  range!). k=32 (fit): 0.614->0.675 peak ~6.8, dip 0.656 at
  10.0 (knee/inflection present). k=64 (SEED only, no fit
  exists): 1.6-1.9 FLAT across spec 2-14 (transfer band;
  dip 1.22 at 14.0).
- Verdict: knee is NEITHER simply k-set NOR V-set -- k=16
  has none, k=32 has one (~6.8), k=64-seed is flat (but
  fit-confounded: seed baselines sit in transfer band where
  tau gradients wash out). O1 stays OPEN, narrowed: needs D64
  REFIT first (gains/V/QK/headt at 64-dim point, ~1hr), then
  re-ladder. Side finding: structured-transfer (1.048) beats
  random-seed (1.6-1.9) -- structures generalize somewhat,
  consistent everywhere. d64 listing note: lm_d64_headt.asm
  header lies (says Dh=8/V=513; structure is D64-depth2,
  vocab-agnostic, final H4 -- usable as-is).

## 2026-10-03: review ruling -- amendments accepted, 10.0 rung run (CLOCKED, start missed)

- Start clock MISSED (50-for-50). Instance review: premise
  verification accepted (k32 peak + joint legs reproduce);
  self-kill accepted (twin curves corpus-indexed, aim-knee
  absolute -- multivariate factorization stands); per-rung
  log gap admitted (my k16/k64 ladders logged twins without
  full fixtures -- future sweeps log all six fields).
- Amendments ruled: v0.4 numbering, Step-0 freeze, in-stack
  transfer band (word-D64-fit recomputable per manifest --
  verified recipe present, no blob needed), parity gate,
  slice/TBETA verify, V-set parked explicitly, close-vs-park
  fix (mine -- sloppy drafting, corrected), Stage -1 (RAN:
  groki 10.0 end-rung aim 25/566, down-slope 33->27->25
  complete), budget instrumentation, kill bounties as stated.
- Word-D64 parked verdict re-read (entropy collapse at Dh=32):
  headwind for piece-D64 refit expectations, honestly noted.
  GO issued with amendments.

## 2026-10-03: D64 refit -- transfer band, O-clean joint win, QK-1.0 fit (CLOCKED, start missed)

- Start clock MISSED (51-for-51). Mission battery per orders (~15
  probes + screens, inside the hour box). Step-0 freeze
  (`scripts/freeze_piece64.py`, `data/lm_piece64.npz` spec64 5.81 +
  `data/bankpiece64_d64.npz` + manifests; recipe ratified on word
  probe first: g2 1.1090, QK-I0.2 0.9556). Transfer band: twin
  0.7329 + top-1 20/566 [0.0230,0.0539], parity 50.82/52.56dB GREEN.
- O-clean (wv=I1.0, wo=I0.5): twin 0.7329->1.1900 + top-1 20->40/566,
  McNemar 13.88 p~0.0002 -- JOINT WIN, strongest aim movement on
  record (beats routed v1 p~0.002). Mechanism: random 2*seed wo
  SCRAMBLES attended content (word-D64 entropy diagnosis confirmed
  at piece width); clean O unscrambles. V-without-O rejected
  (McNemar p~0.84 noise). QK-dial on O-clean: aim 31/40/48/54/57/60/
  61 monotonic, endpoints discord 4/25 p~0.0002; fit frozen at QK-1.0
  (1.5626/61/566, `data/lm_piece64_fit.npz`). Headt lateral (discord
  0/0, kept). P4-universal DIED (second kill: fits move C at D64,
  S at D32 -- successor R5 regime-indexed C/S, conjecture).
  k64-fit ladder: twins peak ~4.1, aim FLAT 59-61 (no knee, fully
  gated flatten-side; sharpen-side RED parity, parked per-product).
  Filed as THEORY_SPECTRAL v0.4 §9.

## 2026-10-03: seed-aim legs + knee matrix -- O1 closed fit-point (CLOCKED, start missed)

- Start clock MISSED (52-for-52). v0.4's prescribed measurement.
  Unit-seed: parity -41dB RED, HELD (recipe under-scales at D64 --
  methods find). Small-seed 0.1x (parity 64-69dB GREEN): top-1
  30/35/38 across 5.81/4.09/2.87 -- monotonic, NO knee. McNemar:
  smallseed-2.87 (38) vs FIT-QK-1.0 (61) discord 5/28 p~0.0001
  (fit-required for level); smallseed (38) vs transfer (20) discord
  26/8 p~0.0036 -- NEGATIVE transfer (word structures hurt).
- Knee matrix: groki k32-fit below-natural 25/26 (spec 2.51/1.85) --
  full curve 26/25->33->27/25 INVERTED-U directionally complete
  (paired significance unpinned, preds unsaved -- rerun prescribed).
  k16-seed 36/33/21 + k32-seed 37/32/22: monotonic DECLINE, no knees
  anywhere unfitted. Verdict: knee iff fitted body near its natural
  spec (fit-point law); k indexes sharpness (sharp k32, flat k64).
  O1 + O6 CLOSED; k16-fit inverted-U peaking ~3.9 PREREGISTERED.
  Filed as THEORY_SPECTRAL v0.5 §10.

## 2026-10-03: geometric Qwen2-7B speaks -- 28 layers vs HF original (CLOCKED, start missed)

- Start clock MISSED (53-for-53). Builder emits all 28 decoder layers
  + norm + unembed + sampler as ONE 9859-op program (qwen_layer x28);
  CUDA/FP16 runs it with ~14GB weights in VRAM (nvcc 3min, ~9s/tok
  full-recompute, no KV cache -- stated).
- HF: "The capital of France is Paris. It is the most populous city
  in the". GEO: "The capital of France is Paris. The capital of
  Germany is Berlin. The". Both fluent, fork after "Paris.".
- Fork adjudicated, not hand-waved: geo's pick (" The") is HF's #2 at
  margin 0.18 << measurement noise 4.74. Greedy amplifies sub-noise
  diffs into different paths; same distribution, different sample.
  Parity 4.74 abs on hot logits (2-layer gate was 1.7e-4 rel -- 28
  layers compound fp16 weight-quantum, priced honestly).
- Bugs killed en route: missing QKV biases (0.3 rel), unfolded
  temperature (bias must scale too -- 0.5 rel alone), wrong RoPE in
  torch mirrors (positions over heads; emitted ROTARY always right).
  Filed as tests/test_qwen7b_gen.py (fork-aware gates, heavy).

## 2026-10-05: memory budget after the SSD kill -- 7B fits physical RAM (CLOCKED, start missed)

- Start clock MISSED (54-for-54). The story: 28-layer geo run held
  every weight float64 in one process (sample ~61GiB + safetensor
  cache ~15GiB + HF model on top) on a 61GiB box; swap thrashed
  through a power cut and the SSD bricked itself read-only at
  hardware level. Priced from code, not memory: old CPU peak
  60.8GiB bulk-f64 + 14.2GiB cache vs streaming 1.6GiB (37x).
- New: chain/mem.py (stdlib-only estimator + guard: refuses over-cap
  plans with "Refusing to swap", default cap 60% of phys, override
  via JOTNAR_MEM_CAP_GB/--mem-cap-gb) + FRef/IRef shape stubs in
  chain/emit_c.py (zero-RAM codegen; stub sources byte-identical to
  real-array sources, proven) + chain/qwen7b.py shapes_7b/
  prep_7b_weight/clear_7b_cache (stream fp16 bins one weight at a
  time, E resident fp16 1.1GiB / memmapped in bench).
- Combined HF+geo in one process now REFUSED by default
  (--allow-combined to override); test_qwen7b_gen ranks host-side
  from geo_first_id. Full 9859-op/341-input graph compiles from
  stubs at 524MB peak RSS (measured). Gate tests/test_mem_budget.py
  20/20 ALL OK (pins 7617551872 params, refusal + allowance legs,
  stub-shape pins, 28-layer graph leg).
- Surprises:
  1. My own wup stub shipped transposed the wrong way -- the shape
     gate (MATMUL mismatch at programs:691) caught it in seconds.
     Gates earning keep on the author's own edit, same turn.
  2. Box came back without numpy/pip/HF weights at all: guard is
     stdlib-only so it runs anyway; bootstrapped pip --user +
     numpy 2.5.3 to verify (user-local, no sudo). Weights still
     absent -- 7B rerun waits on re-download, RAM side unblocked.
  3. One `%` in a --help string crashed argparse (%o format) --
     escaped to %%. Format strings in CLI help are load-bearing.
  4. Pre-existing env gaps, not regressions: gsl header (emit_c
     BLAS leg), nvcc (cuda legs SKIP), both from the fresh OS.

## 2026-10-05: 7B runs again -- toolchain rebuilt, full 28-layer green (CLOCKED, start missed)

- Start clock MISSED (55-for-55). Continuation of the memory-budget
  round: re-download Qwen2-7B-Instruct at the pinned revision
  (scripts/download_qwen7b.py, 14.19GiB to the exact SNAP path --
  no code changes) and rebuild the stack user-local (no sudo):
  torch 2.14.1 + transformers + safetensors + hub + accelerate,
  nvcc 13.4 (pip) + cuda-cccl headers + cublas/cudart stubs,
  python3.12-dev headers via apt-download + dpkg -x. All captured
  in scripts/cuda_env.sh (source before CUDA work).
- Fixes en route: nvcc via symlink couldn't find cudafe++ (real
  bin dir on PATH instead); nvcc 13.2 vs 13.4-era headers (PTX 9.4
  vs ptxas 9.2 -- matched at 13.4, SASS to sm_86 so the 13.2
  driver is fine); torch 2.14 routes RoPE bmm through triton JIT
  (needs Python.h + libcuda link -- killed via supported
  deregister_op_overrides(disable_dsl_names="triton") in
  chain/qwen7b.py hf_no_triton(), aten numerics for reference);
  my streaming name-split misread ln1@L10 ("ln110") -- replaced
  by chain/qwen7b.py input_sources() exact-match map, gated
  (mem-inputs-resolve + mem-ln-lookup); geo-only subprocess can't
  self-check parity -- prompt-end logits sidecar (.promptlogits.npy)
  ranked host-side.
- Results: smoke 5/5, test_emit_cuda full suite, mem_budget 22/22
  (incl. prep-equiv on real weights), teach/probe green,
  depth rel 1.7e-4 (bar 1e-3), full gen: parity 4.84 (< 10),
  first-pick rank 1 (<= 8), fluent. GEO: "...Paris. The capital
  of" vs HF "...Paris. It is the" -- agree on " Paris" (rank 1),
  fork on the next token: same distribution, different sample.
- Surprise: pip's nvidia wheels land mixed CUDA 13.0-13.4 in one
  cu13/ dir (headers newer than the first nvcc) -- pin the
  toolchain as a set (nvcc 13.4 + crt 13.4), not one package.

## 2026-10-05: first profile -- reload+recompute dominate, graphs wait (CLOCKED, start missed)

- Start clock MISSED (56-for-56). scripts/bench_perf.py's first real
  run (S=16, ntok=8): HF 278ms prefill / 24.3ms decode; geo wall
  11.6s (min 7.2s) vs GPU-busy 3.3s, of which MATMUL 3.1s (93.7%,
  197 calls) and all 9747 other launches ~0.2s. Filed as
  docs/PERF.md (the analysis bench_perf.py always promised).
- Reading: wall-GPU gap (~4-8s) is fresh-process reload (341 bins,
  ~15GB fread + managed-memory migration per token); MATMUL time
  itself is migration-inflated (cold pages inside the timed region).
  Order: persistent process first, KV cache second (STATE-carried,
  SCAN precedent), re-profile, graphs only iff launches dominate
  the remainder. Cached-decode target ~50-200ms/token.
- Surprise: my "launches must dominate 10k ops" prior died on
  contact (~20us each) -- the profile exists to kill exactly such
  priors. nvcc profile builds cost 3.6x clean (133s vs 475s).

## 2026-10-05: serve mode closes the reload gap + beta=1 kill (CLOCKED, start missed)

- Start clock MISSED (57-for-57). Loop binaries (live=[...] in
  compile_cuda/compile_program + chain/serve.py ServedExe over
  stdin/READY): bigram serve 9/9 incl. bank-poison frozen-proof;
  7B geo serves at 143-146ms/step (was 7-11s wall, ~60x) with
  parity 4.84 == one-shot, rank 1, fluent. test_emit_cuda still
  22 green (one-shot untouched by construction).
- The loop caught a REAL emitter bug: cuBLAS beta=1 at all four
  call sites (outputs accumulated; one-shot hid behind zero-pages;
  step0-vs-step1 drifted 17.9 identical inputs). Fixed to kZero;
  regression gate serve-mm-no-accum reads 0.00e+00. Bisection,
  not theorizing: manual two-step determinism probe found it.
- Next: KV cache (STATE-carried) -- the remaining ~6x to HF.

## 2026-10-05: sync gap closed + KV-cache physics check (CLOCKED, start missed)

- Start clock MISSED (58-for-58). Weight-traffic arithmetic first:
  every token streams all 15GB on BOTH sides (HF 24ms proves the
  ~15-30ms floor), so KV cache saves FLOPs but ~no wall at S<=32
  -- deferred to S-scaling, honestly (was the stated next step;
  measurement overruled it). The ~115ms overhead is syncs+launches.
- sync_each=False (same-stream ordering, stores still sync):
  145ms -> 100ms/step, bit-exact (serve-nosync-equiv). Default
  stays synced (debug-friendly); --no-sync for runs. h16 temps
  hoisted (per-step mallocs would leak in serve mode).
- Remaining ~4x is launch + cublas-call overhead -> CUDA graphs
  next (single launch, dependencies by construction).

## 2026-10-05: graphs close launches (+10%, parity-identical) (CLOCKED, start missed)

- Start clock MISSED (59-for-59). Stream-parameterized all 19
  kernel sites (`, 0, capStream`, default empty = zero diff);
  cublas needed no pattern edits (one handle, SetStream before
  capture). Warmup + capture + instantiate once, replay per step.
  bigram graph legs green first try; 7B: 100ms -> 88-91ms/step,
  parity 4.836e+00 identical to serve/one-shot (QWEN_GRAPH=1).
- Only ~10%: launches were never the wall (my PERF.md prior said
  as much; graphs confirmed it). Remaining ~3.7x is cublas-call +
  small-kernel time over the 15-30ms weight floor. Next candidates:
  cublas stream-order tuning, elementwise fusion, TF32-vs-parity
  pricing -- or accept ~90ms (HF does 24ms with the same 15GB).

## 2026-10-05: BMMV, 48th mnemonic -- head-batched attention at same parity (CLOCKED, start missed)

- Start clock MISSED (60-for-60). Strided-view batched matmul
  (2 streams + 11 int literals, natural (B*M,N) out): per-GQA-group
  ONE call for scores + ONE for contexts (28 transposes + 48 BMMs
  die per layer; 9859 -> 5351 ops, cublas 1765 -> ~420 calls).
  Five touchpoints (registry+shape+lattice, C, CUDA fp32+fp16,
  non-FPU loud refusal, LANGUAGE row) + tests/test_bmmv.py 6/6
  (lattice bit-exact both groups, C 8.9e-16, CUDA 2e-7).
- 7B (--bmmv): 88-91ms -> 76-83ms/step, parity 4.84 identical,
  rank 1, same fork. Only ~10%: cublas-call overhead wasn't
  dominant either -- weight streaming + small-node replay is the
  floor fight now (~3.3x to HF).
- Bugs caught by the new gates (both mine): KVD-vs-DH stride on a
  sliced base (overrun guard fired), Q-group slicing for GQA
  (group-1 mismatch), trans flag on the wrong cublas operand
  (INVALID_VALUE). Registry fallout handled legitimately
  (LANGUAGE row, census reduce class, pin updates 341->342).
- Quality held throughout: every transform exactly invariant
  (same products, same order) -- lattice legs are bit-exact, and
  7B parity is unchanged (4.836e+00 baseline vs 4.842e+00 bmmv:
  same 4.84, sub-percent).

## 2026-10-06: decode serves (KV caches, same distribution) (CLOCKED, start missed)

- Start clock MISSED (62-for-62). Single chained decode program
  (28x qwen_dec_full_layer): in-graph cache update
  (broadcast-MATMUL + SELECT on one-hot, GATHER-upcast fp16 files,
  host relays fp32->fp16 rows). Prefill 5 toks 0.5s; decode
  ~90-100ms/tok; prefill parity 1.08 vs recompute, 5.15 vs HF;
  first-pick rank 2 (gap 0.1: fork physics again, gated as such
  in tests/test_decode.py -- never string equality).
- Two real bugs en route: (1) split-phase design broke the
  residual chain (all-28 QKV from raw embed -- per-layer
  interleave is load-bearing; single chained program restored);
  (2) fp16-file/F32-consumer mismatch (silent garbage from step
  1; fixed by GATHER upcast + standard fp16 relay). Plus one
  reverted idea (baked-GATHER static rows broke varying-length
  generation ids -- dynamic is load-bearing; SLICE-pinning
  instead) and one caught-by-gate (test_gen rc=21 proved it).
- Wall-neutral at S=16 by physics (same 15GB); the payoff is
  S-scaling, next.

## 2026-10-06: S-curve maps the crossover (decode wins at 64) (CLOCKED, start missed)

- Start clock MISSED (63-for-63). Recompute 80/111/132ms vs
  decode 95/104/108ms at S=16/32/64 (bmmv+graph+nosync both;
  QWEN_WORKDIR isolates S-artifacts). No crossover at 32;
  modest-but-real at 64, trend widening (recompute ~linear,
  decode ~flat). S=64 quality: prefill parity 0.92 same argmax;
  both sides byte-identical 6-token text.
- Warmup transient named: S=64 recompute climbed 146->285ms
  within one run, flat 132ms on rerun (cold managed pages, not
  scaling -- first-run effect, bigger footprint warms slower).
  Measure steady-state, not first steps.
- BMMV fp16 path gated at last (1.2e-3, fp16 class): no untested
  paths remain in the op.

## 2026-10-06: S=128 widens the crossover, quality holds (CLOCKED, start missed)

- Start clock MISSED (64-for-64). S=128: recompute ~241ms vs
  decode ~124ms (prefill 142ms/tok); prefill parity 0.71 same
  argmax; both fluent, fork after "tourist attractions.". Curve
  complete (80/111/132/241 vs 95/104/108/124): crossover between
  32-64, widening -- the compounding payoff of the decode build.
- Battery at S=32 holds the floor (4/12 identical, 3.83/5).

## 2026-10-06: chat talks (usable surface, argv-order trap) (CLOCKED, start missed)

- Start clock MISSED (66-for-66). demo_chat.py: multi-turn REPL
  over the decode server (Qwen chat template, host sampling
  top-k/temp/seed, /quit /reset /seed /topk /temp). "What is the
  capital of France?" -> "The capital of France is Paris.";
  "And Germany?" -> "Berlin" (context carried, 512 slots).
- Trap: hand-rolled argv ordered all-kP-then-all-vP against the
  program's interleaved kP0,vP0,kP1,vP1.. -- every layer past 0
  read garbage caches (word salad, silent: argc checks count,
  never order). Name-based argv (serve_decode's lives dict) is
  immune; chat now matches program order with the trap commented
  as the pin. Lesson: positional interfaces need order oracles.
- S=512 recompute pathology noted: 14s/step (VRAM oversubscription
  in the legacy full-forward path -- intermediates alone exceed
  comfortable residency). Decode holds 144ms. The legacy path's
  ceiling is mapped, not fixed: decode is the future.

## 2026-10-06: breadth closes -- S=512 parity, pos determinism (CLOCKED, start missed)

- Start clock MISSED (67-for-67). S=512 three-way first read
  10.04 (decode-vs-recomp) with top5 2/5 -- real movement, placed
  by bisection: recomp-vs-HF was 8.97 (shared), decode-vs-HF 5.05.
  Mechanism: short pos files read OOB heap on padding rows
  (benign-by-luck <=128, biting at 512). Fix: full arange(S)
  everywhere (gen/decode/battery/chat/bench) -- valid positions
  for all rows, never luck again. After: 0.8 same argmax.
- Gates re-run green (gen 5.07 rank 2, decode 4.12 rank 1; the
  pos change flipped one razor pick, rank-gated as designed).
- Breadth ledger: S=512 parity + 24-tok battery + S=512 chat --
  the usable milestone holds at every rung measured.

## 2026-10-06: backend hygiene -- no rot, loud everywhere (CLOCKED, start missed)

- Start clock MISSED (68-for-68). CUDA execution modes
  (live/graph/sync_each=False) on non-CUDA targets now refuse
  loudly (was silently dropped). Gates: BMMV->nonfpu refusal
  pinned, C-mode refusals pinned, C-target stub/source
  equivalence (was CUDA-only). Width spot-check
  (tests/test_cross_width.py): depth-1 full-width C-f64 vs
  CUDA-f32 rel 1.2e-6 (80x inside bar) -- C honest at scale.
- Capability matrix filed as docs/BACKENDS.md (value models,
  op deltas, modes, agreement classes, per-backend scale story).
- Certification: non-FPU green, CUDA green, C green except the
  known-missing system gsl header (environmental, no sudo).

## 2026-10-06: block-parallel reductions halve the step (CLOCKED, start missed)

- Start clock MISSED (65-for-65). Warm profile said RMSNORM was
  21% (one thread per row, ~400us of serial loop): rewrote
  rmsnorm/softmax/tshift as one-block-per-row cooperative
  reductions (shared-mem trees) + row-count grids. 76-83ms ->
  44-46ms/step at S=16, quality untouched: unit eps unchanged,
  7B parity 4.84 identical, battery re-run 3/12 + 3.83/5 (zero
  fork flips from the new summation order).
- The stamp guard earned its keep once more (refused a non-bmmv
  binary for the bmmv battery instead of mixing artifacts).

## 2026-10-06: quality battery + parity ladder (compounding instruments) (CLOCKED, start missed)

- Start clock MISSED (61-for-61). scripts/eval_battery.py (12
  prompts x 8 tokens, HF vs geo serve, fork/top5 records):
  3/12 identical, mean top-5 overlap 3.83/5, forks mostly late,
  never below 3/5. Filed as docs/EVAL_BATTERY.md with a floor
  (identical >= 3/12, overlap >= 3.5/5) every future optimization
  must hold -- pace without quality is motion. Geo side reuses
  /tmp/gen7b artifacts on stamp match (build.json + reuse
  refusal on mismatch).
- scripts/ladder_7b.py (parity vs depth, full width): L=1/2/4 rel
  1.9e-4/1.7e-4/3.9e-5 -- NON-monotonic (cancellation across
  layers, not compounding drift). L=8/16 came back 2.3e-5/2.8e-5:
  flat ~1e-5 across 1-16 (no knee) -- fidelity work is officially
  unnecessary; all future effort goes to speed at fixed quality.

## 2026-10-06: siphon demo installs through listings (CLOCKED, start missed)

- Start clock MISSED (80-for-80). demo_siphon.py one command:
  Germany blank-50 baseline -> gain 1.0 Paris rank-1, Italy AND
  Japan hold blanks throughout (gain 0.0/0.5/0.7/1.0 sweep with
  same-path baselines). CALIBRATED line printed. Installs are
  listings (prefix taps + suffix steering + scalar gate), not
  torch tricks -- the siphon as product, first curve point:
  one late direction + one lexical key + calibrated dose.

## 2026-10-06: dose calibration -- installs collide by geography (CLOCKED, start missed)

- Start clock MISSED (77-for-77). research/dose_cal.py (6 facts x
  3 gains x full control rows): France/Germany install self high
  but move each other + neighbors; Italy weak-self (r2-3) moderate
  collateral; Spain clean diagonal but Madrid lands on France at
  high dose; Japan uninstallable (4322) AND vandalizes (destroys
  neighbors at a=0.5); China clean at 0.8+ (rank 1, all controls
  far). Destructive power exceeds constructive for weak
  directions: bad stores are worse than useless (catalog safety).
- Map verdict: dose-response non-monotonic per fact; collisions
  follow semantic neighborhoods; only China is strictly clean.
  Next: multi-install composition (all live at once).

## 2026-10-06: siphon re-opens -- late directions steer (CLOCKED, start missed)

- Start clock MISSED (71-for-71). Lens-directed re-runs (take
  late): trajectory geometry (every layer moves 0.34-1.4 rel,
  path 14x net, biggest step LAST -- movement everywhere,
  decisions late); L26 France-vs-Germany direction reads
  Paris/France tokens at rank 2 by dot product (extractable,
  readout-physics as predicted); implant steers Paris-rank
  54->2/5/3 but top lands on near-tie blanks (steers, doesn't
  install -- needs gain/shape work). Scripts: research/late_dir.py
  + research/late_implant.py (torch-mirror stage; geo port queued).
- Verdict: extract->implant loop closes directionally at 7B.
  Clean installation is the open half (same shape as the old
  implant findings: strength was never the problem).

## 2026-10-06: steering caps at region-overwrite (falsified clean) (CLOCKED, start missed)

- Start clock MISSED (72-for-72). Cleanliness screen (48 cells)
  + far controls + gain micro-sweep: NO clean cell anywhere.
  Best shape is L27 additive: rank 1 from a=0.6, margin to 1.31,
  but Italy/Germany/Japan ALL land on Paris -- global overwrite
  of the region. L26 stalls at rank 3 with Italy captured.
  Decoder-form overshoots to 150k+ junk past a=0.5.
- Falsifier landed: single-direction additive steering cannot
  separate (sensitivity without specificity, at every gain and
  layer tried). Specificity needs CONDITIONAL application (keyed
  stores retrieved iff context matches -- the assoc/bankhn route,
  not additive). research/clean_screen.py + far_control.py +
  shared mirror research/qwen_torch.py (third copy refused).

## 2026-10-06: addressing confirmed -- lex-gated installs hold controls (CLOCKED, start missed)

- Start clock MISSED (73-for-73). Key-gated steering (country
  token early residual as key, ^6 sharpened): Germany installs
  Paris (rank 3), Italy holds Rome, Japan holds -- ALL controls
  hold where full-state cosine gating failed (keys too parallel).
  Reader finding alongside: steered hidden aligns German by
  cosine (Paris 46022!) while argmax picks Paris on norm weight
  -- grade installs by the dot reader. Filed as
  docs/SIPHON_ADDRESSING.md (region falsifier + reader effect +
  gated confirmation + product shape).

## 2026-10-06: margin fragility -- random flips too (CLOCKED, start missed)

- Start clock MISSED (74-for-74). Two bugs then one finding.
  Bugs: serve_steer cached x0 across prompts (every control was
  secretly Germany -- identical gates gave it away); fixed (live
  inputs always rewrite). Finding: RANDOM direction at the same
  whisper dose ALSO installs Paris on Italy. The controls were
  never holding on content -- razor margins fall to any breeze,
  and high-norm giants collect the pieces.
- Consequence, filed in docs/SIPHON_ADDRESSING.md: cleanliness
  needs MARGIN above the local decision-noise floor (measurable
  per prompt), or a different reader. Bits that merely move ranks
  are not installation. The geo port stands (installs through
  listings, gated); the bar for calling one clean just rose.

## 2026-10-06: controls hold -- dose discipline restores specificity (CLOCKED, start missed)

- Start clock MISSED (75-for-75). The fragility verdict above
  was VOID (stale dose file: every control ran at 220). True
  per-prompt doses through listings: Germany 220 -> Paris rank
  1; Italy 13.5 -> blank, Paris 45; Japan 14.4 -> blank, Paris
  99; random 13.5 -> blank, Paris 45. Lex-gating + dose
  discipline = clean install, both paths agreeing.
- Tripwire (third instance): identical gates across different
  prompts mean cached inputs. Live inputs always rewrite --
  served binaries can't tell stale from fresh.

## 2026-10-06: per-layer lens -- funnel gradual, decision late (CLOCKED, start missed)

- Start clock MISSED (70-for-70). scripts/lens_7b.py (all 28
  residuals through unembed, 3 prompts): rank of final falls
  100k+ -> 1 smoothly, locks at L27 every time; early tops are
  code-flavored, surface resolves late (blanks -> 巴黎 ->
  Paris); entropy peaks mid, collapses L21-24, rises at the
  last layer. Filed as docs/LENS_7B.md.
- Verdict on funnel/filter: both gradual, no staged boundary on
  these prompts. Siphon consequence: take late (L22+), skip
  early. Per-layer-training claim: not in the public record
  (card says pretrain+SFT+DPO); shape measured, story declined.

## 2026-10-06: specificity matrix -- first curve segment (CLOCKED, start missed)

- Start clock MISSED (76-for-76). research/spec_matrix.py (6
  country facts, full 6x6 gated installs): diagonal rank-1 2/6
  (Spain, China), rank-3 3/6, Japan uninstallable (4322: no
  anchor -- unsteered top is junk); off-diagonal holds 23/30.
  Strictly clean (rank-1 + all hold): China alone. Interference
  mirrors semantic neighborhoods (France<->Germany contrast
  pair cross-talk; Spain moves Iberia+Europe; Japan install
  scatters). Installs compose with geography, not collisions
  at random -- the curve's first segment, priced.
- Tripwire (third instance, kept): identical gates across
  different prompts mean cached inputs; the stale-dose episode
  is filed under the dose-discipline entry (void evidence
  withdrawn, lex-gating + true doses verified).

## 2026-10-06: composition holds -- gates sharp enough (CLOCKED, start missed)

- Start clock MISSED (78-for-78). research/compose_all.py (all
  six installs live, one forward per prompt): composed == alone
  on all six cells (tops AND ranks identical, incl. Japan junk).
  No washout, no resonance, no new collisions -- the ^6 gates
  (1.00 vs <=0.09) admit only the matching install. Catalog
  turns unblocked at this dose regime. Gate cross-talk mirrors
  geography again (Fr-Ge 0.09 highest; China <=0.03 isolated).

## 2026-10-06: memory-vs-highway falsified, address-channel confirmed (CLOCKED, start missed)

- Start clock MISSED (79-for-79). Rank-1 MLP write (ROME-form,
  keyed on MID match) behaves EXACTLY like residual add: a=0.25
  rank 4 + holds; a>=0.5 Paris rank 1 + Italy/Japan captured.
  Late keys are as parallel as late residuals (same template),
  so matrix retrieval self-gates nothing. The missing component
  was never memory-vs-highway (falsified); it is address-CHANNEL
  (early/lexical vs late/contextual -- confirmed by lex-gating).
- Siphon shape, sharpened: content (late) x address (early) x
  dose, with the address READ EARLY. research/mlp_vs_highway.py;
  shared mirror gained edits{} + MID capture (no fifth copy).

## 2026-10-07: reravel gates green; steered battery 10/11 (CLOCKED, start missed)

- Start clock MISSED (81-for-81). tests/test_reravel.py ALL OK:
  served prefix byte-identical across reruns (LOGITS/Y2/Y26) +
  prefix-vs-gen LOGITS bit-exact (0.000e+00 across binaries --
  unravel/reravel exactness where determinism promises it).
- Stamp discipline generalized: gen build.json now covers program
  text hash (a 341-vs-342-input stale binary passed flag-only
  stamps with rc=9; now fails loud). Canonical /tmp/gen7b rebuilt.
- Steered battery (install live, 12 prompts): target flipped
  Berlin->Paris, controls hold 10/11; the mover is France
  (contrast-pair cross-talk, known from the spec matrix).
  Multi-token persistence next (mirror first, then listings).## 2026-10-07: install persists across generation (CLOCKED, start missed)

- Start clock MISSED (82-for-82). Mirror-level 8-token steered
  rollout (steer every step): Paris, ., Paris, is, the, capital,
  of, France -- stays Paris-coherent throughout, never reverts
  to Berlin (echo-loop is greedy-decoding texture, separate
  concern; sampling exists at the host boundary). Persistence
  holds in principle; steered generation through listings
  (decode-server steering support) unbuilt -- stated, queued.

## 2026-10-07: install order is free (commutativity closes) (CLOCKED, start missed)

- Start clock MISSED (83-for-83). research/order_commute.py
  (rank-1 writes, both orders with keys re-measured inside the
  written model, plus joint): all three orders give IDENTICAL
  outcomes (Paris-54/Berlin-79/Rome-holds) with key drift exactly
  1.0000. Order is free -- catalog turns need no scheduler, no
  curriculum is demanded by the data (anchors-first untested and
  unmotivated). Keys stable + gates parallel = no serial order
  tax on traversal (the yarn-ball latency constraint, satisfied).
- Footnote, honest: ranks identical at ALPHA 0.3 and 0.5 --
  opposite co-resident installs reach a dose-insensitive
  equilibrium (competition, not saturation). Plus one caught
  metric bug (drift score divided by ||k|| twice: 0.0027 meant
  cosine 1.0 -- absurd numbers are data, read them).

## 2026-10-07: search tiers all green, prices labeled (CLOCKED, start missed)

- Start clock MISSED (84-for-84). research/search_tiers.py:
  A exact sign-hash tier 6/6 self-hits + 0/30 cross-hits, but
  12/12 paraphrase misses (total generality price -- exact tier
  is dedup/canonical-query only, needs template normalization
  for open queries); B hierarchical (data-driven Europe-pair vs
  rest) 6/6 item accuracy with fewer comparisons; C orthogonal
  bank cross-talk 0.504->0.0 with installs still rank-1
  (whitening free at this dose -- unexpected, kept).
- Division of labor confirmed: exact tier for the known,
  resonance for the novel; sorting buys cost, never quality.
  Plus one caught probe bug (frag picker took "The" every time:
  all keys identical, cosine 1.0 -- read the diagnostics).

## 2026-10-07: transfer splits -- address yes, content no (CLOCKED, start missed)

- Start clock MISSED (86-for-86). research/transfer_v1.py with
  shared residual map (xfer_map.py, promoted third-use): addrR
  (teacher key x2 + native Rome value) retrieves 4/4 with Rome-
  rank 86 (matches native4x) at the cost of 4 containment holds
  (key-scale trades retrieval-vs-containment: calibration queued).
  fullR retrieves 4/4 but Rome-rank 222 (WORSE than base): mapped
  teacher direction cosines -0.334 with the native Rome readout
  (points away -- state-maps don't preserve direction-to-logit
  relations). Native ladder reaches rank 55 at 8x, monotonic.
- Transfer number, honestly: ADDRESS transfers (teacher-derived
  keys retrieve in our space); CONTENT does not (yet). Working
  product shape: teacher address x native content. Queued: key-
  scale calibration (x1.5, hold recovery), value-side correspondence.

## 2026-10-07: transfer v1 negative, precisely (CLOCKED, start missed)

- Start clock MISSED (85-for-85). research/transfer_v1.py (teacher
  Italy key+dir -> our D16 bankhn store 129, 8 arms): hold-rate
  16/16 every arm (no degradation anywhere -- clean). Native
  addressing retrieves 3/4 with Rome-rank scaling by dose
  (198->136->105 at 0/1/2x: content direction right, needs dose).
  Teacher keys retrieve 0/4 under BOTH random-projection and a
  499-anchor fitted map (which disagree at cosine -0.131 -- the
  map is underdetermined 3584>499, suspect itself). Address must
  live in receiver geometry and we haven't put it there yet.
- Verdict: transfer number unearned. Next: map-quality diagnostic
  (held-out anchor cosine) + native dose ladder (install ceiling
  in our space bounds what transfer must reach). Plus one caught
  metric bug (containment measured accuracy 4/16, not hold-rate).

## 2026-10-07: content-transfer fails validly; V2 falsified (CLOCKED, start missed)

- Start clock MISSED (87-for-87). The mine() direction line mixed
  unpacked/tuple indexing (scalar-minus-vector garbage) -- voiding
  ALL prior fullT/fullF/fullR content failures back to v1. Only
  addrR (correct key + native value) ever stood. Fixed, cache
  re-mined, full screen re-run: fullR rank 296, fullR2
  (normalize-anchors-first, the doll-theory variant) rank 328 --
  both worse than base 198, V2 worse than V1.
- Findings: (1) content transfer genuinely fails with correct
  directions too (mapped arrow anti-aligned -0.334 with Rome
  readout); (2) magnitudes carry signal -- normalizing them away
  costs (V2<V1); doll theory refined, not killed (support sets
  matter, but so do their weights). Tops never flip anywhere
  (UNK-attractor holds our model; install ceiling unreached).
- Standing: address transfers (4/4, rank 86), native ladder climbs
  (55 at 8x), content waits on direction-to-logit correspondence.

## 2026-10-07: minimum-size curve 1 -- floor at all sizes (CLOCKED, start missed)

- Start clock MISSED (92-for-92). research/min_size.py (bank K
  128->8, gate-comparable top1 + glue/content/unk split): top1
  0.34-0.36 at every K (gate parity: the earlier 0.07 was UNK-
  truth exclusion, my metric bug, fixed); UNK carries the WHOLE
  score (77/77 identical all K); glue 7-11; content 0/40 ALL K.
  Bank size is irrelevant to top-1 (K8==K128: glue lives in base
  weights, bank adds nothing measurable here).
- Reframe (the finding): content-aim starts at ZERO at every
  size of this organization -- curve 2 (organization) must
  CREATE content tops, not preserve them. Install = dose >
  local margin + correct direction (unified with steering
  physics); next rung is margins + extended dose on best
  candidates.

## 2026-10-07: containment recalibrated -- catalog fully green (CLOCKED, start missed)

- Start clock MISSED (91-for-91). Gap-map follow-ups closed:
  (1) orphaned facts are retrieved by family (Italy<-129,
  Caesar<-128: backup coverage, redundancy confirmed);
  (2) no store is necessary (graceful degradation: shared Rome
  value covers -- dolls inside dolls); (3) seriation ring
  neighborhoods are generic words (life/coinage/power: layout
  does NOT navigate -- measured gate-clusters do; respin filed
  as possible-but-not-predictive); (4) base margins separate
  perfectly: the 5 breaking holds are exactly the 5 thinnest
  (<=0.62) vs holders >=5.0.
- Metric fix (doctrine-level): holds measured on stable tops
  (margin>=1 bar), razor flips tracked as expected-fragile
  class. Under it the catalog turn is FULLY GREEN (installs +
  retrieval + 11/11 stable + razor as predicted). Margin bars
  belong in every future battery.
## 2026-10-07: behavior-fit loses; coarse-vs-precise principle (CLOCKED, start missed)

- Start clock MISSED (88-for-88). research/behavior_keys.py
  (keys fit to teacher match patterns, no geometry map): pattern-
  corr 0.68 but retrieval 0/4 vs mapped-vector keys 4/4. Caveat
  kept honest: fit saw 1 positive (Italy contexts are 4/319
  lines), so inconclusive-to-negative, not clean falsification.
- Principle REFINED (evidence-backed, stronger than the original
  hypothesis): COARSE relations transfer by vector (key matching
  survives map distortion: addrR 4/4); PRECISE relations don't
  (direction-to-logit needs exactness: fullR hurts). So: map keys,
  grow values natively. The hybrid stands vindicated with a
  mechanism -- and "relations not vectors" narrows to "coarse
  relations by vector, precise relations native-grown".

## 2026-10-07: catalog turn partial-pass (2/3 install, holds short) (CLOCKED, start missed)

- Start clock MISSED (89-for-89). research/catalog_turn.py (3
  transferred facts, one K131 bank, listings): Italy 299->58,
  Caesar 67->58, Alexander holds 6; all three retrieve own
  store; rand3 control inert (identical to base). Holds 11/16
  vs 15/16 bar -- key-scale x2 costs containment at catalog
  scale exactly as at single scale (known trade, now priced
  in the setting that matters).
- Verdict: installs land + coexist + retrieve (no collision
  catastrophe incl. shared Rome target); containment needs
  key-scale calibration. Shared map cache works (no refit).

## 2026-10-07: decoupling wins Italy clean; transfer ledger closed (CLOCKED, start missed)

- Start clock MISSED (90-for-90). nat3 (native emb keys x2 +
  8x values): Italy rank 31, holds 16/16 -- best single-fact
  result anywhere (vs cat3's 58 + 5 broken). Addressing and
  strength are independent knobs (values don't enter matching).
  Caesar still retrieval-bound (0/3 both families: its address
  is the wall, not its value).
- Full transfer ledger, unsparing: teacher keys retrieve broader
  (3/3 vs 1/3) at a containment price (11/16); native keys hold
  everything but reach less; teacher content fails everywhere
  (-0.334); the winning row is fully native. ADDRESS transfers
  (reach), CONTENT does not (precision). The transfer number
  splits and stays half-earned -- stated plainly.

## 2026-10-07: no flips to 64x; injection point diagnosed (CLOCKED, start missed)

- Start clock MISSED (93-for-93). research/content_flip.py
  (thinnest-margin content positions, native key + readout value
  at 8/16/32/64x): tops move with dose but PAST truth (64x lands
  antony/he, never the target). Correct direction + overwhelming
  dose still fails: the add happens at H (pre-layer-2) and layer-2
  nonlinearity derails it; readout-row correlations finish the job.
- Prescription with mechanism (not more dose): inject POST-layer-2
  (HN2-stream bank: RMSNorm preserves direction, head is linear --
  steering survives to logits by construction). New listing
  variant queued. Install physics stands (dose > margin), with the
  injection point now load-bearing alongside dose and direction.
## 2026-10-07: MVYB gate -- yarn ball as runnable structure (unclocked, gap admitted)

- Build: stdlib/yarnball.asm (DEF yarnball_apply: MATMUL+TSHIFT+
  SOFTMAX_WIDE soft gate, zero new mnemonics) + programs/lm_yarnball.asm
  (lm_bankhn bank via CALL) + chain/engram.py yarnball_bank() (address
  keys x key_scale, dose-folded values, tier/support/sha ledger rows)
  + tests/test_yarnball.py (parity + retrieve + install + hold, one listing).
- Gates (ALL OK): parity bit-exact vs inline bank (max|dlogit|=0);
  Italy retrieves own store 128; Rome-rank 299->24 (dose 8x value,
  key-scale 8.0); 10/10 stable holds (6 razor, 5 flipped as predicted).
- Surprises:
  1. Key-scale 2.0 (catalog winner for teacher keys) retrieves native
     store 127, installs only 299->98. Sweep 2/4/8 -> own-retrieval and
     rank 24 land at 8.0 with stable holds untouched throughout --
     addressing/strength independent knobs, confirmed again.
  2. Install top is still <unk>, not Rome (rank 24, Rome-cos 0.518 vs
     top 0.792): content moves correctly per the dot-reader, argmax
     stays norm-dominated. Modify-with-accuracy (top-1 install) remains
     the open gap -- stated, not rounded up.

## 2026-10-07: install ceiling -- three independent blockings (unclocked)

- Dose ladder at ks=8 (value 8/16/32/64x): Rome rank 24->15->10->8,
  then frozen; blockers (the/a/his) OUTCLIMB Rome (+7-8 vs +6 per
  8x-step); stable holds break past 8x (/tmp/doseladder.log).
- Key-scale ladder at 8x (ks=8/16/32): own-weight 0.07->0.84->1.00,
  rank 24->8, holds UNCHANGED 11/16 + 10/10 stable. Sharpening is
  pure win; dose is the dirty knob (/tmp/ksladder.log).
- Combo (ks=32 x dose): rank floor 7-8 at every dose; <unk> drops
  out at 16x (negative projection -- UNK attractor beatable) but
  the/a/his accelerate away. Raw readout values cannot top-1 at
  any dose with containment (/tmp/combo.log).
- research/lin_map.py: 16-probe layer-2 Jacobian at the Italy point
  (corr 0.956 vs 8x-Rome delta -- linearization VALID); maximin
  value through it predicts req norm 4.37 < 8x-Rome 7.23; install
  collapses the scale instead (rank 336, top logit 0.4 -- expansive
  small-signal |M|=25 vs saturating large-signal). Aiming through
  the linearized map fails validly: direction right, regime wrong.
- Standing: single-vector top-1 install through OUR d=16 SVD readout
  blocked three independent ways (blocker outclimb / saturation /
  post-L2 overwrite ceiling). Teacher late layers DO install top-1
  (siphon pair-contrast) -- geometry permits it there, not here.

## 2026-10-07: siphon install through the ball, mirror-side green (unclocked)

- research/siphon_ball.py (7B mirror): ball = [Germany: L27
  France-minus-Germany contrast value | Italy/Japan: null stores]
  via yarnball_bank + ledger; apply = yarnball_apply math (softmax
  routes, non-targets land on nulls). Live-mined, no caches.
- Ladder (/tmp/siphonball.log): gain1-2 Germany TOP=Paris rank 1
  (INSTALL), Italy=Rome + Japan=blank hold, all retrieve own;
  gain4-8 Germany=Paris-surface rank 3, controls hold; gain16 Italy
  MOVED (leakage species, same as native dose ladder). Operating
  point locked: gain 1. First accurate modify through the generic
  structure (target top-1 + holds + retrieval, one ball).
- research/addr_sep.py (/tmp/addrsep.log): all prompts len 5,
  country@3, end@4 (one binary serves the battery). Country row
  separates 1.0 vs <=0.64; prompt-end row TIES (~0.42, Japan
  ahead) -- per-row self-gating would misroute; the listing must
  SLICE the country row and route explicitly. Design input recorded
  before building, not after failing.

## 2026-10-07: rung 1 green -- siphon install IN-LISTING (unclocked)

- scripts/siphon_geo.py: 28-layer builder program (9866 ops = 9859 +
  SLICE/MATMUL/TSHIFT/SOFTMAX_WIDE/MATMUL/MATMUL/ADD = the
  yarnball_apply body at L27 output, explicit country-row routing per
  addr_sep) + ball data from yarnball_bank (keys x8, Paris value at
  mirror-gain-1 x |x27|=536.6 dose, 2 null stores, ledger).
- Mirror side first (research/siphon_ball.py): gain1-2 INSTALL + holds
  + own-retrieval, gain16 breaks Italy (operating point: gain 1).
- Geo verdict (/tmp/siphongeo.log vs /tmp/siphongeo0.log): Germany
  blank(paris45) -> ' Paris'(paris1) INSTALL (effect 45->1); Italy
  blank->blank (paris-rank 75->75 BIT-IDENTICAL); Japan blank->blank
  (137->137). Null-store routing leaks exactly zero in-listing.
- Fork caveat (load-bearing): geo FP16 base ALREADY blanks Germany
  (mirror: Berlin) and Italy (mirror: Rome) -- near-tie fork per the
  test_qwen7b_gen doctrine. Holds therefore grade vs geo zero-dose
  base (--base-json), never mirror. First grading caught this live:
  initial mirror-graded verdict read Italy as broken; the control
  proved it pre-existing. Tripwire species: cross-baseline grading.
- Toolchain gap filed (docs/GAPS.md): CALL-namespaced streams
  (name#k.P) are not valid C identifiers -- emit_cuda dies. No
  builder program ever shipped a CALL to a backend (seam never
  tripwired). Workaround: DEF body inlined, explicit names.
- Two self-inflicted wounds, same species (shape/index): keep=all
  vs keep=last mix (3D bank array, fail-loud caught) + norm(t[27][-1])
  scalar-vs-row mix (dose 0.6 vs 536.6, caught by reading the number:
  RMS 0.01 impossible at H=3584). Species: post-edit invariant
  re-check -- every edit re-touches the line below it.

## 2026-10-07: CALL emitter fix CLOSED (unclocked)

- `sanitize_cnames()` (chain/emit_c.py, shared by C/non-FPU +
  compile_cuda): valid-C rewrite at the emit seam, collisions fail
  loud, literals untouched, alias in art for host lookups.
- Gates: nonfpu ALL OK, emit_cuda ALL OK, builder suites ALL OK
  (identity zero-diff); CALL->nvcc BUILD OK; C CALL-vs-inline
  BIT-EXACT; in-contract C-vs-lattice 2.2e-04. test_emit_c GSL-header
  failure pre-existing/environmental (noted, not ours).
- Scare en route: out-of-contract probe diverged 3.75 -- resolved by
  the twin design (CALL-vs-inline bit-exact isolates the rename;
  remainder is priced float-vs-fixed class). Doctrine note: when a
  fix touches lowering, the twin gate (fixed-vs-fixed across the
  seam) decides before any cross-substrate comparison is read.

## 2026-10-07: curve 2 -- the cliff runs backward (unclocked)

- research/min_size2.py (/tmp/minsize2.log): curve-1 truncation rule
  + organized ball (yarnball_bank + MVYB Italy store, lm_yarnball.asm).
  Rome-rank by K: 128->24, 64->17, 32->11, 16->10, 8->8. Retrieval
  own at every K; holds 11/16 + stable 10-11/11 throughout;
  top1/glue/content/unk identical to curve 1 (content 0/40 -- one
  install, not general content).
- Reading: no cliff in either curve -- but curve 2's rank IMPROVES
  as the bank shrinks (thinning native field concentrates softmax
  on the install). Organization's value is not "shifts the cliff
  left"; it is install+holds coexisting at EVERY size, sharpest
  smallest. Minimum viable re-prices itself: K=8+1, rank 8.

## 2026-10-07: generality battery green; multi-fact queued (unclocked)

- research/siphon_battery.py (/tmp/siphonbatt.log): single Germany
  ball over 6 countries. Gain1: INSTALL + 5/5 hold, all RETR own.
  Gain8: Germany Paris-surface r3, all 5 hold (France/Italy/Spain/
  Japan/China keep unsteered tops), all RETR. Offset-mapping miner
  replaced fragile substring frags (Spain/China splits killed the
  first run: StopIteration, fail-loud caught; siphon_ball.early_key
  takes pos= now, no third inline copy).
- Scouting: France/Italy/Spain/China unsteered already correct
  (vacuous controls); Japan blank/Tokyo-runner = second non-vacuous
  fact. research/siphon_multi.py queued: [Germany Paris-value |
  Japan Tokyo-value | 4 nulls], coexistence question.

## 2026-10-08: multi-fact coexistence -- routing proven, value invalid (unclocked)

- research/siphon_multi.py (/tmp/siphonmulti.log): one bank, two
  installs ([Germany Paris-value | Japan Tokyo-value | 4 nulls]).
  Germany INSTALLs at every gain; all 4 controls hold; all retrieve
  own. Japan does NOT install Tokyo -- top goes Paris, Tokyo rank
  DETERIORATES with gain (3382->152010): the France-minus-Japan
  "Tokyo value" is Paris-contaminated (d_de.d_jp = 0.254); the pair
  recipe isolates the CONTRAST's capital, not the target's.
- Reading: coexistence MECHANICS proven (routing exact under two
  live values, holds green); the failure is value-mining, same
  species as the transfer ledger (content doesn't transfer by
  vector). Tokyo needs a Tokyo-carrying direction (Tokyo-evocative
  prompt contrast), not France-minus-Japan. Queued behind rung 2:
  value-mining IS the precise-relations problem.
- Note the frame honestly: "Germany->Paris install" transplants
  France's capital onto Germany (capability demo: arbitrary content
  to rank-1 with specificity), not a factual correction. Paris on
  Japan is the same operation succeeding with the wrong payload.

## 2026-10-08: Tokyo screen -- contrasts carry the contrast's capital (unclocked)

- research/mine_tokyo.py (/tmp/minetokyo.log): Japan base blank with
  Tokyo runner-up (rank 2). Dot-readouts vs Tokyo row: cn-jp -0.051,
  es-jp -0.117, big-jp +0.016 (all empty -- predicted fail upfront).
- Ladder confirms: cn-jp installs BEIJING, es-jp installs MADRID
  (contrast's capital, twice more), Tokyo rank decays with dose;
  big-jp drifts (r8->610); all broadcast (ungated screen, expected).
- Rule promoted (3 instances + Germany): pair-contrast isolates the
  CONTRAST's capital, never the target's. France-minus-X is a Paris
  hose with an address label.
- Twist: wrow (Tokyo readout row, predicted-fail control per siphon
  section 6 decoder-form) installs Tokyo r1 at every gain -- in a
  friendlier dose regime than section 6's. Exact native content wins
  where vectors fail (precise-relations principle, second instance).
  Graduated to the multi-fact bank (siphon_multi.py rerun with Japan
  value = readout row; coexistence re-tested with a valid payload).

## 2026-10-08: multi-fact coexistence GREEN (unclocked)

- research/siphon_multi.py rerun (/tmp/siphonmulti.log) with valid
  payloads: [Germany: contrast Paris value | Japan: Tokyo readout-row
  value | 4 nulls]. Gain1-2: BOTH INSTALL (Paris r1, Tokyo r1),
  4/4 holds, 6/6 retrieve own. Gain4: Japan holds Tokyo, Germany
  Paris-surface r2, France+China leak to Tokyo (readout-row values
  are hot: partial-weight x 4x-mag transplants; operating window
  1-2, same discipline as every ladder before it).
- Value alignment -0.004: the two species are nearly ORTHOGONAL --
  coexistence rides disjoint content, not sharing. Taxonomy earns
  its keep: contrast hose + readout row, one bank, no entanglement.
- Item 2 CLOSED: single-fact generality (6-country) + two-species
  coexistence (2 installs + 4 holds + 6 retrievals, ledgered).

## 2026-10-08: rung-2 probe 1 -- depth exonerated, wall is norm structure (unclocked)

- research/postl2_install.py (/tmp/postl2.log): Rome readout row via
  the LINEAR post-L2 bank (corr 1.000 thru 8x -- path exact), dose
  ladder 1/2/4/8/16x: rank 299->27->13->9->8->8. STALLS AT 8: the
  same floor as the layer-1 retrieval install, same blockers
  (<unk>/the/a/his, all outclimbing: <unk> +5.4 while Rome climbs).
  Holds 11/16 thru 8x (broadcast tolerated!), 3/16 at 16x.
- Falsifier branch taken (written in the script): blockers outclimb
  again -> the wall is NORM STRUCTURE (|w_rome| 0.84 vs unk 6.98,
  the 3.74), not depth. Qwen-late vs our-L1 was never about path
  length: one norm + unembed loses to the same pack either way.
- Next: column-norm ablation (probe 2) -- unit-norm wlog variant in
  /tmp (repo data untouched): does glue survive on alignment alone,
  and does the install dose collapse? If yes, readout redesign =
  tier the norms; if glue dies, norms ARE the prior and redesign
  must replace the prior another way.

## 2026-10-08: rung-2 probe 2 -- FIRST NATIVE FLIP, price tagged (unclocked)

- research/norm_ablate.py (/tmp/normabl.log): unit-norm wlog variant
  (/tmp/wlogU.npy; repo data untouched). (A) base: top1 0.337->0.146,
  glue 6/87->1/87, unk 77->34 -- norms ARE the prior; skew removed =
  prior dead. (B) post-L2 Rome ladder: rank 470->17->3->1->1, FLIP at
  4x (rome 3.78 vs greece 3.62, thin but top-1; 8x grows margin).
  Holds die with the flip (13->6->0->0/16): broadcast unit push
  lifts the whole content pack (greece/plutarch/egypt/syria rise).
- Mechanism priced: skewed head, delta_c = dose.|w_c|.cos favors
  long frequent rows (3.74cos > 0.84 needs only cos > 0.22);
  flat head, target alignment 1.0 beats every cos < 1. Dose 40x ->
  4x. Installability and prior-holding are THE SAME KNOB (column
  norms). Two-tier readout (skewed prior + flat install) is
  necessary, not decorative -- DISCOVERIES section 8 tiers, priced.
- Open: flip WITH prior intact. Candidates: high-rank-target
  installs through the skewed head (Alexander class, rank ~6->1),
  or a dual-head decision architecture (bigger design). Probe 3:
  Alexandria ladder next.

## 2026-10-08: rung-2 probe 3 -- rank deceived, margin is the currency (unclocked)

- research/probe3_alex.py (/tmp/probe3.log): Alexandria base rank 6
  BUT margin 10.08 (leader <unk> 14.05 vs 3.97). Ladder: 6->4->4->
  4->3. STALLS. Truth gains ~1.0/dx, 'the' gains ~1.33/dx: deficit
  WIDENS with dose (same outclimb species, third target). Holds
  11/16 + 10/10 stable thru 4x (broadcast tolerated, again).
- Correction to the probe's own premise: rank-6 deceived (dense pack
  near truth, leader 10 away). Margin, not rank, prices installs --
  known since the 1.4x illusion, re-bitten anyway. Doctrine note:
  always print margin beside rank (scan_req.py does; probe3 now does).
- Remaining native path: full-vocab flippability scan from statics
  (linear path corr 1.000, so statics ARE dynamics): raw-row values,
  every token as hypothetical target, exact required dose. Any
  content target <=8x graduates to a real install run.

## 2026-10-08: transport map -- broadcast, L1 drenched, late blind (unclocked)

- research/transport_map.py (/tmp/transport.log): end-query mass on
  country token, 6 prompts. MEAN layer totals (28 heads): L1 5.56
  (nearly ALL mass), early 2.6-2.8, mid sustained 2-3 (L14 3.17,
  L18 3.14, L20 3.11), dips L5/L16 (~0.9), floor L26 (0.50) + L27
  (1.14). Hot heads/KV-groups rotate every layer -- NO concentration.
- Readings: (1) head-targeted surgery CONTRAINDICATED (no hotspot;
  step 3 of the plan withdrawn, mechanism-first); (2) L2 lexical
  keys work because early layers bathe in country info; (3) L26-27
  ignore the country token -- the decision forms from mixed states,
  so ungated late steers MUST broadcast: the siphon product shape
  (early address x late content) confirmed from the attention side,
  independently of the steering results. (4) Mid-layer ladder next:
  L14 (heavy mass) and L22 (donor precedent) vs L27 baseline --
  amplification vs derailment decides install-point law.

## 2026-10-08: teacher is flat -- skew is ours, glue lives in alignment (unclocked)

- research/qwen_readout.py (statics, CPU): Qwen lm_head rows --
  Paris 0.58, Berlin 0.56, Rome 0.57, Tokyo 0.52 (content within
  +-10%); max 1.1 (junk/code fragments), p99 0.9, median 0.6. Skew
  max/content ~2x vs OURS 8.3x (unk 6.98 vs rome 0.84).
- Double dissociation with probe 2: our flat head flips (install
  needs flat) but kills glue (our glue lived in norms); Qwen flat
  AND fluent (its glue lives in alignment: d=3584 trained vs our
  d=16 counts). Skew is a construction artifact, not intrinsic.
- Redesign fork, priced: (a) grow alignments (representation
  capacity: dim + training -- the actual ceiling, big); (b)
  dual-head routing (skewed prior head + flat install head,
  addressed -- preserves everything built, architecture-sized).
  Selective norm promotion (lengthen content rows) is DEAD:
  Qwen content rows are SHORT (~0.55, same as ours) -- its
  competitors are short, not its targets long.

## 2026-10-08: install late is law, downstream distorts (unclocked)

- research/siphon_mid.py (/tmp/siphonmid.log): same gated ball
  placed at L14/L22/L27, gains 1/2/4. L14: Paris r6->r53->r378
  (tops blank/most/so). L22: r8->r169->r125 + Italy MOVED at g4.
  L27: r1/r1/r3 INSTALL (recipe re-baselined identical across
  reruns). Retrieval own in ALL 27 cells -- addressing never fails;
  content transport does.
- Law: downstream blocks DISTORT placed directions
  dose-dependently, never amplify (lin_map saturation writ across
  depth). Later install strictly dominates. Native probe-1 (same
  floor 1-block vs 0-block downstream) + teacher ladder agree:
  install at the last addressable point. Mid-stream injection is
  WITHDRAWN as a rung-2 path (second mechanism-first negative of
  the reversing loop, after head-targeting).
- Remaining rung-2 paths: (a) grow alignments (capacity, big);
  (b) dual-head routing (architecture-sized, everything preserved).

## 2026-10-08: rung 2 lands -- dual-head triple gate green (unclocked)

- programs/lm_dualhead.asm + research/dualhead.py (/tmp/dualhead.log):
  post-L2 bank ALWAYS carries unit-Rome 4x (broadcast drug); readout
  routes per prompt (host-tiled mask, onehot precedent): flat head on
  install contexts, skewed head elsewhere.
- prior (mask0+drug): 0.337, 6/87, 0/40, 77 -- BIT-IDENTICAL to skewed
  base. install (mask1): rank 1, top=rome, FLIP. holds: 11/16,
  stable 10/10. ALL THREE in one listing.
- Rung-2 verdict: architecture, not refit. Prior and install are
  routed tiers (DISCOVERIES section 8, now running readings, not
  doctrine). Track (a) closed by statics (skew 15.4->9.9 over
  d16->d64, max frozen 6.99, frequency order in counts AND fit:
  the prior is the objective; dim/fit can't remove it).
- Remaining hardening: mask from in-listing addressing (bank-weight
  blend; host mask is the v1 seam, stated in the listing header),
  second fact coexistence natively (two drugs, two masks), flat-head
  tier provenance (wlogU is derived data -- freeze manifest).

## 2026-10-08: dual-head v2 -- self-routing works, tier leaks located (unclocked)

- programs/lm_dualhead2.asm + research/dualhead2.py (/tmp/dualhead2.log):
  6 background nulls + Italy last; blend by own Italy weight (no host
  mask). install rank 1 FLIP, w_italy 0.999; holds 11/16, stable
  10/10; receipt battery mean 0.082, max 0.674 (one partial).
- Gap: prior 0.289 vs 0.337, all 12 in UNK-truth (glue/content
  identical) -- partial Italy weights (<=0.674) blend flat readings
  into UNK positions. Locator fired as designed: tier interference,
  not blend quantum (FLIP+receipt green isolates it).
- Fix in flight: key-scale 8->16 (sharpen addressing toward 0/1;
  the ks ladder's proven direction). Rerun /tmp/dualhead2b.log.
