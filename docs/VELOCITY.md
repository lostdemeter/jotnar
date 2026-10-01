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

## 2026-10-01: anatomy-2 + fan-in law + LLM design (CLOCKED)

- GPT-2 survey (biases/pos-emb/gelu_new) + LAYERNORM exposure (#43) +
  split-claim gates + formal design draft (#LIB-079).
- Wall time: clock 15:52:34Z -> (ends at commit; survey + exposure +
  5 misdiagnoses + design).
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
