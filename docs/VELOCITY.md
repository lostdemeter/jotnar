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
