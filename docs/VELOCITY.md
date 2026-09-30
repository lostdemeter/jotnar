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
