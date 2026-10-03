# Spectral Theory of Aim (v0.5, 2026-10-03 — seed-aim adjudication + O1 closure; supersedes v0.4, prior text retained with marks)

A theory of why prediction accuracy moves (or doesn't) in
count-built integer transformers. Status per section: DEFINED,
POSTULATED, MEASURED, or OPEN. Nothing here is claimed beyond
its tag. Falsification criteria are part of the theory.

## 1. Primitives (DEFINED)

- **Ends**: embedding matrix E (V×D) and readout W (D×V) from
  log1p-SVD rank-k of bigram counts: E = U√s, W = √sVᵀ.
- **Peakiness** P = s₀/s_{k-1} (spec ratio). Measured range in
  this program: 1.8 → 23.2.
- **Twin number** T: relative endpoint distance between
  active/passive (order-transformed) runs of the same content.
  Lower has historically been read as "more invariant"
  (this reading is REFINED in §3).
- **Aim** A: top-1 accuracy (piece-top1 over 566 probe
  positions; word-top1 over 330 boundaries; same frozen probe).
- **Norm map**: readout row norms ||w_i|| vs piece frequency
  (v0.1: r = 0.776; frequent rows ~12x mean norm. v0.2 fixture,
  §7.1: r = corr(log1p-freq, ||W[:,i]||) = 0.778 reproduced on
  groki piece32 ends + piece_bigrams col-sums; plain-log r = 0.748
  does NOT match — prior phrasing underspecified. 12x = top-1 row
  12.4x / max 13.9x mean; top-64 mean is 5.5x — prior phrasing
  imprecise. Numbers kept with fixture, not as bare constants.)
- **Capacity** C and **sensitivity** S (proposed decomposition,
  §3): twin# = C + S (additive working hypothesis; functional
  form OPEN).

## 2. Postulates

- **P1 (giants carry frequent-aim).** Top singular modes align
  with frequent directions. MEASURED: norm-freq r = 0.776 (v0.2:
  0.778 under log1p fixture, §7.1);
  groki flatten 4.6->1.8 kills frequent 32->24 with rare flat
  at 1 (giants diluted, nothing gained).
  [v0.2 AMENDED — exact pin FALSIFIED, direction holds: rebuilt-bank
  re-measurement gives frequent 32->25 (total 33->26/566), orig-bank
  32->23 (total 33->24/566); claimed 32->24 matches neither exactly.
  Twin 0.6126 vs claimed 0.614 reproduces (rebuilt-bank fixture).
  McNemar base-vs-flat 12/5, chi2 2.12, p~0.15 n.s. — see §7.3.]
- **P2 (tail carries discrimination).** Rare-continuation
  distinctions live in small singular modes. MEASURED:
  w103 flatten 12.9->4.6 recovers aim 0.011->0.034 AND rare
  0->1 (tail reweighted upward). BOUND: tail mass is finite
  (top-200 shares); pushing C past corpus tail buys nothing
  (prediction, untested).
  [v0.2 FALSIFIED-AS-STATED (conflation, not mechanism): the sentence
  mixes w20k spec (12.9) + w103full aim (6->19/566) + w20k rare (0->1)
  into one experiment that never ran. Superseded by §7.4 split rows:
  (a) w20k flatten 12.9->4.6: 18->23/566, rare 0->1, McNemar 1.07
  p~0.30 n.s.; (b) w103full flatten 23.2->4.6: 6->19/566, rare 0->0
  (groki 178/388 split), McNemar 6.26 p~0.012. Tail mechanism survives
  as two rows; the single-row claim dies.]
- **P3 (threshold gate).** Readout argmax sees only order;
  sub-threshold mass is invisible. MEASURED: temperature moves
  truth-rank +3.9 / truth-prob up 251/330 while argmax frozen
  33/566 x5 (coupling is threshold-gated, not absent).
  [v0.2 SURVIVES (ranked strongest, §7.2; not re-run this turn —
  5x replication stands; re-run queued, not assumed).]
- **P4 (capacity/sensitivity sum).** Spectra move C (twins AND
  aim co-move); fits move S (twins down, aim holds-or-up).
  MEASURED four cases: refit 1.048->0.665 + 0.049->0.058 (S);
  headt twin down + top1 holds (S); w103-flatten twin#
  0.292->0.520 + aim up (C up); groki-flatten twin# 0.665->
  0.614 + aim down (C down). All four sort with zero
  exceptions to date.
  [v0.2 AMENDED — fixtures pinned (§7.5): transfer via
  programs/lm_d32.asm, refits/flattens via programs/lm_d32_headt.asm;
  flatten banks REBUILT from flattened ends with own words (orig-bank
  twins wrong: groki 0.578 vs 0.614, w103 0.380 vs 0.292).
  Paired: refit 28->33/566 discord 2/7 McNemar 1.78 p~0.18 n.s.;
  w103full 6->19 discord 5/18 McNemar 6.26 p~0.012; groki-flat 33->26
  discord 12/5 McNemar 2.12 p~0.15 n.s. Directional sort 4/4 holds;
  inferential strength 1/4. Additive twin#=C+S stays POSTULATED (no C/S
  values derived). Universality FALSIFIED by R1 MISS (§7.7) —
  superseded by scoped P4': C-gains scale with distance from optimum.]
  [v0.3 LADDER RATIFIED (§8): w103full tau ladder at fixed fit traces
  the first C(P) curve — twins 0.2920/0.4146/0.4997/0.5200/0.5275/0.5284
  across spec 23.18/12.36/6.59/4.60/3.30/2.19 (exact reproduction).
  Joint cells now triple-filled (down/up refit; up/up peaky->sweet;
  down/down groki-overflatten + w103full-plateau + BPE-500-collapse-aim);
  up/down cell predicted EMPTY (R4). Twin-only readings ambiguous,
  period — joint or nothing (D5).]
  [v0.4 P4-AS-UNIVERSAL DEAD (second kill, §9): piece-D64 O-clean
  (wv=I1.0, wo=I0.5) gives twin UP 0.7329->1.1900 AND aim UP 20->40/566,
  McNemar 13.88, p~0.0002 — a fit moving twins UP with aim UP
  falsifies "fits move S (twins down)". QK-dial confirms (31->40->48
  ->54->57->60->61 across s=0.1-1.0, twins co-moving, endpoints
  significant). D32 QK (1.048->0.952, 28->32 p~0.29) keeps S-pattern:
  same-path moves sort S at D32, C at D64 — the differentiator is
  attention-regime, not path. Successor: regime-indexed C/S (R5,
  CONJECTURE). Historical rows stand; D5 is now the central law.]
- **P5 (norm map correct).** Base rates stay: whitening
  destroys (0.058->0.012), tempering declines monotonically
  to incumbent optimum. Feed likelihood; never divide the map.
  [v0.2 STRONG-FORM FALSIFIED (decisive probe, §7.6): whitening recipe
  was missing. Scale-collapsed whiten (sv->1, rebuilt bank) gives
  0/566 (destroys — scale confound); scale-preserved whiten
  (sv->mean(sv), rebuilt bank) gives 33->25/566, discord 17/9,
  McNemar 1.88, p~0.17 n.s. — equalizing the map at constant scale
  does NOT significantly destroy. Inference "norms ARE the base rate"
  confounded by operating-point scale (cf. w20k 2.7x norms). Weak form
  survives (map correlates r=0.778, causality unproven). Policy "never
  divide the map" demoted MEASURED->CONJECTURE.]

## 3. Derived claims

- **D1 (twin dial has two needles).** Any twin movement must
  be attributed C or S before interpretation. Corollary: past
  "twin wins" split into sensitivity cuts (refit, headt) vs
  capacity moves (spectra); future twin claims must state
  which. (Methodological claim; enforced going forward.)
- **D2 (routed specialization = S-cut + C-keep).**
  Flat@starts + sharp@continuations holds twins (0.665, twin
  contexts all start-words) while top-1 moves 33->46
  (McNemar 9.6, p~0.002). The only joint win on record, and
  it fits: routing cuts sensitivity where it matters
  (fragments) while holding the geometry where twins live.
- **D3 (supply irrelevance range).** Within one corpus
  family, counts scale moves P but not A past the flattening
  correction (w20k/w103-pilot/w103full converge post-tau).
  Cross-family supply (new distributions) is the untested
  exception (c4/dolma need downloads).
  [v0.3 SUPERSEDED (convergence reading FALSIFIED): flattened w103full
  plateaus at twin 0.528, groki sits at 0.665 — same spec 4.6, different
  twins. Stacks do NOT converge to one number. Succeeded by D4.]
- **D4 (floors are content, curves are spectra).** (v0.3, MEASURED)
  Flattening converges each corpus toward its OWN twin floor
  (w103full ~0.53, groki 0.665 at the same spec 4.6); spectra trace the
  curve, content sets the floor. Corollary: twin LEVELS are not
  comparable across corpora — only within-corpus deltas and joint
  cells travel. Cross-corpus twin rankings (e.g., "w103 more invariant
  than groki") are hereby uninterpretable without a joint aim leg.
  [v0.4 EXTENDED: k=64-fit levels 1.54–1.57 join the incomparables
  (groki-32 0.61–0.68, w103full-32 0.29–0.53). Floors differ by
  (k, corpus, fit) jointly — levels travel nowhere, deltas + cells
  travel everywhere.]
- **D5 (joint or nothing).** (v0.3, instrument doctrine) The completed
  joint table — down/up S-cut (refit, headt), up/up C-gain (peaky->sweet
  flattening), down-or-hold/down collapse (groki-overflatten,
  w103full-plateau, BPE-500-collapse-aim) — shows twin direction alone
  does not determine aim direction (down->up in refit, down->down in
  collapse). Every twin claim must carry its joint aim leg; twin-only
  readings are ambiguous, period.   Twin bars additionally require the
  collapse-floor check (§8.5: twin≈0 readings HELD pending fixture —
  degenerate indistinguishability is not invariance).
  [v0.4 D5 PROMOTED to central law: with P4-universal dead twice over
  (R1 spectra-universality, O-clean fit-directionality), the joint table
  is what survives — four cells (down/up S-cuts: refit, headt, D32-QK;
  up/up C-gains: peaky->sweet spectra, O-clean, QK-D64-dial; down/down
  collapses: groki-overflatten, w103full-plateau, BPE-500-collapse-aim;
  up/down: R4-threat-row only). All future moves are filed by cell,
  never by twin direction alone.]

## 4. Preregistered predictions (falsifiable)

- **R1.** BPE-500 ends flattened toward spec ~4.6 (tau ~0.81):
  twin# RISES toward ~0.5 AND piece-top1 rises from transfer
  baseline. Both up together, or P4 dies. (Sealed 2026-10-02;
  unrun.)
  [v0.2 RUN 2026-10-03: MISS, no spin (§7.7). Fixture: BPE-500 ends
  (V=538, nat spec 6.469) + piece32-fit body + data/bpe500/bank64,
  programs/lm_d32_headt.asm, BPE-500 probe N=832 (same 10 sents,
  BPE-500 tokenization). Transfer twin 0.3589 (reproduces P3 pin
  0.359); flat tau 0.8174 (spec 4.6, rebuilt bank) twin 0.3964 —
  RISES (direction hit) but NOT toward ~0.5 (miss by 0.104; orig-bank
  variant 0.3542, wrong direction — rebuilt fixture stands per
  precedent). Top-1 58->59/832 (0.0697->0.0709), discord 7/8, McNemar
  0.0, p~1.0, 95% CIs [0.0543,0.0891] vs [0.0554,0.0904] overlap —
  NO RISE. Joint "both up" prediction MISS. Implication: C-gains vanish
  near optimum; P4-as-universal dies, P4' scoped successor stands.]
- **R2.** Any pure sensitivity move (new head specialization,
  new twin-orthogonal router): twins down-or-flat, aim NEVER
  down beyond noise (±0.019). An aim-down sensitivity move
  kills P4's independence claim.
- **R3.** Flattening past corpus tail buys nothing: pushing
  any stack's spec below ~half its natural value costs aim
  without twin benefit (groki 1.8 point generalizes).
  [v0.3 ADJUDICATED (§8): mechanism CONFIRMED, bound REVISED.
  w103full gains continue 11.6->4.6 in BOTH twins (0.415->0.520)
  and aim (→19/566) — the ~half-natural bound (11.6) is FALSIFIED
  as universal. Successor R3': cost region begins below ABSOLUTE
  spec ~4.6 (k=32 rung resolution; optimum interval (3.3,6.6]):
  plateau 19->14/566 while twins saturate +0.008; groki 1.8 and
  BPE-500 tau-0.4 (spec 2.11, 58->38/832, McNemar 9.03 p~0.003)
  both below-4.6 cost-aim rows. Knee ≈ groki natural 4.64 across
  four corpora/tokenizations — see O1.]
- **R5.** (v0.4, regime-indexed C/S, §9 — CONJECTURE with anchors)
  Cleaning/refit moves sort S (twin down, aim holds-or-up) iff head
  attention entropy is healthy (anchor: D32 ~0.6), and C (twins AND aim
  co-move, either sign) iff attention is collapsed (anchor: D64 ~0.15).
  Entropy is measured per run from mirror-side P1/P2 weights (free).
  Falsifier: a cleaning move sorting S under collapsed entropy, or C
  under healthy entropy, with aim legs + paired tests. (Preregistered
  2026-10-03; unrun. Quantitative cut OPEN — anchors only.)
- **R4.** (v0.3, from the joint table, §8) The twin-up + aim-down cell
  is EMPTY: no spectra move (C co-moves) and no fit move (S holds-or-up)
  produces it. Any future move with twins up beyond noise AND aim down
  beyond the R2 band (±0.019) kills P4'. (Preregistered 2026-10-03;
  unrun — the falsifier the table owes the theory.)
  [v0.4 THREAT ROW LOGGED, letter holds (§9.6): groki-32 sharpened to
  spec 6.8 (twin 0.6745, pin 0.675 verified; end rung tau=1.5/spec=10.0
  twin 0.6564 verified; aim 33->27/566, discord 7/1, McNemar 3.12,
  p~0.08 n.s., -6ct within band; down-slope 33->27->25 with 6.8-vs-10.0
  discord 7/5 p~0.77 and nat-vs-10.0 12/4 p~0.08. Directional up/down
  occupation with sub-band magnitude: threat model, not body. The 6.8
  joint point rides gated execution (D32 parity 47/50dB retro-gated,
  §9.7); the 10.0 aim leg rides RED LOGITS headroom (1.36dB, refmax
  96 — HELD). R4's letter (band + noise qualifiers) holds; its spirit
  is on notice — every future twin-up move takes aim legs by D5.]

## 5. Open derivations (not hidden)

- O1. What sets the optimum (~4.6 here): corpus Zipf
  exponent? rank k? V? No functional form yet -- points,
  not curves.
  [v0.3 NARROWED (§8.4): knee at spec 4.6 across groki (natural),
  w20k (->4.6 gains), w103full (knee rung 4.60), BPE-500 (->4.6 target,
  flat gradient at optimum per R1 miss) — four corpora/tokenizations
  with different Zipf exponents, densities, V. Corpus-Zipf RULED OUT
  as the setter; geometry (k=32?) remains. k-sweep unbuilt — the
  functional form is still open, the candidate list is shorter.]
  [v0.4 CLOSED-MULTIVARIATE (§9.5): k=64-fit aim is FLAT 59–61/566
  across spec 1.55–5.81 (all pairs n.s., p 0.75–0.84; fully gated
  57–64dB) — NO aim-knee where k=32 knees at 4.6 (same corpus, same V,
  same probe). Universal aim-knee FALSIFIED; k-set-as-single-number
  FALSIFIED with it (no second located knee exists to fit any f(k)).
  Verdict: knee presence/location = f(k, fit, corpus-shape); absolute
  4.6 holds at k=32-fit only. REMAINDERS (explicit, not hidden): k-vs-fit
  confounded by design (each k refit at its own point — separation
  needs same-fit-across-k, unbuilt); V-set parked (frozen V=2038);
  seed-aim legs unmeasured (k=64 seed twins flat 1.6–1.9, aim unknown).
  O1's as-posed question is answered; O6 (k-vs-fit separation) + V-set
  + seed-aim succeed it.]
- O2. Functional forms C(P), S(fit): four anchor points
  exist (table §2/P4); the curves between them are unmeasured
  (needs spec sweep at fixed fit + fit sweep at fixed spec).
  [v0.3 FIRST CURVE (§8.1): twin ladder at fixed fit IS C(P) traced
  (rise + plateau + knee). S(fit) curves still unmeasured; additive
  twin#=C+S still untested — the ladder alone cannot decompose C vs S.]
  [v0.4 THREE CURVES (§9.4): w103full-32 saturates (0.292->0.528),
  groki-32 peaks (~6.8: 0.614->0.665->0.675->0.656), k64-fit peaks
  ~4.1 (1.414->...->1.571->...->1.541, fully gated flatten-side).
  Twin-curve SHAPE is corpus-indexed (saturate/peak/monotonic).
  S(fit) curves still unmeasured; additive form abandoned with P4
  (D5 needs no decomposition — cells, not components).]
- O3. Temperature's address: moves twins 66% with argmax
  frozen and decision-mass shifted -- S, C, or third axis?
  Does not file cleanly under P4 (stated gap).
  [v0.4 CLOSED (S, §9.3): temps move twins-only in BOTH widths —
  D32 beta_b sweep (0.853/1.107, aim lateral), D64 headt (twin
  1.1900->1.1654 with literally ZERO decision flips, discord 0/0;
  beta_b honors-key verified 1.1654 vs 1.1900). The "third axis" was
  threshold-gating all along (P3 explains the shifted-mass/frozen-argmax
  split: mass moves sub-argmax). No residue left open.]
- O4. D-dependence: does capacity scale with width, or only
  with spectra? (D64 parked with separate diagnosis; piece-D
  sweep unbuilt.)
  [v0.4 ANSWERED-IN-PART (§9.2): width scales aim WHEN refit at the
  operating point — piece-D64 refit 20->61/566 (p~0.0002 twice) beats
  piece-D32 fit 33/566 on the same corpus/probe/ends-family; aim LEVEL
  rises with (k, D) jointly (k=64 aim ~0.106 vs k=32 ~0.058/~0.034).
  Word-D64's park stands scoped to word-body (piece refit succeeds where
  word refit stalled). Spectra still gate the optimum SHAPE (knees,
  plateaus); width raises the LEVEL. Full D-dependence curve (D16/D32/D64
  aim at own optima) unbuilt — one rung exists (D32 33, D64 61).]
- O5. Norm-map origin: WHY does SVD concentrate frequency
  into row norms (0.776)? Derivable from Zipf + log1p?
  Unproven (would upgrade P5 from measured to derived).
  [v0.2 PARTIAL CLOSURE (synthetic, host-side, seconds; §7.8):
  Zipf+log1p rank-32 SVD reproduces concentration (r=0.907 vs real
  0.778); uniform+log1p gives r=0.24 (Zipf necessary); raw counts
  without log1p give r=0.75 with max/mean 192x (log1p tames scale,
  doesn't create correlation). REMAINS: quantitative gap (synthetic
  max/mean 4.6x vs real 13.9x — real burstiness beyond
  independent-Zipf) + rank-k truncation curve. O5 promoted
  OPEN -> MECHANISM-IDENTIFIED / QUANTITY-OPEN.]

## 6. Scope

Applies to: count-built SVD ends, causal integer
transformers, twin/argmax instruments as defined. Does NOT
claim: gradient-trained embeddings (different formation!),
non-causal/bidirectional readout, or frontier-scale dynamics
 (single-domain evidence: 0.5B-assay scale + own tiny stacks).
Related work to reconcile (not yet done): representation
degeneration/anisotropy (adjacent: static observation vs our
tradeoff law), neural collapse (ETF structure vs our
giant/tail split), RMT spectral rigidity (possible source
for P4's stiffness -- zeta-adjacent thread, speculative).

## 7. Adversarial audit changelog (v0.2, 2026-10-03)

Host-side preferred; every listing number below re-ran the frozen
probe end-to-end (same 10 sents, seed-0 shuffle, WIN=16, piece probe
N=566 unless stated; twin = H-last relative distance on
"alexander founded alexandria" vs "alexandria was founded by
alexander"). Baselines reproduced before any harness was trusted:
fit twin 0.6654 (pin 0.665), fit top-1 33/566 = 0.0583, 95% CI
[0.0418, 0.0807] (pin 33/566 exact); transfer twin 1.0483 via
programs/lm_d32.asm (pin 1.048; lm_d32_headt.asm gives 1.0016 —
listing is load-bearing fixture), transfer top-1 28/566 = 0.0495,
95% CI [0.0344, 0.0706] (pin 0.049). Flattening helper (ends-only,
sign-corrected SVD: tau=1 reproduces bytes to 1e-13) + bank rule
(rebuild evb/ukt from flattened ends with OWN words — orig-bank
twins wrong in all three stacks) are the two methods findings that
make every number below reproducible. 95% CIs are Wilson; paired
tests are McNemar with continuity correction.

### 7.1 Norm-map fixture (P1/P5 primitives)

Groki piece32 ends + piece_bigrams counts: corr(log1p-freq,
||W[:,i]||) = 0.778 (claim 0.776 — reproduces under log1p; plain
log r = 0.748 does NOT). Frequent 12x: top-1 row 12.4x mean, max
13.9x mean, top-10 mean 10.7x, top-64 mean 5.5x (claim holds for
top/max rows, not for top-64 mean). Frequent split = bank words
(row-sum top-64): base fit 33 = 32/178 + 1/388 reproduces exactly.
Status: numbers kept WITH fixture; bare-constant phrasing retired.

### 7.2 Postulate ranking (evidential strength, strongest first)

P3 (threshold gate) > P4-directional > P1 > P2-split > P5-weak.
P3 untouched (5x argmax-freeze + continuous movement is the only
replicated dissociation; not re-run this turn). P5 ranked weakest
(missing recipe + policy flavored) and was attacked — see §7.6.

### 7.3 P1 re-measurement (groki 4.6->1.8, tau 0.3829)

Rebuilt-bank: twin 0.6126 (claim 0.614 — reproduces); top-1 26/566
= 0.0459 [0.0315, 0.0665] = 25/178 + 1/388 (claim 32->24/total 25 —
off by 1; orig-bank variant 24/566 = 23/178 + 1/388, also off by 1;
twin with orig-bank 0.5778, wrong). Rare 1->1 holds under both
fixtures. Discord base-vs-flat 12/5, McNemar 2.12, p~0.15 n.s.
Verdict: direction holds, exact pin FALSIFIED, inferential weight
low — the giants leg of P1 rests on the correlation (strong), not on
this flattening point.

### 7.4 P2 split (the conflation)

Claim "w103 flatten 12.9->4.6 ... 0.011->0.034 AND rare 0->1" never
ran as one experiment. (a) w20k flatten 12.9->4.6 (tau 0.596,
own-bank): twin 0.4072->0.6052 (pins 0.407/0.604); top-1 18->23/566
(0.0318->0.0406, CIs [0.0202,0.0497]/[0.0272,0.0602]), rare 0->1
(18/178+0/388 -> 22/178+1/388); discord 5/10, McNemar 1.07, p~0.30
n.s. (b) w103full flatten 23.2->4.6 (tau 0.4855, own-bank): twin
0.2920->0.5201 (pins 0.292/0.520); top-1 6->19/566
(0.0106->0.0336, CIs [0.0049,0.0229]/[0.0216,0.0518]), groki-split
rare 0->0 (6/178+0/388 -> 19/178+0/388); discord 5/18, McNemar 6.26,
p~0.012 (the only significant flattening gain on record). Tail
mechanism survives as two rows; single-row claim FALSIFIED.

### 7.5 P4 fixtures + paired tests

Transfer = lm_d32.asm + word-D32 body + own bank; all refits/flattens
= lm_d32_headt.asm + fitted body + own (rebuilt where flattened)
bank. Refit 28->33/566 discord 2/7 McNemar 1.78 p~0.18 n.s. (twin
1.0483->0.6654). Headt sub-step intermediate 0.735 manifest-only
(no frozen intermediate body — not live-reproduced; word-headt
0.743->0.700/top1-holds cited from headt_manifest.json). Directional
sort 4/4 holds; paired-significant 1/4. twin#=C+S stays POSTULATED
(no C/S values, no curve — O2 untouched). Negative results kept:
all discord counts above are the full write-ups (no hidden runs).

### 7.6 Decisive probe: P5-strong dies on scale (falsification)

Question: is whitening destruction the norm MAP or mere SCALE?
Scale-collapsed whiten (sv->1, rebuilt bank): 0/566 = 0.0000
[0.0, 0.0067] (destroys). Scale-preserved whiten (sv->mean(sv),
rebuilt bank, norms exactly equal, scale held): 25/566 = 0.0442
[0.0301, 0.0644] vs base 33/566; discord 17/9, McNemar 1.88, p~0.17
n.s. Equalizing the map at constant scale does NOT significantly
destroy; collapsing scale does. The "norms ARE the base rate"
inference is confounded by operating-point scale (same confound as
w20k 2.7x end norms moving the transfer point). P5-strong
FALSIFIED; P5-weak (correlation) survives; policy demoted to
CONJECTURE. Compute: 3 top-1 runs (~20 s each, host-side
rebuilt-bank recipe), bounded <15 min. Falsification counted double
per the brief.

### 7.7 R1 run: MISS (sealed prediction tested, no spin)

BPE-500 (V=538, nat spec 6.469, tau 0.8174 = sealed ~0.81; probe
N=832 under BPE-500 tokenization): transfer twin 0.3589 (P3 pin
0.359 reproduced); flat twin 0.3964 (RISES — direction hit — but
misses ~0.5 by 0.104; orig-bank variant 0.3542 goes the wrong way,
confirming rebuilt fixture). Top-1 58->59/832 (0.0697->0.0709),
discord 7/8, McNemar 0.0, p~1.0, CIs [0.0543,0.0891] vs
[0.0554,0.0904] fully overlap — NO RISE. Joint "both up" MISS.
Reading: effect size scales with distance from optimum (w103full
+13, w20k +5, BPE-500 +1) — C-gains vanish near optimum. P4-universal
dies; scoped P4' (gains scale with distance-to-optimum) succeeds it.
R2 (±0.019 noise band) validated as the probe's resolution limit:
all sub-significant moves above sit inside or near it.

### 7.8 O5 partial closure (most tractable open derivation)

Synthetic Zipf(α≈1.3, V=538) bigram counts -> log1p -> rank-32 SVD:
r(log1p-freq, col-norm) = 0.907 (real 0.778 — mechanism reproduced,
overshoots). Controls: uniform+log1p r = 0.24 (Zipf necessary);
raw counts without log1p r = 0.75 but max/mean 192x (log1p tames
scale, doesn't create correlation). REMAINS: quantity gap
(synthetic max/mean 4.6x vs real 13.9x — real burstiness beyond
independent-Zipf: top200-mass 0.54-0.66, density effects per
w103full 21%) + rank-k truncation curve + closed-form norm(f).
O5: OPEN -> MECHANISM-IDENTIFIED / QUANTITY-OPEN. O1–O4 untouched.

### Verdict (adversarial)

Survived: P3 (strongest, untouched); P4-directional (fixture-pinned
4/4 sort, now with paired tests); P1-directional + norm correlation
(0.778); P2-split rows (mechanism, two rows); D1/D2 (D2's routed
33->46 McNemar 9.6 p~0.002 untouched — still the only joint win).
Died: P2-single-sentence (conflation); P5-strong (scale confound);
P4-universal (R1 miss); R1 joint prediction (miss, no spin); exact
pins groki-frequent 24 (measured 25/23) and whitening 7/566 (recipe
missing; measured 0 or 25 by variant). Most load-bearing next
measurement: spec sweep at FIXED fit (O2/C(P) curve) — w103full tau
ladder 1.0/0.8/0.6/0.4855/0.38/0.25 + BPE-500 tau 1.0/0.8174/0.6,
twin-only first (2 runs/point, minutes; <15 min total), top-1 only at
curve knees. Decides P4' scope (where C-gains vanish) and prices R3
(half-natural bound) in one sweep — the single measurement every
open thread now depends on.

## 8. Sweep adjudication changelog (v0.3, 2026-10-03)

Instance-prescribed sweep (v0.2 §7 verdict) run by host, logged in
VELOCITY.md 2026-10-03 ("three regimes, joint table"); adjudicated
here by re-running every leg from frozen artifacts under v0.2
fixtures (fit body + own-words rebuilt bank per rung,
programs/lm_d32_headt.asm, H4 twin pair, 566-probe WIN=16 / BPE-500
832-probe). Twin values are bytes-deterministic (instrument floor
±0.0001 per the origin-leak 4th-decimal finding); aim carries Wilson
95% CIs + McNemar paired tests.

### 8.1 Ladder RATIFIED (first C(P) curve)

w103full tau 1.0/0.8/0.6/0.4855/0.38/0.25 → spec
23.18/12.36/6.59/4.60/3.30/2.19 → twin
0.2920/0.4146/0.4997/0.5200/0.5275/0.5284 vs logged
0.292/0.415/0.500/0.520/0.528/0.528 — EXACT, all six rungs.
Monotonic rise then plateau ~0.53, well below groki's 0.665 at the
same spec: the C(P) curve is corpus-indexed, converging to a content
floor (D4). Aim along the ladder: 6(raw) → 17/566 = 0.0300
[0.0188,0.0476] (6.59) → 19/566 = 0.0336 [0.0216,0.0518] (4.60,
knee) → 15/566 = 0.0265 [0.0161,0.0433] (3.30) → 14/566 = 0.0247
[0.0148,0.0411] (2.19) — inverted-U, peak at the 4.60 rung.
Paired: knee-vs-plateau-last discord 6/1, McNemar 2.29, p~0.13 n.s.;
supra-vs-knee 0/2, McNemar 0.5, p~0.48 n.s. — the 566-probe cannot
resolve 2–5-count steps (R2 band validated a third time); the knee
rests on the deterministic twin-saturation break (slope collapses
two orders of magnitude at the 4.60 rung) + directional aim peak +
four-corpus corroboration (§8.4), not on any single paired test.

### 8.2 R3: mechanism CONFIRMED, bound REVISED (R3')

Plateau past the knee: aim 19->14/566 while twins move only
0.5200->0.5284 (hold, +0.008 — above the ±0.0001 instrument floor,
so real, but negligible beside the aim drop: no MEANINGFUL twin
benefit). Over-flattening costs aim while twins saturate — the R3
mechanism, priced. But the v0.1/v0.2 bound (~half natural = 11.6 for
w103full) is FALSIFIED: the 11.6->4.6 interval contains twin gains
(0.415->0.520) AND aim gains (→19) — the bound forbids gains where
gains materialize. Successor R3': the cost region begins below
ABSOLUTE spec ~4.6 (k=32; knee rung 4.60, optimum interval (3.3,6.6]
at ladder resolution). Supporting rows below 4.6: groki 1.8
(33->26, v0.2) and BPE-500 tau-0.4 (spec 2.11, 58->38/832 = 0.0697
->0.0457 [0.0335,0.0621], discord 30/10, McNemar 9.03, p~0.003;
vs R1-flat 26/5, McNemar 12.9, p~0.0003 — the collapse aim leg is
SIGNIFICANT, the strongest aim movement in the flattening family).

### 8.3 Knee ≈ 4.6, not 4.7 (rung resolution)

Logged "~4.7" is rounding: the knee rung is spec 4.60 (tau 0.4855),
identical to groki natural 4.64 within rung resolution. Claimed
precision is ladder-limited — optimum interval (3.3, 6.6] — not a
point estimate. Coincidence with groki natural is exact at available
resolution, which is precisely what narrows O1 (§8.4).

### 8.4 O1 narrowed: corpus-Zipf ruled out as the setter

Knee/operating optimum at spec ~4.6 across groki (natural 4.64),
w20k (flattened to 4.6, gains), w103full (knee rung 4.60),
BPE-500 (4.6 target, flat gradient at optimum per the R1 miss) —
corpora differing in Zipf skew, density (5% vs 21%), V (538 vs
2038), and tokenization. A corpus-set optimum would move with the
corpus; it does not move. Geometry (k=32 rank? V-scaling?) remains;
k-sweep is now the load-bearing unbuilt (O1/O2 joint).

### 8.5 HELD, not ratified: twin≈0 bit-identical collapse

Logged BPE-500 tau-0.4/0.25 twins 0.001/0.085 do NOT reproduce.
Fixtures tried (all with fit body, tau 0.4 spec 2.11 / tau 0.25 spec
1.59): rebuilt-bank 0.5032/0.5386; orig-bank 0.3821/0.3997; H2-read
0.6955; transfer-body+plain-listing 0.7702/0.4983 — none within 0.4
of logged. Twin contexts verified non-degenerate (5 vs 7 BPE-500
ids). The AIM leg of the same runs reproduces exactly (38/832 =
0.0457 vs logged 0.046), so the failure is isolated to the twin leg
— prime suspect is a twin-leg/aim-leg fixture divergence in the sweep
script (same mixed-fixture species as v0.2 §7.3). The "degenerate
indistinguishability" reading is therefore HELD as CONJECTURE, not
MEASURED — with a predicted signature (tau->0 limit: twin->0 +
aim->tiebreak floor) queued as a cheap twin-only probe, and a
recovery recipe for the runner: re-run the twin leg printing
listing/body/bank/keys, assert distinct non-empty context ids, and
confirm both legs share one config. The joint table (D5) does NOT
depend on this point — its collapse cell is already doubly filled
by groki-overflatten and w103full-plateau with reproduced twins —
and the collapse AIM leg (p~0.003) stands as the significant row.
Negative result kept in full per the rules; no silent rewrite.

### Verdict (v0.3)

Survived: P3; P4-directional (now with first C(P) curve); P1/P2-split
rows; D1/D2; R3-mechanism (bound revised, stronger: absolute beats
relative); O1/O2 (narrowed + first curve).
Died or demoted: R3-half-bound (FALSIFIED as universal → R3');
D3-convergence (FALSIFIED → D4 floors); twin-only interpretation
(period — D5 joint-or-nothing); twin≈0 as MEASURED (HELD pending
fixture recovery, §8.5).
Born: D4 (content floors), D5 (joint or nothing), R4 (empty up/down
cell — the falsifier the table owes the theory), R3' (absolute
knee), collapse-floor τ→0 probe (queued).
Single most load-bearing next measurement: rank-k sweep at fixed
corpus+fit (k = 16/32/64 ends from one counts matrix, twin ladder +
knee per k) — decides whether the 4.6 knee is k-set (O1 closure
candidate) or V-set, and prices the k-dependence the whole
absolute-knee claim now depends on. Twin-only first (<15 min), top-1
at knees only.
(RUN 2026-10-03 as the D64 mission — adjudicated in §9 below.)

## 9. D64 mission changelog (v0.4, 2026-10-03)

Commander orders + 10 ruled amendments (Step-0 freeze, in-stack band,
parity gate, slice verify, V parked, no closure-by-parking, Stage −1,
budget instrumentation, kill bounties, v0.4 numbering). Refit battery
~15×566-probes + screens, inside the ~1hr box. Every top-1 below ran
the frozen 566-probe (WIN=16, seed-0 split); every twin is the H-last
pair; Wilson CIs + McNemar paired throughout; parity bar ≥40dB both
keys or the number is HELD, not filed.

### 9.1 Step 0 + transfer band (MEASURED)

`scripts/freeze_piece64.py`: groki BPE-2000 counts → log1p SVD rank-64
(`data/lm_piece64.npz`, natural spec64 **5.81**) + row-sum top-64 bank
(`data/bankpiece64_d64.npz`, same words as bankp32_64 — recipe
consistent) + manifest. Word-D64-fit recompute (g2 = ×2 on
wq/wk/wv/wo/wup/wgate/wdown + wq=wk=I*0.2) verified on the word probe
first: g2 twin 1.1090 (pin 1.109), QK-I0.2 twin 0.9556 (pin 0.956) —
recipe RATIFIED before porting. Transfer (piece64 ends + word-fit body
+ piece64 bank, `lm_d64.asm`): twin **0.7329**, top-1 **20/566** =
0.0353 [0.0230, 0.0539] (33s/probe at D64), parity 50.82/52.56dB GREEN.
BAND filed. (The orders' "1.048" was a D32 number — measured, not
imported, per amendment 3.)

### 9.2 Refit WINS — no parking (MEASURED, twice significant)

V-family: V0.5 twin 1.2004, V1.0 0.8836/18/566 (up/down, McNemar 0.04
p~0.84 noise — REJECTED), V1.5 0.8610 — V-without-O cannot save
scramble (word precedent holds). O-clean (wv=I1.0, wo=I0.5): twin
0.7329->**1.1900** + top-1 20->**40/566**, McNemar 13.88, p~0.0002,
parity 52.61/54.56dB GREEN — JOINT WIN (strongest aim movement on
record; beats routed v1's p~0.002). Mechanism: random 2*seed wo
SCRAMBLES attended content (word-D64 entropy diagnosis confirmed);
clean O unscrambles — and twin RISES because honest content divergence
(1.19) exceeds scramble-mush fake-invariance (0.73). QK-dial on O-clean
(headt listing): twins 0.6949/0.9287/1.1654/1.4113/1.4381/1.4491/
1.4707/1.4975/1.5215/1.5626 across s=0.1–1.0, aim 31/40/48/54/57/60/61
— monotonic C-dial (endpoints significant: O-clean-vs-QK-1.0 discord
4/25, McNemar 13.79, p~0.0002; steps 40->48 p~0.043, 48->54 p~0.11,
57->60 p~0.37, all parity-green 51–55dB). Fit stands at QK-1.0
(**1.5626/61/566**, natural unit scale — principled stop): frozen
`data/lm_piece64_fit.npz` + manifest. Headt: twin 1.1900->1.1654 with
ZERO decision flips (discord 0/0) — LATERAL, kept (playbook shape,
10× smaller effect than D32). Honors-key verified (1.1654 vs 1.1900).
Word-D64's park narrows to word-body: width per se is fittable when
refit at the operating point. O4 answered-in-part (aim level rises
with width: D64 61 > D32 33 same corpus/probe).

### 9.3 P4-universal DEAD, second kill (falsification)

O-clean (a fit at fixed spectra) moved twins UP with aim UP
significantly — "fits move S (twins down)" is falsified as universal.
D32 QK re-examined for the path-sort: 28->32/566 (discord 2/6,
McNemar 1.12, p~0.29 n.s.) with twin down = S-pattern HOLDS for QK.
Same-path (V/O-clean) sorts S at D32, C at D64 → the differentiator is
attention-regime (healthy ~0.6 vs collapsed ~0.15), not path.
Successor: regime-indexed C/S (R5, CONJECTURE with entropy anchors).
Historical rows stand; D5 (joint cells, no directionality by move-type)
is now the central law and never claimed what died.

### 9.4 k=64 ladder + knee verdict (MEASURED, fully gated flatten-side)

Fit-body ladder (headt listing): tau 1.5/1.2/1.0/0.8/0.6/0.4855/0.38/
0.25 → spec 14.00/8.26/5.81/4.09/2.87/2.35/1.95/1.55 → twin 1.4139/
1.5308/1.5626/1.5705/1.5674/1.5613/1.5541/1.5409. Twin-peak ≈4.1
(interval (2.87,5.81], overlaps k=32-groki peak interval (4.6,10.0] at
(4.6,5.81] — a common ~5 NOT ruled out for twins). Aim: 61/61/59/59
across 5.81/4.09/2.87/1.55 — FLAT (pairs p 0.75–0.84, pure noise). NO
aim-knee anywhere gated. Parity: flatten-side 51–64dB GREEN all rungs
(first fully-gated ladder — new standard); sharpen-side RED (-0.68dB
at tau=1.5, re-price 60000 makes it WORSE at -12.81dB: per-product
fold, m-independent — PARKED with L1-range-law diagnosis; needs
per-block vehicle or wide-path. The 8.26/14.0 twin rungs are HELD.)
Knee verdict: k-set-as-single-number FALSIFIED (k=64 aim-kneeless
where k=32 knees); V-set parked (frozen V); k-vs-fit confounded by
design (each k refit at own point); seed-aim unmeasured. O1 CLOSED
as multivariate-scoped (O6 + V-set + seed-aim succeed it).

### 9.5 R3' + R4 + O3 status

R3' survives: every below-4.6 aim move points down or flat-noise
(groki-flat, w103-plateau, BPE-collapse p~0.003, k64-1.55 59-vs-61
noise). R4 letter holds (6.8 threat-row within band, p~0.08);
directional occupation stands declared. O3 CLOSED (S): temps move
twins-only in both widths, cleanest demonstration on record
(D64-headt discord 0/0); P3 explains the shifted-mass half. No residue.

### 9.6 Stage −1 paired numbers (replacing unsaved hits)

Groki down-slope 33->27->25: nat-vs-10.0 discord 12/4, McNemar 3.06,
p~0.08; 6.8-vs-10.0 discord 7/5, McNemar 0.08, p~0.77. Slope is the
finding (directionally clean 3/3 rungs), no rung is individually
significant — as briefed. D32-6.8 joint point retro-gated (47/50dB);
D32-10.0 TWIN gated (47.73dB) but its AIM leg rides RED LOGITS headroom
(1.36dB, refmax 96 — HELD; slope inference loses nothing: 6.8-vs-10.0
was noise regardless). BPE-500 rows stamped 37dB (below bar; re-price
attempts collapse to negative dB — global-m optimum, structural;
verdicts robust: nulls need no precision, 20-ct systematic moves exceed
noise scale).

### 9.7 Methods finds (keep)

Slices in `lm_d64_headt.asm` verified Dh=32-correct in-file (only the
header comment lies: Dh=8/V=513); TBETA = full-tensor beta_b scalar,
head2 scores ×0.25 pre-softmax (mirror-verified to 68dB at SC2);
tempering is LAYER-1-ONLY by listing design (mirror over-tempered layer
2 until bisection caught it — new bug species logged: listing-width/
layer-asymmetric temper vs mirror assumption); MLP weights structurally
dead in all fit listings (IN-declared, never read — g2-on-MLP is inert
by construction); per-key feeds (SC/P/C/H/H2/LOGITS all surfaced) make
mirror bisection seconds-cheap; BETA mask value is -30.0 not -1e9
(verified in-situ, harmless post-TSHIFT); bank mass K64 holds at D64;
`/tmp` host scratch identified as 2026-10-01 word-D64 battery (stale,
no collision — and its word joint (V/O-I0.5+QK0.2) corroborates recipe
while contrasting outcome: word-parked vs piece-wins).

### Verdict (v0.4)

Survived: P3; D1/D2 (routed still the only routed win; O-clean is the
only flat-stack significant win — two organs, two winners); R3'
(cost-region, third corroboration wave); R4-letter; D4 (extended);
P2-split rows; norm correlation (untouched).
Died: P4-universal (O-clean up/up p~0.0002 — clean kill); universal
aim-knee ~4.6 (k=64 flat across it, gated); k-set-as-single-number;
O1-as-posed (closed-multivariate); O3-as-open (closed-S); V-without-O
(battery class); QK-0.2-at-D64 (dial continues to 1.0: 61 beats 40).
Held: twin≈0 (unchanged, §8.5); k64 sharpen-side twins (red parity,
per-product-fold diagnosis); BPE-500 rows at 37dB (verdicts robust,
grade stamped); seed-aim (unmeasured); V-set; R5 (conjecture).
Born: R5 (regime-indexed C/S); O6 (k-vs-fit separation); D5-central;
k64-fit stack (1.5626/61, frozen); Dh-parametrized mirror; per-rung
parity standard.
Single most load-bearing next measurement: seed-aim legs at k=64 —
reconstruct unit-gaussian seed-0 bodies per the host recipe (tied
layers) and run top-1 at ladder specs 5.81/4.09/2.87 (3×33s; twin band
1.6–1.9 already logged). Flat seed-aim vs structured fit-aim decides
fit-set (knee needs a fitted body) against k-set (structured seed-aim
with knee≠4.6); either outcome closes an O1-remainder, and R5's regime
anchors plus D4's floors both depend on whether unfitted geometry
already carries any knee at all.

## 10. Seed-aim adjudication + O1 closure changelog (v0.5, 2026-10-03)

The §9-verdict measurement ran (3 seed-aim legs + 2 groki
below-natural legs + 6 knee-matrix legs, all on the frozen 566-probe;
preds in /tmp/d64_sseed_*.npz, /tmp/groki32_tau*.npz, §9 legs in
/tmp/d64_*.npz — host scratch, hashes in §9.7-note below).

### 10.1 Seed-aim legs at k=64 (MEASURED, gated)

Unit-seed (host recipe 1.0x, tied layers): twin 1.6663, top-1 5/566,
parity −41dB RED → HELD, not filed (the seed-0 recipe under-scales at
D64 — unit-gaussian weights too hot for Dh=32 products; methods find,
not a result). Small-seed (0.1x, parity GREEN 64–69dB both keys):
spec 5.81/4.09/2.87 → twin 1.1690/1.2181/1.2118, top-1 30/35/38-566
([0.0374,0.0747]/[0.0448,0.0848]/[0.0493,0.0908]) — monotonic rise as
spec falls, NO knee (CIs overlap throughout). Unfitted geometry carries
no knee at k=64.

McNemar paired (decisive, all three): smallseed-2.87 (38) vs
FIT-QK-1.0 (61): discord 5/28, chi2 14.67, p~0.0001 — FIT WINS
(fit-required for aim LEVEL, strongest paired gap on record).
Smallseed (38) vs transfer (20): discord 26/8, chi2 8.50, p~0.0036 —
small random BEATS structured transfer: NEGATIVE transfer confirmed
(word-D64 structures actively HURT piece-D64; operating-point
magnitude mismatch. The in-stack band (§9.1) is reinterpreted: the
recipe transports, the content does not). FIT vs transfer 61-vs-20,
chi2 31.37 (sanity, matches §9 pattern).

### 10.2 Knee matrix — below-natural + seeds (MEASURED)

Groki k32-fit below-natural (the missing half of R3's curve): tau
0.6/0.4 (spec 2.51/1.85) → top-1 25/26-566 ([0.0301,0.0644]/
[0.0315,0.0665]). Full k32-fit curve: 26/25 → 33 (natural 4.64) →
27/25 (6.8/10.0): INVERTED-U directionally complete — both sides fall.
CIs overlap (paired arrays unsaved; rung significance UNPINNED —
follow-up: rerun the five k32 legs with pair-save + McNemar; slope is
the finding, as in §9.6).
k16-seed: 36/33/21 across spec 1.7/3.87/7.6 — monotonic DECLINE, no
knee (flattest bests). k32-seed: 37/32/22 across 1.85/4.64/10.0 —
monotonic DECLINE, no knee. No unfitted body knees at ANY k (three
seed bodies, two ks, zero knees).

Matrix state: knee present ONLY at (k32, fit); absent at (k64,
fit — flat 59–61), (k64, seed), (k32, seed), (k16, seed).
k16-fit-aim unmeasured → PREREGISTERED PREDICTION (falsifiable):
a k16-fit tau sweep shows an inverted-U peaking ~3.9 (its natural
spec). If it peaks elsewhere or not at all, the fit-point law as
stated dies and k-indexing returns.

### 10.3 O1 verdict (CLOSED) + O6 with it

Knee ⟺ fitted body near its natural spec (fit-point law, EMPIRICAL
with one preregistered prediction outstanding). Fit is NECESSARY for
knee-presence (zero knees in three unfitted bodies) AND for aim level
(38-vs-61, p~0.0001). k indexes SHARPNESS (sharp at k32, flat at k64),
not existence — the multivariate remainder goes to V-set (parked),
not to a new open. O1-as-posed CLOSED; O6 (k-vs-fit separation, whose
content was exactly these legs) CLOSED with it. R5 strengthened but
still conjecture: knees are fit-point phenomena (healthy fitted
regimes knee; seed/transfer regimes don't); k64-fit flat reads as the
saturation half of R5.

### Verdict (v0.5)

Survived: P3; D1/D2; R3' (fourth wave: below-natural falls 2/2);
R4-letter; D4 (extended + negative-transfer floor); D5-central; R5
(strengthened); fit-point law (empirical); k16-fit prediction
(preregistered, unfalsified).
Died: O1-as-open (closed fit-point); O6-as-open (closed with it);
"seed bodies carry knees" (0/3); transfer-band-as-content (recipe
transports, content mismatches, p~0.0036); unit-seed recipe at D64
(parity −41dB, under-scaled — methods, not theory).
Held: twin≈0 (§8.5); k64 sharpen-side twins; BPE-500 rows at 37dB;
V-set (parked); R5-conjecture status; k32-rung paired significance
(preds unsaved, rerun prescribed).
Born: negative-transfer law (word→piece structures hurt);
fit-point knee law; k16-fit inverted-U prediction; 0.1x-seed parity
band (64–69dB — the valid unfitted-geometry vehicle at D64).
/tmp scratch note (§9.7 addendum): d64_sseed_*.npz (30/35/38),
d64_seed_*.npz (RED, held), groki32_tau2_51/tau1_85.npz (25/26),
d64_QK1.0.npz (61) — the paired arrays behind every §10 number.
