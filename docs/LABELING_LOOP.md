# Labeling loop (draft v0.1): from fingerprints to knowledge

Problem: readout fingerprints are positional (direction d moves position t
on prompt P). Knowledge is a claim that survives repositioning: "d carries
X" must predict behavior on prompts where X sits elsewhere. This loop
closes that gap with the primitives we already gate. Status: DESIGN (no
loop has been run; the shelf map + implant are its first two steps built).

## The loop (4 steps, each gated)

1. **Hypothesize.** From ≥2 readouts: direction d moves tokens sharing a
   property P (e.g. proper nouns, verbs, position-0). Output: (d, P) pair
   + the fingerprints as evidence. Gate: P stated BEFORE step 3.
2. **Match.** Fresh prompt with P at new positions: d's fingerprint must
   follow P, not the positions. Gate: movers(d's rank on P-tokens) above
   chance with a stated bar (top-k overlap, k stated).
3. **Verify by implant.** The implant is the label-verifier: silence d
   (ablate) or amplify d and PREDICT the behavioral delta from the label
   ("France-tokens degrade ~XdB, others <YdB"). Run test_implant-style.
   Gate: predicted band holds on held-out tokens.
4. **Close.** Verified (d, P) enters INVENTORY as SINGLE with the three
   gates as evidence; a second model family promotes to CONFIRMED.
   Labels that predict interventions are knowledge; the rest are stories.

## Why this loop can work here (and mostly doesn't elsewhere)

- Fingerprints are cheap (116s/112 dirs) and exact-language (dB, not
  cosine vibes): steps 1-2 cost minutes.
- The verifier already exists and is gated (implant gaps 26-35dB):
  step 3 reuses test_implant.py's machinery with label-derived aims
  instead of MID-aligned ones.
- Failure is informative at every step (wrong P, unstable fingerprint,
  missed band) -- falsification routes, not dead ends.

## Open (stated, not missing)

Property vocabulary P (what CAN a direction carry? nouns? relations?
positions? -- the first hypotheses will be crude); multi-layer labels
(one matrix read at a time so far); the match bar for step 2 (top-k
overlap? rank correlation? -- first loop run decides, then standardize).
