# VELOCITY LOG (v1.0 Gate 5: "how fast can a newcomer extend it?")

Measured, not claimed. Each entry: date, mnemonic, wall minutes
start-to-finish (op + sig + gate + docs + listing use), surprises.
The log IS the extensibility answer.

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
