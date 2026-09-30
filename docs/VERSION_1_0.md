# v1.0: the extensibility release (definition of done)

v1.0 is the version we put in front of strangers, defined by ONE question:
"how fast can a newcomer extend it?" Everything below serves that question.
Out of scope for v1.0 (named, not forgotten): C performance path, WHILE +
dynamic shapes, per-tile routing, rank arithmetic beyond v1 layouts.

## Acceptance gates (all must hold)

1. **Whole-block float parity.** A transformer encoder block listing holds
   >=40dB vs an independent torch reference (or a stated, measured reason
   it cannot -- but then v1.0 waits; this gate is load-bearing, not
   negotiable, because it is the only values-level proof at composition
   scale).
2. **LANGUAGE.md.** Single reference: every mnemonic with contract + example,
   the tutorial (write your first listing in 15 minutes), the contribution
   process (authoring bar + promotion rule + gate patterns to copy).
3. **Dogfooded procedures.** At least one production listing shares DEFs via
   IMPORT (no copy-pasted prologues in shipped programs) + stdlib seed dir
   with 2+ reused blocks.
4. **IMPORT search paths.** Beyond relative paths: a documented search order
   (stdlib first, then relative) so shared listings are addressable by name.
5. **Extension drill (the v1.0 signature gate).** One NEW mnemonic added
   start-to-finish (op + sig + gate + docs + listing use) with the wall time
   logged. This starts the VELOCITY LOG (docs/VELOCITY.md): date, mnemonic,
   minutes, surprises. The log IS the "how fast" answer, measured not claimed.
6. **Stranger test.** A fresh instance with ONLY the docs builds a small
   program (fizzbuzz-equivalent: 5-line listing, run, green). Friction log
   becomes v1.1 backlog. If it can't be done from docs alone, v1.0 isn't done.

## Deliberately deferred (v1.x)

C speed path for new ops, WHILE/dynamic shapes, per-tile routing, rank
arithmetic, phi-core merges (owner action), stranger-reported hardening.
