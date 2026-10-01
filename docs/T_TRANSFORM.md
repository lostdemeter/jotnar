# T-transform spec (v1.4 gate 4): full-range attention, designed not wished

## Obstruction (measured, all of it)

1. The exp path already does max-subtraction (`softmaxN_fixed`: vmax-shift,
   clip at 16.0 in 2^-14, exact zeros beyond -- correct to below quantum).
2. The wall is the input bridge ONLY: `softmaxN_triples` hardcodes
   `to_fixed @ BIAS` (U=1.0), and out-of-range values FOLD to +/-1.0
   (d<0 -> s*base), never saturate. Verified by reading, consistent with
   every gate (in-contract passes, out-of-contract pins to softmax-of-
   clipped). No max-sub on our side can fix it: shifted rows still span
   [-range, 0] with range >> 1 (table below).
3. Real per-row ranges (DDColor cross-attn, f_014, folded scores):

| layer | max | range p50 | range p99 |
|---|---|---|---|
| 0 | 7.0 | 3.3 | 7.4 |
| 1 | 12.2 | 10.8 | 21.0 |
| 2 | 9.7 | 6.0 | 14.1 |
| 3 | 8.5 | 9.4 | 26.4 |
| 4 | 20.6 | 14.4 | 47.4 |
| 5 | 32.3 | 53.6 | 64.2 |
| 6 | 107.5 | 126.8 | 169.6 |
| 7 | 428.6 | 251.1 | 614.4 |
| 8 | 26.3 | 20.2 | 56.5 |

(Qwen folded: max 963 -- same treatment, bigger m.)

## The fix (two parts, split at the repo seam)

**phi-core side (one assert + pass-through, no LUT change):**
`softmaxN_fixed(q, m)` already takes m -- relax the `m == BIAS` assert to
accept covering scales (m_of(range+margin) per the table; all < 65535
cap), with the exp/DD machinery untouched (clip at 16.0 stays correct:
e^-16 is below quantum). Offered as a staged branch per shared-repo
rules (branch, claim check, no main push); owner merges.

**Our side (composition, ready now):** per-row max-sub in triples via
ARGMAX + GATHER-tiled max + SUB at covering m (CONFIG override path,
proven) feeding softmaxN at the same m. One gap named honestly: tiling
row-max to rows needs broadcast GATHER over repeated ids -- expressible
(CONCAT loop) but ugly; a TILE mnemonic would clean it (contribution
process if demanded). Gates ready: max-sub 0-diff vs manual steps,
full-range parity vs torch on +/-100 scores (bar 40dB), in-contract
regression (existing suites unchanged).

## Why this one is small (and last time wasn't obvious)

Every alternative dies measured: global-max shift breaks rows (verified
by reasoning -- softmax needs PER-ROW shift); clip-to-[-1,0] destroys mid
probs (e^-5 vs e^-1); temperature rescale changes semantics (approximation
with error bounds, not on the table); triple-native exp path = new LUT
machinery (scope explosion into phi-core internals). The bridge param is
the ONLY change that preserves all existing behavior (BIAS callers
untouched) while opening full range. Minimal diff, maximal unlock:
DDColor L0 needs only m_of(~16).
