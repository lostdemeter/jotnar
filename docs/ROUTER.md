# Conditional routing (feedback #3, first half): select AMONG modes per frame

The first structure operating on CONFIGURATIONS instead of pixels. Per frame,
engage temporal memory or run still; temporal state carries ACROSS frames
regardless (state always refreshes -- memory never goes stale).

## Rule v1 (analytic, stated)

`E = mean(|flow|) / 8`; temporal iff `E >= 0.25` (mean motion >= 2px engages).
Decided in float at the seam, consumed as an exact flag: INT and oracle can
never disagree on the mode (structural agreement, the motion-static-mask
pattern again). Flow None -> still. Threshold frozen, flagged for fitting.

## Why this shape (measured, not assumed)

Probe on translating-noisy bar: temporal flicker 0.0254 vs still 0.0317
(memory smooths) at sharpness 1.885 vs 1.998 (IIR smears ~6% -- bounded
cost). Neither mode dominates: the router's claim is per-regime selection,
gated component-wise (flicker-wins AND sharpness-bounded) so no single
number hides the tradeoff.

## Gates (test_router.py, 10 checks ALL OK)

decide-static/none/moving/edge (incl. boundary inclusivity E==thr engages) /
decisions on mixed sequence / seamless-static (3 static frames bit-identical
to still-only: still ignores state entirely) / flicker-wins / sharpness-bounded /
routed-parity 60-63dB (oracle takes the same modes -- shared boundary
decisions agree structurally).

## Scope (stated limits)

Per-FRAME routing (global decision). Per-tile routing is backlog (needs
tiling infra + per-tile state; the pattern transfers). Router arms in v1 are
{still, temporal} around fixed flagship-or-iso chains; arm EXPANSION
(depth-gated arms, beta schedules) is v2 backlog. Demo: demo_router.py.
