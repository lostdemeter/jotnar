# Composition: depth + temporal (step 3) — spec and status

Pipelines compose codecs the way ffmpeg composes filters (phi_core
AI_CODEC_PIPELINE.md): named triple streams, spec strings, runtime toggles,
L2 triple-direct handoffs (no decode/re-encode at seams). This doc scopes
holo-phi's two compositions. Neither is built here; both are specified so
the wiring, not the math, is the remaining work.

## Depth-aware enhancement (DAV2 codec) -- L1 BUILT this round

Rule: `beta_eff *= near ? 1.0 : 0.5` (median split on per-frame-normalized
relative depth; DAV2 is scale-ambiguous so only order is consumed).
`chain/depthprior.py` computes the prior once (torch, offline doctrine),
caches `samples/depth/<tag>.npy` with an input-hash sidecar (collision
raises, never silently reuses). Hot path: median mask at encode +
`select_mux` + `tmul` (integers only). Multiplies onto the controller field
(modulations compose). Toggle: `--depth TAG` (demo), `depth=array`
(chain). Gates: `test_depth.py` -- cache determinism, parity 45.68dB,
near p99 0.063 > far 0.017 (3.6x: foreground person vs background),
toggle-clean. L2 triple-direct handoff stays backlog (geo_int.py already
speaks triples; the export path isn't wired).

## Temporal coherence (feedback #2): IIR on the detail field [BUILT this round]

The first recurrent edge: `D_t = 0.5*D + 0.5*warp(D_{t-1})` (chain/temporal.py).
Warp is nihui-form (unclamped-floor alphas, clamped indices, replicate),
ported to integer fixed-point (flow float pixels -> 2^-14 at the boundary;
weights 2^-28, floor sums matching rife torch >>28). One deliberate
convention split, documented: OUR seam is (dy,dx) while rife's warp_fixed
reads [...,0] as X -- the cross-check gate (test_substitute pattern) would
catch any confusion, and warp-parity (73dB) confirms the mapping.
Mix weight frozen dyadic 0.5 (an add + trunc-halve; non-dyadic a is a
LOWERING ERROR per the interp doctrine -- no fixed-mult mix op exists).
State carries triples; first frame == still output bit-exactly.
Gates (test_temporal.py, 10 checks): mix-frozen, warp-parity 73dB,
warp-identity (fixed counts), static-converged (frames 1+ mutually exact) +
static-firststep (measured bounds max Δe<=3, <=8px -- the mix roundtrip's
honest quantization cost, calibrated over 6 seeds, not shimmer),
firstframe-still exact, memory-carries, step-settles geometrically,
sequence parity 64-65dB on a translating bar with exact flow.
Demo: demo_temporal.py (PNG sequence + uniform flow; selftest GO 58dB).
Per-pixel RIFE flows + L2 depth-direct stay backlog (same producer pattern
as motion Phase 2).

## What ships now

`holo=on:splat=on:ctrl=on` on stills, all gates green. Steps 1+2 are the
intelligence that makes step 3 worth building: depth and temporal modulate
the same beta field the controller already owns.
