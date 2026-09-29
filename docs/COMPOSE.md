# Composition: depth + temporal (step 3) — spec and status

Pipelines compose codecs the way ffmpeg composes filters (phi_core
AI_CODEC_PIPELINE.md): named triple streams, spec strings, runtime toggles,
L2 triple-direct handoffs (no decode/re-encode at seams). This doc scopes
holo-phi's two compositions. Neither is built here; both are specified so
the wiring, not the math, is the remaining work.

## Depth-aware enhancement (DAV2 codec)

Idea: splat sizes + beta scale with scene depth. Near texture (skin, fabric)
resolves fine detail; far field (sky, distance haze) is optically softer, so
boosting it only amplifies haze noise. Rule sketch: `sigma_eff =
sigma * depth_norm` (far -> wider bank = gentler), `beta_eff *= near_gain`.
Depth arrives as triples from the DAV2 codec (perception), rescaled to our
m_cov at the seam (explicit `rescale_`, unified scales where possible = L2
pointer pass). Toggle: `holo=on:depth=on`. Gate (when built): parity on the
seam (>=51dB dividend discipline per the pipeline doc) + a depth-ordering
check (far-field boost < near-field boost on a staged pair).

Status: NOT STARTED. Needs a depth-codec checkout beside this repo; the seam
contract above is the whole interface.

## Temporal coherence (RIFE-style codec / warp op)

Idea: video enhancement must not shimmer. Detail boost computed per-frame
flickers on noise. Rule sketch: warp the previous frame's amplitude detail
field to the current frame (IR `warp` op, nihui-exact, already specified)
and temporally IIR the boost: `D_t = (1-a)*D + a*warp(D_{t-1})`. All integer
(fixed add + warp + tdiv for the mix). Toggle: `holo=on:temporal=on`.
Gate (when built): static-video identity (no shimmer on frozen frames:
frame-to-frame diff of enhanced static clip ~= input diff) + parity on the
warp seam.

Status: NOT STARTED. IR `warp` exists in phi-core with lowerings; the holo
side needs a detail-field delay line + the mix op + a static-clip fixture.

## What ships now

`holo=on:splat=on:ctrl=on` on stills, all gates green. Steps 1+2 are the
intelligence that makes step 3 worth building: depth and temporal modulate
the same beta field the controller already owns.
