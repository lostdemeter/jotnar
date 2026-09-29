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
