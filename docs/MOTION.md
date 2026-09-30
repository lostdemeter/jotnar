# Motion side-channel (side channels #1): flow corroborates tensor

First seam that carries *features* instead of pixels. The depth prior
modulates beta; the motion prior modulates orientation confidence AND beta.
Status: consumer + contract + gates LANDED (synthetic ground-truth flow);
live RIFE flownet producer is Phase 2 (same hashed-cache pattern as
depthprior, #LIB-013 recipe).

## Seam contract (chain/motion.py -- the whole interface)

- `flow`: float64 (H, W, 2), FORWARD displacement (frame A -> frame B) in
  PIXELS at frame-A geometry. `(dy, dx)`, RIFE warp convention.
- Boundary conversion (`quantize`, float, at the seam): angle -> perp bucket
  {0:V,1:H,2:\\,3:/} via octant table; magnitude -> norm [0,1] via /8px;
  exact-zero -> static mask. Downstream is integers only.

## Consumer rule: CONSENSUS (flow confirms/vetoes, never originates)

Flow direction alone cannot assert edge orientation (aperture problem:
panning along an edge breaks any perpendicular assumption -- gated as
pan-preserve/pan-e2e). So:
- `eff = max(coh, norm)`, gate opens where motion explains weak coherence;
- `strong = coh >= hi` (tensor's own testimony, reuses fitted coh_hi);
- `agree = (tensor_bucket == perp)` (independent witness);
- `final = gate AND (strong OR agree)`;
- beta caution composes multiplicatively: `flow_scale = 1-0.5*norm`
  (fast = smeared = restraint; mirrors depth-far).
- Zero flow reproduces blind BIT-EXACTLY (static mask takes the direct path;
  motion=None skips the branch). Direction is always assigned before gating
  (a blind-iso pixel carries no direction -- gating first would let flow only
  ever REMOVE orientation; caught pre-gate, documented).

Frozen analytic v1: FLOW_REF=8px, FLOW_ATTEN=0.5. Both flagged for fitting.

## Gates (test_motion.py, 10 checks ALL OK)

compass (14 angles, table twins agree -- a swap here is parity-invisible, so
this gate is load-bearing) / rescue-V 0.36->0.81 / rescue-no-invent 0.06 /
zeroflow-exact bytes + fixed counts / pan-preserve H 0.80->0.80 /
parity-motion 42.94dB / pan-e2e ratio 0.504 (exactly the attenuator: the only
end-to-end change is the stated caution) / rotation-zeroflow gap 0.04.

## Phase 2 (live producer, not started)

Fetch RIFE weights (network), run flownet per pair, cache under
samples/flow/ with input-hash sidecars (depthprior pattern). Producer-consumer
contract test: cached flow determinism + real-frame (f_012/f_014) demo.
RIFE flownet emits intermediate blobs (flow[:2]/flow[2:]); the adapter maps
them to forward flow at frame geometry -- that mapping gets its own gate.
