"""Motion side-channel (step: side channels #1): flow corroborates tensor.

Seam contract (the whole interface -- producer adapters map to this):
  flow: float64 (H, W, 2), FORWARD displacement (frame A -> frame B) in PIXELS,
  at frame-A geometry. (dy, dx) = flow[..., 0], flow[..., 1].
  Sign convention matches RIFE warp (sample = p + flow).

Boundary doctrine: angle quantization + magnitude normalization happen HERE,
in float, at the seam (like depthprior's median split). Everything downstream
is integers: perp bucket (int8) + norm (encoded once) feed the consensus rule.

Consumer rule -- CONSENSUS (stated reasons, aperture-safe):
  Flow direction alone can never assert edge orientation (panning along an
  edge breaks the perpendicular assumption), so flow only CONFIRMS or VETOES
  the tensor's claim; it never originates one:
    eff_coh = max(coh, motion_norm)          -- motion explains weak coherence
    gate    = eff_coh >= thr                 -- opens on moving structure
    strong  = coh >= hi                      -- tensor's own testimony
    agree   = (tensor_bucket == flow_perp)   -- independent witness
    final   = gate AND (strong OR agree)     -- either suffices; both checked
  Tensor-strong pixels behave exactly as without flow (all current behavior
  preserved). Zero flow is BIT-IDENTICAL to blind (gated exactly, not in dB).
  Beta caution composes multiplicatively (mirrors depth-far):
    flow_scale = 1 - (1-FLOW_ATTEN)*motion_norm  (fast = smeared = restraint)

Frozen analytic v1: FLOW_REF = 8px (typical inter-frame motion at CIF scale),
FLOW_ATTEN = 0.5 (same value as depth FAR_ATTEN: same caution doctrine).
Both flagged for future fitting (v6 candidates). Phase 2 wires the live
RIFE flownet producer behind get_flow() with the depthprior hashed cache.
"""
import numpy as np

FLOW_REF = 8.0
FLOW_ATTEN = 0.5

# edge direction = flow direction + 90deg, quantized to the bank:
# flow 0/180 (moving +-x) -> edge runs vertically -> V (0);
# flow +-90 -> H (1); flow 45/225 (down-right, normal (1,1), Jxy>0) -> / (3);
# flow 135/315 (down-left, normal (-1,1), Jxy<0) -> \\ (2). Table, not
# atan2-per-pixel cleverness; cross-checked vs coherence_bucket diag_pos.
PERP_TABLE = {}  # built by _perp_of below; kept explicit for audit


def _perp_of(ang):
    """Flow angle (deg, atan2(dy,dx), y DOWN) -> bank bucket for the
    perpendicular edge. Checked against coherence_bucket's diag_pos
    (Jxy>=0 -> / = 3), NOT against any mirror:
      flow 0 (moving +x, normal (1,0)) -> V (0)
      flow 90 (moving down, normal (0,1)) -> H (1)
      flow 45 (down-right, normal (1,1), Jxy>0) -> / (3)
      flow 315/-45 (up-right, normal (1,-1), Jxy<0) -> \\ (2)
    Opposite directions share the edge family (normal +-). A swap here once
    shipped in _soft_blend and parity stayed green while rotation failed --
    this table's unit test asserts all eight compass points explicitly."""
    a = ang % 360.0
    if (a < 22.5) or (a >= 337.5) or (157.5 <= a < 202.5):
        return 0  # moving +-x: edge vertical
    if (67.5 <= a < 112.5) or (247.5 <= a < 292.5):
        return 1  # moving +-y: edge horizontal
    if (22.5 <= a < 67.5) or (202.5 <= a < 247.5):
        return 3  # down-right/up-left: Jxy>0 -> slash
    return 2  # down-left/up-right: Jxy<0 -> backslash


def quantize(flow):
    """Boundary conversion: float flow (H,W,2) -> (perp int8 (H,W),
    norm float [0,1] (H,W), static bool (H,W)).
    norm = min(|flow|/FLOW_REF, 1). static = (|flow| == 0 exactly): static
    pixels take the direct (blind) path downstream bit-exactly, so a zero
    flow field reproduces blind EXACTLY (gated) -- the consensus roundtrip
    (fixed max + rescale) may 1-step-perturb near-zero coherence, and static
    pixels must never pay that. motion=None skips the branch entirely (also
    bit-identical, trivially)."""
    flow = np.ascontiguousarray(flow, dtype=np.float64)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"flow geometry {flow.shape}"
    dy, dx = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(dy * dy + dx * dx)
    static = (mag == 0.0)
    norm = np.clip(mag / FLOW_REF, 0, 1)
    ang = np.degrees(np.arctan2(dy, dx)) % 360.0
    # octant table, vectorized (must match _perp_of exactly; the diagonal
    # assignment is cross-checked against coherence_bucket's diag_pos rule --
    # a swap here is parity-invisible, so the compass gate below is load-bearing)
    perp = np.full(ang.shape, 2, dtype=np.int8)
    horiz = ((ang < 22.5) | (ang >= 337.5)) | ((ang >= 157.5) & (ang < 202.5))
    vertm = ((ang >= 67.5) & (ang < 112.5)) | ((ang >= 247.5) & (ang < 292.5))
    slash = ((ang >= 22.5) & (ang < 67.5)) | ((ang >= 202.5) & (ang < 247.5))
    perp[horiz] = 0
    perp[vertm] = 1
    perp[slash] = 3
    return perp, norm, static
