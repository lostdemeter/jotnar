"""Depth prior (step 3a): DAV2 relative depth as an offline modulation source.

Doctrine: the depth map arrives like calibration data -- computed once with
the depth codec (torch allowed here, never in the hot loop), cached to
samples/depth/, consumed as floats at the encode boundary. This is L1
composition (files at the seam); L2 triple-direct handoff is backlog
(dav2_reverse/geo_int.py already speaks triples -- the export path just
isn't wired yet). Rule (analytic v1, stated): relative depth normalized
per-frame (DAV2 is scale-ambiguous, so only ORDER matters), split at the
median, near gets full beta, far gets FAR_ATTEN. Median split, not learned
thresholds: robust with zero params. Fitting joins v3.
"""
import hashlib
import os

import numpy as np
from PIL import Image

DAV2 = "/home/thorin/Documents/OpenCode/dav2_revisited/dav2_reverse"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "samples", "depth")

FAR_ATTEN = 0.5  # analytic v1; fitted ownership moves to v3 (see note above)


def _model():
    import sys
    sys.path.insert(0, DAV2)
    from geo_depth import GeometricDepthAnythingV2
    return GeometricDepthAnythingV2()


def get_depth(rgb01, tag):
    """RGB [0,1] + cache tag -> depth float HxW at INPUT resolution.
    Cached under samples/depth/<tag>.npy (deterministic: same tag+input
    bytes -> same file; mismatch raises instead of silently reusing)."""
    os.makedirs(CACHE, exist_ok=True)
    rgb01 = np.ascontiguousarray(rgb01, dtype=np.float64)
    key = hashlib.md5(rgb01.tobytes()).hexdigest()[:16]
    path = os.path.join(CACHE, tag + ".npy")
    sidecar = os.path.join(CACHE, tag + ".key")
    if os.path.exists(path) and os.path.exists(sidecar):
        with open(sidecar) as fh:
            if fh.read().strip() == key:
                d = np.load(path)
                if d.shape == rgb01.shape[:2]:
                    return d
        raise ValueError(f"depth cache tag collision: {tag} (input changed?)")
    d = _model().predict(rgb01.astype(np.float64))
    # model resolution (longest side 518, multiple of 14) -> input geometry
    im = Image.fromarray(d.astype(np.float32), mode="F").resize(
        (rgb01.shape[1], rgb01.shape[0]), Image.BILINEAR)
    d = np.asarray(im, dtype=np.float64)
    np.save(path, d)
    with open(sidecar, "w") as fh:
        fh.write(key)
    return d


def near_mask(depth):
    """Median split on relative depth (order only -- scale-ambiguous input).
    Returns bool array True=near. Verdict via the shared helper (#LIB-019);
    identical computation to before, routed through one conversion (0-diff
    refactor -- the suites prove it). NOTE: comparing raw depth against the
    raw median would skip a roundtrip (median commutes with monotone maps)
    but differs by up to 1ulp on even-sized arrays -- NOT done here; that
    optimization needs its own gate if anyone wants it."""
    from chain.verdict import verdict_mask
    d = np.ascontiguousarray(depth, dtype=np.float64)
    lo, hi = float(d.min()), float(d.max())
    n = np.zeros_like(d) if hi <= lo else (d - lo) / (hi - lo)
    return verdict_mask(n, float(np.median(n)), "<=")


def depth_mult(mult_shape, near, atten=FAR_ATTEN):
    """Per-pixel depth multiplier triples via exact select (select_mux
    pattern, nth use -- promotion already closed in #LIB-009)."""
    import sys
    sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import phi_core.lattice as S
    from chain.holo_phi import select_mux
    near_t = S.encode(np.ones(mult_shape, np.float64))
    far_t = S.encode(np.full(mult_shape, float(atten), np.float64))
    bucket = np.where(np.ascontiguousarray(near, bool), 0, 1).astype(np.int8)
    return select_mux([near_t, far_t], bucket)
