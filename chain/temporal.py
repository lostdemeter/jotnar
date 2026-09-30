"""Temporal IIR (feedback #2): detail memory across frames, all integer.

Rule: D_t = (1-a)*D + a*warp(D_{t-1}), a = A_MIX = 0.5 frozen (dyadic: the
mix is one add + one trunc-halve, no multiplier needed; non-dyadic a is a
LOWERING ERROR per the interp doctrine -- no fixed-mult mix op exists).
First frame (state None) uses D directly: temporal frame 0 == still output
bit-exactly (gated). Flow None + state present is allowed (treated as zero
flow: warp identity is exact, gated).

Warp is nihui-form (unclamped-floor alphas + clamped indices, edge
replicate), mirrored from rife_ref.warp_nihui in integer fixed-point:
  feature fixed @ m (int64 2^-18); flow float pixels -> 2^-14 ints at the
  boundary; per-pixel sx=(x<<14)+fx (int64, floor // for negatives);
  weights in 2^-28, sum // 2^28 (floor, matching rife torch >>28).
IR: warp(fix@m x flow[2^-14]) -> fix@m, same tag in/out. Output via
from_fixed @ m. State carries TRIPLES (canonical inter-frame format).
Static-input behavior (gated in test_temporal): frames 1+ mutually exact
(no drift); frame 1 within measured bounds of frame 0 (max Δe<=3, <=8 px --
6-seed calibration: decoded values near lattice boundaries re-encode 1-2
steps off on 0-2 px; honest cost of the IR binop contract, not shimmer).

Float exists at: flow quantization (boundary), frame I/O (caller). Hot loop
is integers + gathers. See fpu_trap pattern.
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phi_core.lattice as S

ONE = 1 << 14
A_MIX = 0.5  # frozen dyadic IIR weight; change trips test_temporal::mix-frozen


def flow_to_q14(flow):
    """Boundary: float pixels (H,W,2) -> int64 2^-14 units (H,W,2)."""
    flow = np.ascontiguousarray(flow, dtype=np.float64)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"flow geometry {flow.shape}"
    return np.round(flow * ONE).astype(np.int64)


def warp_q(f_q, fx_q14, H, W):
    """Pure-integer nihui warp: f_q int64 (H*W,) fixed counts ->
    warped int64 (H*W,). fx_q14 int64 (H,W,2). No float anywhere."""
    assert f_q.shape == (H * W,) and fx_q14.shape == (H, W, 2)
    F = f_q.reshape(H, W)
    # seam contract (chain/motion.py): flow[...,0]=dy, [...,1]=dx.
    # (RIFE's warp_fixed uses [...,0] as X -- different convention, NOT ours.)
    FX, FY = fx_q14[:, :, 1], fx_q14[:, :, 0]
    xs = (np.arange(W, dtype=np.int64)[None, :] << 14) + FX
    ys = (np.arange(H, dtype=np.int64)[:, None] << 14) + FY
    x0 = xs // ONE  # floor, unclamped (numpy // floors negatives)
    y0 = ys // ONE
    xf = xs - x0 * ONE  # exact, in [0, ONE) by floor construction
    yf = ys - y0 * ONE
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x0 + 1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y0 + 1, 0, H - 1)
    w00 = (ONE - xf) * (ONE - yf)
    w10 = xf * (ONE - yf)
    w01 = (ONE - xf) * yf
    w11 = xf * yf
    tot = (F[y0c, x0c] * w00 + F[y0c, x1c] * w10
           + F[y1c, x0c] * w01 + F[y1c, x1c] * w11) // (ONE * ONE)
    return np.ascontiguousarray(tot.reshape(-1))


def warp_trips(t, flow, m):
    """Triples (H,W) + float flow -> warped triples (H,W) @ m."""
    q = S.to_fixed(t[0], t[1], t[2], m)
    H, W = t[0].shape
    w = warp_q(q.reshape(-1), flow_to_q14(flow), H, W)
    assert int(np.abs(w).max(initial=0)) < 2 ** 53, "warp accum out of range"
    o = S.from_fixed(w, m)
    return (o[0].reshape(H, W).astype(np.int8),
            o[1].reshape(H, W).astype(np.int32),
            o[2].reshape(H, W).astype(np.uint8))


def mix_detail(d_cur, d_prev_warped, m):
    """IIR mix @ same m: (D + warped)/2 via int add + trunc-halve (tdiv by 2,
    C semantics). a=0.5 ONLY (dyadic). Returns triples."""
    qa = S.to_fixed(d_cur[0], d_cur[1], d_cur[2], m)
    qb = S.to_fixed(d_prev_warped[0], d_prev_warped[1], d_prev_warped[2], m)
    return S.from_fixed(S.tdiv(qa + qb, 2), m)


def new_state():
    """Empty temporal state (no previous detail)."""
    return {"d_prev": None}


_DETAIL_KEYS = ("kernel", "m", "blur", "coh_thr", "motion")
_FINISH_KEYS = ("beta", "use_alpha", "ctrl", "blur", "ctrl_atten", "coh_thr",
                "ctrl_mid", "coh_hi", "depth", "ctrl_strong", "ctrl_v5w0",
                "ctrl_v5w1", "ctrl_v5w2", "motion")


def step(y_lin, flow_prev_to_cur, state, m_acc=None, m_cov=None, **kw):
    """One temporal step: detail -> mix with warped prev -> finish.
    y_lin float (H,W); flow None or float (H,W,2) (zeros allowed);
    state from new_state()/prior step. First frame (d_prev None) ignores
    flow and equals the still path bit-exactly. Returns
    (ienh_trip, new_state, info). Remaining kwargs pass to detail/finish
    (beta/blur/ctrl/depth/motion...); temporal mix applies to D before beta.
    NOTE: per-frame `motion` (side-channel) and inter-frame `flow` (warp)
    are independent inputs that may share one field (usual case)."""
    from chain.holo_phi import luminance_detail, luminance_finish
    dk = {k: v for k, v in kw.items() if k in _DETAIL_KEYS}
    fk = {k: v for k, v in kw.items() if k in _FINISH_KEYS}
    a_t, d_t, diag, m_acc, m_cov, H, W = luminance_detail(
        y_lin, m_acc=m_acc, m_cov=m_cov, **dk)
    d_prev = state.get("d_prev")
    if d_prev is None:
        d_mix, from_mem = d_t, False
    else:
        fl = (np.zeros((H, W, 2)) if flow_prev_to_cur is None
              else np.ascontiguousarray(flow_prev_to_cur, dtype=np.float64))
        assert fl.shape == (H, W, 2), f"flow geometry {fl.shape}"
        warped = warp_trips(d_prev, fl, m_cov)
        d_mix, from_mem = mix_detail(d_t, warped, m_cov), True
    yf = np.ascontiguousarray(y_lin, dtype=np.float64)
    ienh_t, info = luminance_finish(a_t, d_mix, diag, yf, H, W, m_acc, m_cov,
                                    **fk)
    info["temporal"] = from_mem
    return ienh_t, {"d_prev": d_t}, info
