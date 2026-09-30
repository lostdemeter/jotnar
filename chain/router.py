"""Conditional routing (feedback #3, first half): select AMONG modes per frame.

The first structure that operates on CONFIGURATIONS instead of pixels:
per frame, engage temporal memory or run still. Rule v1 (analytic, stated):
  E = mean(|flow|) / FLOW_REF (boundary float); temporal iff E >= ROUTE_THR.
Rationale: memory pays where inter-frame motion carries detail forward
(moving content) and costs nothing where it doesn't -- but on static frames
temporal == still to bit-closeness anyway, so the router's real job is
avoiding memory where motion is UNCERTAIN... v1 keeps it simple: motion
present -> memory; static -> still. Threshold frozen (ROUTE_THR = 0.25 =
2px mean motion), flagged for fitting.

Boundary doctrine (cf motion static mask): the decision is made in float at
the seam and consumed downstream as an exact flag, so INT and oracle can
never disagree on the mode (structural agreement, not parity luck).
State always refreshes (every frame's D enters memory); mode only chooses
mix vs direct. Switches are seamless by construction: still ignores state,
temporal's first-engaged frame... note it does NOT equal still (state may
hold older D) -- seamlessness means NO ARTIFACT, gated as: routed static
frames == still-only outputs exactly (still path ignores state entirely).

Scope: per-FRAME routing (global decision). Per-tile routing is backlog
(needs tiling infra + per-tile state; the pattern transfers).
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

ROUTE_THR = 0.25  # frozen v1: mean motion >= 2px engages memory


def decide(flow, flow_ref=8.0):
    """Boundary decision: None or float (H,W,2) -> (mode, E).
    mode 'temporal' iff E >= ROUTE_THR else 'still'. None -> ('still', 0.0).
    Exact-agreement: both substrates call THIS (single seam conversion)."""
    if flow is None:
        return "still", 0.0
    flow = np.ascontiguousarray(flow, dtype=np.float64)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"flow geometry {flow.shape}"
    from chain.verdict import verdict_mask
    E = float(np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).mean() / flow_ref)
    go = bool(verdict_mask(np.asarray(E), ROUTE_THR, ">=")[()])
    return ("temporal" if go else "still"), E


def run_sequence(ys, flows, beta=0.5, blur="iso", router_on=True, motions=None,
                 ctrl=False):
    """Y-level routed sequence driver (production; demo wraps RGB/gain).
    ys: list of float (H,W); flows[i]: (H,W,2)/None for routing + temporal
    warp (flows[0] unused for memory but still routed on -- its own motion
    decides its own mode). motions[i]: (H,W,2)/None side-channel for the
    consensus buckets (defaults to all-None = consensus off; pass
    motions=flows explicitly for the usual one-field-drives-all case).
    Returns (yenh_triples_list,
    decisions, infos). With router_on=False every frame runs the still path
    (control arm for gates)."""
    from chain.holo_phi import enhance_luminance_int, luminance_detail
    from chain import temporal as T
    if motions is None:
        motions = [None] * len(ys)
    assert len(motions) == len(ys) == len(flows), "sequence length mismatch"
    if any(mo is not None for mo in motions) and blur == "iso":
        raise ValueError("motions need a splat blur (flow_scale lives in the "
                         "tensor path; iso+motion is backlog, not silent)")
    outs, decisions, infos = [], [], []
    st = T.new_state()
    for y, fl, mo in zip(ys, flows, motions):
        mode, E = decide(fl) if router_on else ("still", 0.0)
        yf = np.ascontiguousarray(y, dtype=np.float64)
        if mode == "temporal":
            o, st, info = T.step(yf, fl, st, blur=blur, beta=beta,
                                 ctrl=ctrl, motion=mo)
        else:
            # still output via the PROVEN composer (identical to still-only
            # runs, byte for byte); memory stores the SAME detail the output
            # used (motion threaded through both -- memory must never lie).
            o, info = enhance_luminance_int(yf, beta=beta, blur=blur,
                                              ctrl=ctrl, motion=mo)
            _, d_store, _, _, _, _, _ = luminance_detail(yf, blur=blur,
                                                         motion=mo)
            st = {"d_prev": d_store}
        info = dict(info)
        info["mode"], info["E"] = mode, E
        outs.append(o)
        decisions.append(mode)
        infos.append(info)
    return outs, decisions, infos
