"""Fit v5 detail-gate weights offline (learned-v5, the tiny net).

Grid search over (w0, w1, w2), scored on the INT chain (ships what we
score). Zero weights give scale~=1 (sigmoid(0)=0.50023 to LUT precision),
so the base IS v4: the grid must beat v4 to rewrite (must-beat doctrine).
Pairs reused unchanged from fit_ctrl (corner fixture included); pairs-hash
uses the v5 grid strings so the gate detects grid/pair drift. Runtime never
fits. Usage: python3 scripts/fit_v5.py [--write]
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain.holo_phi import enhance_image_int
from chain.control import load_ctrl
from fit_ctrl import pairs, lapvar

GRID_W0 = [-2.0, -1.0, 0.0]
GRID_W1 = [0.0, 2.0, 4.0]
# W2 extended to 8 after the first grid hit its edge at 4.0 (probe: w1
# plateaus from 4.0 on -- identical scores at 6/8, i.e. sigmoid-saturated
# interior optimum; w2 still climbing but flattening +0.05/+0.03).
GRID_W2 = [4.0, 6.0, 8.0]
BETA = 0.5


def score(w0, w1, w2, P):
    """Sharpness gain (bar+dbar+corner+tex) minus overshoot minus flat.
    DELIBERATELY no noise-parity term: a max(0,40-npsnr) veto was tried and
    reverted -- nothing on the grid clears 40 (best 37.8), so an unreachable
    veto is pure drag that distorts ranking (it elected (0,0,4), killing the
    coherence term the rule exists for). Lesson banked (#LIB-015): verify a
    veto's reachability before adding it; price unreachable bars in the gate
    suite (measured rows + mechanism), not in the objective."""
    s = 0.0
    parts = {}
    for tag in ("bar", "dbar", "corner", "tex"):
        clean, deg = P[tag]
        rgb = np.stack([deg] * 3, -1).astype(np.float32)
        out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat_soft",
                                      ctrl="v5", ctrl_v5w0=w0,
                                      ctrl_v5w1=w1, ctrl_v5w2=w2)
        g = lapvar(out[:, :, 0]) / max(lapvar(deg), 1e-9)
        parts[tag] = g
        s += g
    clean, deg = P["step"]
    rgb = np.stack([deg] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat_soft",
                                  ctrl="v5", ctrl_v5w0=w0,
                                  ctrl_v5w1=w1, ctrl_v5w2=w2)
    over = max(float(out[:, :, 0].max() - 1.0), float(-out[:, :, 0].min()), 0.0)
    parts["over"] = over
    s -= 3.0 * over * 10.0
    clean, deg = P["flat"]
    rgb = np.stack([deg] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat_soft",
                                  ctrl="v5", ctrl_v5w0=w0,
                                  ctrl_v5w1=w1, ctrl_v5w2=w2)
    fr = float(np.abs(out[:, :, 0] - deg).mean())
    parts["flat"] = fr
    s -= 5.0 * fr * 100.0
    return s, parts


def main():
    P = pairs()
    sig = repr(sorted(((k, round(float(v[0].sum() + v[1].sum()), 6)) for k, v in P.items())))
    h = hashlib.md5((str(GRID_W0) + str(GRID_W1) + str(GRID_W2) + sig).encode()).hexdigest()[:16]
    print(f"pairs-hash: {h}")
    cur = load_ctrl()
    base, base_parts = score(cur.get("v5_w0", 0.0), cur.get("v5_w1", 0.0),
                             cur.get("v5_w2", 0.0), P)
    print(f"current (v5 zeros ~= v4): score={base:.3f} {base_parts}")
    best, bestp = base, {"v5_w0": cur.get("v5_w0", 0.0),
                         "v5_w1": cur.get("v5_w1", 0.0),
                         "v5_w2": cur.get("v5_w2", 0.0)}
    for a in GRID_W0:
        for b in GRID_W1:
            for c in GRID_W2:
                s, parts = score(a, b, c, P)
                flag = ""
                if s > best:
                    best, bestp, flag = s, {"v5_w0": a, "v5_w1": b,
                                            "v5_w2": c}, " <-- new best"
                print(f"w0={a} w1={b} w2={c}: score={s:.3f} {parts}{flag}")
    print(f"best: {bestp} score={best:.3f} (base {base:.3f})")
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                               "chain", "CTRL.json")) as fh:
            raw_keys = set(json.load(fh).keys())
    except Exception:
        raw_keys = set()
    schema_gap = not {"v5_w0", "v5_w1", "v5_w2"} <= raw_keys
    if "--write" in sys.argv and (best > base or schema_gap):
        if schema_gap and best <= base:
            bestp = {"v5_w0": 0.0, "v5_w1": 0.0, "v5_w2": 0.0}
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain", "CTRL.json")
        with open(mp) as fh:
            d = json.load(fh)
        d.update({"v5_w0": bestp["v5_w0"], "v5_w1": bestp["v5_w1"],
                  "v5_w2": bestp["v5_w2"], "v5_score": best,
                  "v5_pairs_hash": h,
                  "note": "v5: learned detail gate (w0,w1,w2) on (1,coh,dhat); "
                          "zeros ~= v4 (sigmoid(0)=0.50023 LUT); must-beat held."})
        with open(mp, "w") as fh:
            json.dump(d, fh, indent=2)
        print(f"wrote {mp}")


if __name__ == "__main__":
    main()
