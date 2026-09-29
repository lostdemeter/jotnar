"""Fit controller params offline on synthetic degradations (step 2, v2).

Grid search over (iso_atten, mid_atten, coh_hi), scored on the INT chain
(ships what we score). coh_thr stays at its fitted value (grid-flat finding
stands). Pairs deterministic (fixed seeds; stable md5 hash printed/gated).
Best beats the v1 file to rewrite chain/CTRL.json; runtime never fits.
Usage: python3 scripts/fit_ctrl.py [--write]
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chain.holo_phi import enhance_image_int
from chain.control import DEFAULTS, load_ctrl

GRID_ATTEN = [0.25, 0.5]
GRID_MID = [0.4, 0.6]
GRID_STRONG = [0.75, 1.0]
GRID_HI = [0.4, 0.5, 0.6]
THR = 0.15  # fitted v1 value; grid-flat finding stands
BETA = 0.5
N = 48


def pairs():
    rng = np.random.default_rng(11)
    y, x = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    bar = np.where(x > 0.5, 0.85, 0.15)
    diag = np.where(((x * 8).astype(int) + (y * 8).astype(int)) % 2, 0.85, 0.15)
    tex = rng.uniform(0.2, 0.8, (N, N))
    flat = np.full((N, N), 0.5)
    step = np.where(x > 0.5, 1.0, 0.0)
    drng = np.random.default_rng(23)
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    dbar = np.where(xx + yy > 1.0, 0.85, 0.15)  # diagonal edge (v1.1 resolves)
    # corner fixture (v2 governs): quadrant junction has weak coherence
    # (multiple orientations in one window) with LARGE detail -- the pixels
    # mid_atten exists for. Without this fixture the fitter cannot see v2.
    corner = np.where((xx > 0.5) ^ (yy > 0.5), 0.85, 0.15)
    out = {}
    for tag, clean in [("bar", bar), ("diag", diag), ("dbar", dbar),
                       ("corner", corner), ("tex", tex), ("flat", flat),
                       ("step", step)]:
        deg = np.clip(gaussian_filter(clean, 1.5) + drng.normal(0, 0.02, clean.shape), 0, 1)
        out[tag] = (clean, deg)
    return out


def lapvar(im):
    return float(np.var(im[2:, 1:-1] + im[:-2, 1:-1] + im[1:-1, 2:] + im[1:-1, :-2] - 4 * im[1:-1, 1:-1]))


def score(atten, mid, hi, strong, P):
    """Sharpness gain (bar+dbar+corner+tex) minus overshoot minus flat."""
    s = 0.0
    parts = {}
    for tag in ("bar", "dbar", "corner", "tex"):
        clean, deg = P[tag]
        rgb = np.stack([deg] * 3, -1).astype(np.float32)
        out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat", ctrl=True,
                                      ctrl_atten=atten, coh_thr=THR,
                                      ctrl_mid=mid, coh_hi=hi,
                                      ctrl_strong=strong)
        g = lapvar(out[:, :, 0]) / max(lapvar(deg), 1e-9)
        parts[tag] = g
        s += g
    clean, deg = P["step"]
    rgb = np.stack([deg] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat", ctrl=True,
                                  ctrl_atten=atten, coh_thr=THR,
                                  ctrl_mid=mid, coh_hi=hi,
                                  ctrl_strong=strong)
    over = max(float(out[:, :, 0].max() - 1.0), float(-out[:, :, 0].min()), 0.0)
    parts["over"] = over
    s -= 3.0 * over * 10.0
    clean, deg = P["flat"]
    rgb = np.stack([deg] * 3, -1).astype(np.float32)
    out, _, _ = enhance_image_int(rgb, beta=BETA, blur="splat", ctrl=True,
                                  ctrl_atten=atten, coh_thr=THR,
                                  ctrl_mid=mid, coh_hi=hi,
                                  ctrl_strong=strong)
    fr = float(np.abs(out[:, :, 0] - deg).mean())
    parts["flat"] = fr
    s -= 5.0 * fr * 100.0
    return s, parts


def main():
    P = pairs()
    sig = repr(sorted(((k, round(float(v[0].sum() + v[1].sum()), 6)) for k, v in P.items())))
    h = hashlib.md5((str(GRID_ATTEN) + str(GRID_MID) + str(GRID_STRONG)
                     + str(GRID_HI) + sig).encode()).hexdigest()[:16]
    print(f"pairs-hash: {h}")
    cur = load_ctrl()
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                               "chain", "CTRL.json")) as fh:
            raw_keys = set(json.load(fh).keys())
    except Exception:
        raw_keys = set()
    base, base_parts = score(cur.get("iso_atten", 0.25), cur.get("mid_atten", 0.6),
                             cur.get("coh_hi", 0.5), cur.get("strong_atten", 1.0), P)
    print(f"current file: score={base:.3f} {base_parts}")
    best, bestp = base, {"iso_atten": cur.get("iso_atten", 0.25),
                         "mid_atten": cur.get("mid_atten", 0.6),
                         "coh_hi": cur.get("coh_hi", 0.5),
                         "strong_atten": cur.get("strong_atten", 1.0)}
    for a in GRID_ATTEN:
        for m in GRID_MID:
            for st in GRID_STRONG:
                for hh in GRID_HI:
                    s, parts = score(a, m, hh, st, P)
                    flag = ""
                    if s > best:
                        best, bestp, flag = s, {"iso_atten": a, "mid_atten": m,
                                                "coh_hi": hh,
                                                "strong_atten": st}, " <-- new best"
                    print(f"atten={a} mid={m} strong={st} hi={hh}: score={s:.3f} {parts}{flag}")
    print(f"best: {bestp} score={best:.3f} (file {base:.3f})")
    # schema completion: v3 adds strong_atten; if the file predates the key,
    # write the fitted value even on a score tie (content-neutral, but the
    # file must state every param the runtime consumes -- #LIB-012).
    # raw file keys (not the defaults-merged cache): schema completion must
    # detect keys the FILE predates, or the merge hides the gap forever.
    schema_gap = "strong_atten" not in raw_keys
    if "--write" in sys.argv and (best > base or schema_gap):
        if schema_gap and best <= base:
            bestp = {"iso_atten": cur.get("iso_atten", 0.25),
                     "mid_atten": cur.get("mid_atten", 0.6),
                     "coh_hi": cur.get("coh_hi", 0.5),
                     "strong_atten": 1.0}
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain", "CTRL.json")
        with open(mp, "w") as fh:
            json.dump({"iso_atten": bestp["iso_atten"], "mid_atten": bestp["mid_atten"],
                       "strong_atten": bestp["strong_atten"], "coh_hi": bestp["coh_hi"],
                       "coh_thr": THR, "score": best, "default": base,
                       "pairs_hash": h,
                       "note": "v3: fitted strong level completes the decision "
                               "table (iso/mid/strong x coherence bands). "
                               "Rotation symmetry kept (no per-direction params). "
                               "coh_thr frozen at v1 fit."}, fh, indent=2)
        print(f"wrote {mp}")
    elif "--write" in sys.argv:
        print("file unbeaten: untouched (v2 stands)")


if __name__ == "__main__":
    main()
