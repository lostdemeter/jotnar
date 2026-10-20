"""Resonance-depth correlation analysis (Phase: measurement, CPU-only).

Banked teacher fixtures -> patch tokens -> frozen key variants ->
phase-distance vs teacher-depth-difference. Banks
samples/marigold/resonance_depth.json (the gate reads it; this script
does the numpy, the test only verifies). Sibling
resonant_array imported read-only (dont_write_bytecode: not even
__pycache__ lands in their tree). See docs/RESONANCE.md (frozen).
Usage: python3 scripts/analyze_resonance_depth.py
"""
import sys

sys.dont_write_bytecode = True

import glob
import json
import os

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "/home/balrog/Documents/OpenCode/resonant_array")

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTP = os.path.join(FIXDIR, "resonance_depth.json")

CELL = 32
EPS = 0.1
TAU = 0.1
HEADS = 4
BAR = 0.10
REAL = ["church", "dogs", "squirrel", "train",
        "geo_webcam_0_combined", "geo_webcam_1_combined"]


def load_pair(name):
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    row = next(r for r in man["fixtures"] if r["name"] == name)
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", name + ".*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    return im, t


def main():
    import resonant_array as RA
    assert RA.GAMMAS[:4] == [14.134725, 21.02204, 25.010858, 30.424876], \
        "sibling GAMMAS drifted — stop, do not silently re-derive"
    gammas = np.array(RA.GAMMAS[:HEADS])
    twopi = 2.0 * np.pi

    rgbs, depths, frame_id = [], [], []
    for fi, n in enumerate(REAL):
        im, t = load_pair(n)
        H, W, _ = im.shape
        for y in range(0, H, CELL):
            for x in range(0, W, CELL):
                rgbs.append(im[y:y + CELL, x:x + CELL].reshape(-1, 3).mean(0))
                depths.append(t[y:y + CELL, x:x + CELL].mean())
                frame_id.append(fi)
    rgb = np.array(rgbs)
    dep = np.array(depths)
    frm = np.array(frame_id)
    n = len(dep)
    print(f"tokens: {n} ({len(REAL)} frames)", flush=True)
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    rq = np.clip((rgb * 255 + 0.5).astype(int), 0, 255)
    keys = {
        "V1_luma": np.round(luma * 1000).astype(np.int64),
        "V2_lumapos": (np.round(luma * 1000).astype(np.int64) * 4096
                       + np.arange(n)),
        "V3_rgbpack": (rq[:, 0] * 65536 + rq[:, 1] * 256 + rq[:, 2]),
    }
    assert max(int(k.max()) for k in keys.values()) < 2 ** 24, \
        "key exceeds float-exact range"

    iu = np.triu_indices(n, 1)
    dlt = np.abs(dep[iu[0]] - dep[iu[1]])
    same = dlt <= EPS
    cross = frm[iu[0]] != frm[iu[1]]
    print(f"pairs: {len(iu[0])} same-depth: {int(same.sum())} "
          f"(cross-frame same: {int((same & cross).sum())})", flush=True)

    def phases(key):
        return (gammas[:, None] * key[None, :]) % twopi

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        rx -= rx.mean()
        ry -= ry.mean()
        return float((rx * ry).mean() / (rx.std() * ry.std()))

    def dmin_all(P):
        """Min-over-heads circular distance for all pairs, row-chunked."""
        out = np.empty(len(iu[0]))
        CH = 512
        for a in range(0, n, CH):
            sel = (iu[0] >= a) & (iu[0] < a + CH)
            A = P[:, a:a + CH].T[iu[0][sel] - a]  # (m, H)
            B = P[:, iu[1][sel]].T                # (m, H)
            df = np.abs(A - B)
            out[sel] = (np.minimum(df, twopi - df) / np.pi).min(1)
        return out

    rng = np.random.RandomState(0)
    results = {}
    for vname, key in keys.items():
        dmin = dmin_all(phases(key))
        res = dmin < TAU
        rate_same = float(res[same].mean())
        rate_same_cross = float(res[same & cross].mean())
        rho = spearman(dmin, dlt)
        # null: shuffled keys (seeded), same pipeline
        dnull = dmin_all(phases(rng.permutation(key)))
        rate_null = float((dnull < TAU)[same].mean())
        lift = rate_same - rate_null
        results[vname] = {"n_tokens": n,
                          "rate_same": rate_same,
                          "rate_same_cross": rate_same_cross,
                          "rate_null": rate_null,
                          "lift": lift,
                          "spearman": rho,
                          "pass": bool(lift >= BAR)}
        print(f"{vname}: same={rate_same:.4f} cross={rate_same_cross:.4f} "
              f"null={rate_null:.4f} lift={lift:+.4f} rho={rho:+.4f} "
              f"{'PASS' if lift >= BAR else 'fail'}", flush=True)

    rep = {"variants": {k: "frozen docs/RESONANCE.md" for k in keys},
           "key_spec": {"V1_luma": "round(mean_luma*1000)",
                        "V2_lumapos": "luma_q*4096+cell_index",
                        "V3_rgbpack": "R*65536+G*256+B of quantized mean"},
           "frames": REAL, "cell": CELL, "eps": EPS, "tau": TAU,
           "heads": HEADS, "null": "seeded shuffle (seed 0)", "bar": BAR,
           "results": results,
           "verdict": ("signal" if any(r["pass"] for r in results.values())
                       else "dead-for-depth")}
    json.dump(rep, open(OUTP, "w"), indent=1)
    print(f"wrote {OUTP} verdict={rep['verdict']}", flush=True)


if __name__ == "__main__":
    main()
