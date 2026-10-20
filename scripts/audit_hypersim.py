"""Hypersim feasibility audit (student Phase 22): measure, don't train.

Pairs rgb/depth from the extracted shard (out-of-tree), first 24
valid frames: GT quality (invalid rates, gradient/discontinuity
stats), strata transfer (our 6 rules — coverage holds or
median-collapses), scaffold-floor re-measurement (fresh bars;
inherited bars never cross distributions). Scalars banked to
samples/marigold/hypersim_audit.json. CPU only.
Usage: python3 scripts/audit_hypersim.py [--frames N]
"""
import glob
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")
FIXDIR = os.path.join(ROOT, "samples", "marigold")
N_FRAMES = int(sys.argv[sys.argv.index("--frames") + 1]) \
    if "--frames" in sys.argv else 24


def pairs():
    """(rgb_path, depth_path) sorted, both present."""
    out = []
    for dp in sorted(glob.glob(os.path.join(SHARD, "*", "depth_plane_cam_00_fr*.png"))):
        rp = os.path.join(os.path.dirname(dp),
                          os.path.basename(dp).replace("depth_plane_", "rgb_"))
        if os.path.isfile(rp):
            out.append((rp, dp))
    return out


def sobel_mag(t):
    import numpy as np
    p = np.pad(t, 1, mode="edge")
    gx = (-p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
          + p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    return np.sqrt(gx * gx + gy * gy)


def main():
    import numpy as np
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H

    all_pairs = pairs()
    print(f"paired frames found: {len(all_pairs)}", flush=True)
    frames, seen = [], 0
    for rp, dp in all_pairs:
        if len(frames) >= N_FRAMES:
            break
        seen += 1
        try:
            rgb = np.asarray(Image.open(rp).convert("RGB")).astype(np.float64) / 255.0
            raw = np.asarray(Image.open(dp))
            if raw.shape[:2] != rgb.shape[:2]:
                continue
            depth = raw.astype(np.float64) / 1000.0
        except Exception:
            continue
        frames.append((os.path.basename(os.path.dirname(rp))
                       + "/" + os.path.basename(rp), rgb, depth))
    print(f"audited frames: {len(frames)} (scanned {seen})", flush=True)
    assert len(frames) == N_FRAMES, f"only {len(frames)} valid frames"

    per, cov, scaff = [], {k: [] for k in H.STRATA}, {k: [] for k in H.STRATA}
    for name, rgb, depth in frames:
        valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.0)
        invalid = 1.0 - valid.mean()
        g = sobel_mag(np.where(valid, depth, np.median(depth[valid])))
        per.append({
            "name": name, "shape": list(depth.shape),
            "invalid_rate": float(invalid),
            "depth_min": float(depth[valid].min()),
            "depth_p50": float(np.median(depth[valid])),
            "depth_p95": float(np.percentile(depth[valid], 95)),
            "depth_max": float(depth[valid].max()),
            "grad_p50": float(np.median(g)),
            "grad_p90": float(np.percentile(g, 90)),
            "grad_p99": float(np.percentile(g, 99)),
            "discont_frac": float((g > 1.0).mean()),
        })
        masks = H.strata_masks(depth, rgb)
        for k in H.STRATA:
            cov[k].append(float(masks[k].mean()))
        c = float(depth[valid].mean())
        sc = {}
        for k in H.STRATA:
            m = masks[k]
            if int(m.sum()) == 0:
                scaff[k].append(None)
                sc[k] = None
            else:
                A = np.stack([np.full(m.sum(), c), np.ones(m.sum())], 1)
                (a, b), _, _, _ = np.linalg.lstsq(
                    A, depth[m].ravel(), rcond=None)
                v = float(np.abs(a * c + b - depth[m]).mean())
                scaff[k].append(v)
                sc[k] = v
        per[-1]["scaffold"] = sc

    cov_ok = {k: sum(1 for v in vals if 0.30 < v < 0.70) for k, vals in cov.items()}
    scaff_min = {}
    for k in H.STRATA:
        vals = [v for v in scaff[k] if v is not None]
        scaff_min[k] = min(vals) if vals else None
    bars = {k: (v / 5.0 if v else None) for k, v in scaff_min.items()}
    inv = [p["invalid_rate"] for p in per]
    verdict = {
        "invalid_mean": float(np.mean(inv)),
        "invalid_worst": float(max(inv)),
        "coverage_ok_rules": sum(1 for k in H.STRATA if cov_ok[k] == N_FRAMES),
        "bars_computable": all(v is not None and np.isfinite(v) and v > 0
                               for v in bars.values()),
    }
    go = (verdict["invalid_mean"] < 0.05 and verdict["invalid_worst"] < 0.20
          and verdict["coverage_ok_rules"] >= 5 and verdict["bars_computable"])
    verdict["go"] = bool(go)
    out = {"frames": per,
           "coverage_30_70_counts": cov_ok,
           "scaffold_minima": scaff_min,
           "bars_fresh_div5": bars,
           "verdict": verdict}
    json.dump(out, open(os.path.join(FIXDIR, "hypersim_audit.json"), "w"), indent=1)
    print("wrote samples/marigold/hypersim_audit.json", flush=True)
    print(f"invalid mean {verdict['invalid_mean']:.4f} "
          f"worst {verdict['invalid_worst']:.4f}", flush=True)
    print(f"coverage-ok rules: {verdict['coverage_ok_rules']}/6", flush=True)
    print(f"bars: { {k: (None if v is None else round(v, 4)) for k, v in bars.items()} }",
          flush=True)
    print(f"verdict: {'GO' if go else 'NO-GO'}", flush=True)


if __name__ == "__main__":
    main()
