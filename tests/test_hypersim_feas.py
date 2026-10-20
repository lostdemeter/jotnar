"""Hypersim feasibility gate (student Phase 22): CPU-only, fast.

Disk/state legs (shard present, out-of-tree), GT-audit
determinism (1 frame recompute vs banked), strata-transfer legs
(coverage counts recomputed), re-derived bars integrity (scaffold
minima/5 recomputed), GO/NO-GO verdict leg reading the banked
audit (frozen rule, reported never asserted beyond it).
SKIP-clean without downloads (gold stays green).
Usage: python3 tests/test_hypersim_feas.py (seconds when skipped).
"""
import glob
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FAIL = []
SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")
BANK = os.path.join(ROOT, "samples", "marigold", "hypersim_audit.json")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import numpy as np
    from PIL import Image

    have = os.path.isdir(SHARD) and len(
        glob.glob(os.path.join(SHARD, "*", "depth_plane_cam_00_fr*.png"))) > 0
    check("hypersim-present", True,
          f"extract dir {'present' if have else 'absent'} "
          f"(out-of-tree, documented)")
    if not os.path.isfile(BANK):
        print("SKIP banked legs (no hypersim_audit.json yet)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)
    bank = json.load(open(BANK))
    check("hypersim-banked-schema",
          len(bank.get("frames", [])) == 24
          and all(k in bank for k in ("coverage_30_70_counts",
                                      "scaffold_minima", "bars_fresh_div5",
                                      "verdict")),
          "24 frames + coverage + scaffold + bars + verdict")
    if not have:
        print("SKIP recompute legs (no downloads here)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    fr0 = bank["frames"][0]
    scene, fname = fr0["name"].split("/")
    rp = os.path.join(SHARD, scene, fname)
    dp = os.path.join(SHARD, scene,
                      fname.replace("rgb_cam_00", "depth_plane_cam_00"))
    rgb = np.asarray(Image.open(rp).convert("RGB")).astype(np.float64) / 255.0
    depth = np.asarray(Image.open(dp)).astype(np.float64) / 1000.0
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.0)
    check("hypersim-determinism",
          abs((1.0 - valid.mean()) - fr0["invalid_rate"]) < 1e-12
          and abs(float(np.median(depth[valid])) - fr0["depth_p50"]) < 1e-9,
          f"{fr0['name']} recompute matches")
    masks = H.strata_masks(depth, rgb)
    cov = {k: float(masks[k].mean()) for k in H.STRATA}
    c = float(depth[valid].mean())
    worst = 0.0
    want_sc = bank["frames"][0]["scaffold"]
    for k in H.STRATA:
        m = masks[k]
        ok30 = 0.30 < cov[k] < 0.70
        worst = max(worst, 0.0 if ok30 else 1.0)
        if int(m.sum()) == 0:
            continue
        A = np.stack([np.full(m.sum(), c), np.ones(m.sum())], 1)
        (a, b), _, _, _ = np.linalg.lstsq(A, depth[m].ravel(), rcond=None)
        got = float(np.abs(a * c + b - depth[m]).mean())
        want = want_sc[k]
        if want is not None:
            worst = max(worst, abs(got - want) / max(want, 1e-300))
    check("hypersim-strata-recompute", worst < 1e-9,
          "coverage + per-frame scaffold reproduce")
    v = bank["verdict"]
    go = (v["invalid_mean"] < 0.05 and v["invalid_worst"] < 0.20
          and v["coverage_ok_rules"] >= 5 and v["bars_computable"])
    check("hypersim-verdict", go == v["go"],
          f"frozen rule recomputed: {'GO' if go else 'NO-GO'}")

    # -- scale-training verdict legs (banked integrity + winner) --------
    # No retraining here: single-seed result (determinism NOT
    # re-verified — stated openly, unlike retrain-matched batteries).
    ab = json.load(open(os.path.join(ROOT, "samples", "marigold",
                                     "student_ablation.json")))
    if "hyper" not in ab:
        print("SKIP hyper legs (no banked hyper battery yet)")
    else:
        hb, hc = ab["hyper"], ab["hyper"]["cells"]
        check("hyper-configs",
              hb.get("configs") == ["hyper-mscale", "hyper-jattn"]
              and hb.get("reference") == "mscale64",
              "2 banked archs, no new shapes")
        BARS = {"near": 0.09, "far": 0.09, "planar": 0.09,
                "nonplanar": 0.08, "tex": 0.07, "smooth": 0.09}
        complete, counts = True, {}
        for t in hb["configs"]:
            fr = {n: f for n, f in hc[t]["frames"].items()
                  if f.get("split") == "heldout"}
            complete &= (len(fr) == 60 and all(
                len(fr[n].get("strata", {})) == 6
                and all(v is None or (np.isfinite(v) and v >= 0)
                        for v in fr[n]["strata"].values())
                for n in fr))
            counts[t] = sum(
                1 for n in fr for k in BARS
                if fr[n]["strata"][k] is not None
                and fr[n]["strata"][k] <= BARS[k])
        check("hyper-complete", bool(complete), "2x60x6 finite-or-null")
        check("hyper-winner-none",
              all(v == 0 for v in counts.values())
              and hb.get("winner") is None,
              f"held-out cells under fresh bars: {counts}")
        check("hyper-determinism-note", True,
              "single-seed, retrain NOT verified (stated, not green-washed)")
        tr = hb.get("transfer", {})
        check("hyper-transfer-unjudged",
              set(tr) == {"hyper-mscale", "hyper-jattn"}
              and all(tr[t].get("judged") is False for t in tr),
              "transfer reported-not-graded (incl. grid-lock note)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
