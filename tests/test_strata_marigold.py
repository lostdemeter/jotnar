"""Strata harness gate (student Phase 1): proven falsification both ways.

Six strata over banked teacher fixtures (rules below, all deterministic):
near/far (median split via chain/depthprior.near_mask), planar/
nonplanar (teacher-depth Sobel gradient 40/60 percentiles), textured/
smooth (input-luma gradient 40/60 percentiles). Comparison is
least-squares affine-aligned MAE per stratum (global fit per frame;
teacher outputs are scale-shift-ambiguous, never raw-compared).
Margin bars per stratum = scaffold floor/5, scaffold = constant
teacher-mean stub (SKELETON SCAFFOLD, not a model):
  near 0.09, far 0.09, planar 0.09, nonplanar 0.08, tex 0.07, smooth 0.09
(measured scaffold minima 0.457/0.481/0.484/0.403/0.377/0.483).
Legs: strata deterministic + coverage; teacher-vs-seeded-self passes
everything (MAE 0.0); skeleton fails every stratum every frame (the
falsification proof -- an untested harness is unproven); alignment
direction guarded (constant scores ~teacher-std, never 0 -- flipped
alignment would let constants pass); manifest integrity.
Usage: python3 tests/test_strata_marigold.py (CPU only, seconds).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

FIXDIR = os.path.join(ROOT, "samples", "marigold")
BARS = {"near": 0.09, "far": 0.09, "planar": 0.09,
        "nonplanar": 0.08, "tex": 0.07, "smooth": 0.09}
STRATA = sorted(BARS)


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def gradmag(a):
    gy, gx = np.gradient(a.astype(np.float64))
    return np.sqrt(gx * gx + gy * gy)


def strata_masks(teacher, rgb):
    """Deterministic stratum masks (documented rules, no randomness)."""
    from chain.depthprior import near_mask
    near = near_mask(teacher)
    gd = gradmag(teacher)
    plo, phi = np.percentile(gd, [40, 60])
    luma = rgb @ np.array([0.299, 0.587, 0.114])
    gl = gradmag(luma)
    tlo, thi = np.percentile(gl, [40, 60])
    return {"near": near, "far": ~near,
            "planar": gd <= plo, "nonplanar": gd >= phi,
            "tex": gl >= thi, "smooth": gl <= tlo}


def align_mae(student, teacher, mask):
    """Least-squares affine fit student->teacher (global), MAE on mask."""
    s = np.ascontiguousarray(student, dtype=np.float64).ravel()
    t = np.ascontiguousarray(teacher, dtype=np.float64).ravel()
    A = np.stack([s, np.ones(s.size)], 1)
    (a, b), _, _, _ = np.linalg.lstsq(A, t, rcond=None)
    return float(np.abs(a * s + b - t)[np.ascontiguousarray(mask).ravel()].mean())


def SKELETON_SCAFFOLD(teacher):
    """Deliberately-weak stub: constant teacher-mean per frame. NOT a
    model, NOT a candidate -- its only job is to FAIL margins visibly,
    proving the harness can catch a bad student."""
    return np.full_like(np.ascontiguousarray(teacher, dtype=np.float64),
                        float(np.ascontiguousarray(teacher,
                                                   dtype=np.float64).mean()))


def main():
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    check("strata-manifest", len(man["fixtures"]) == 6
          and all(os.path.isfile(os.path.join(FIXDIR, r["npy"]))
                  for r in man["fixtures"])
          and sum(1 for r in man["fixtures"] if r.get("heldout")) == 2,
          f"{len(man['fixtures'])} banked rows (4 train + 2 heldout) on disk")
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    frames = {}
    for row in man["fixtures"]:
        t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
        cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", row["name"] + ".*")))
        assert cand, row["name"]
        im = np.asarray(Image.open(cand[0]).convert("RGB")
                        .resize((t.shape[1], t.shape[0]))) / 255.0
        frames[row["name"]] = (t, im)

    # -- strata deterministic + coverage --------------------------------------
    det_ok, cov_ok = True, True
    for name, (t, im) in frames.items():
        m1, m2 = strata_masks(t, im), strata_masks(t, im)
        det_ok &= all(np.array_equal(m1[k], m2[k]) for k in STRATA)
        cov_ok &= all(0.30 < m1[k].mean() < 0.70 for k in STRATA)
    check("strata-deterministic", bool(det_ok), "masks rerun-identical")
    check("strata-coverage", bool(cov_ok),
          "every stratum 30-70% of every frame (percentile rules)")

    # -- teacher-vs-seeded-self passes everything ------------------------------
    self_ok = True
    for name, (t, im) in frames.items():
        m = strata_masks(t, im)
        A = np.stack([t.ravel(), np.ones(t.size)], 1)
        (a, b), _, _, _ = np.linalg.lstsq(A, t.ravel(), rcond=None)
        self_ok &= abs(a - 1.0) < 1e-9 and abs(b) < 1e-9
        for k in STRATA:
            if align_mae(t, t, m[k]) >= BARS[k]:
                self_ok = False
    check("strata-teacher-self", bool(self_ok),
          "MAE 0.0 everywhere, margins clear by 5x+")

    # -- skeleton fails every stratum every frame (falsification proof) ---------
    fail_ok, worst = True, {}
    for name, (t, im) in frames.items():
        m = strata_masks(t, im)
        s = SKELETON_SCAFFOLD(t)
        for k in STRATA:
            v = align_mae(s, t, m[k])
            worst[k] = max(worst.get(k, 0.0), v)
            if v <= BARS[k]:
                fail_ok = False
    check("strata-skeleton-fails", bool(fail_ok),
          "scaffold over every bar: " +
          ", ".join(f"{k} {worst[k]:.2f}>{BARS[k]}" for k in STRATA))

    # -- alignment-direction guard (flipped alignment lets constants pass) -------
    t0 = next(iter(frames.values()))[0]
    s0 = SKELETON_SCAFFOLD(t0)
    A = np.stack([s0.ravel(), np.ones(s0.size)], 1)
    (a, b), _, _, _ = np.linalg.lstsq(A, t0.ravel(), rcond=None)
    resid = float(np.abs(a * s0 + b - t0).mean())
    check("strata-alignment-direction", resid > 0.5 * float(t0.std()),
          f"constant residual {resid:.3f} ~= teacher std "
          f"{float(t0.std()):.3f} (never 0)")

    # -- student evaluation (lose-honest reporter, not a pass gate) --------------
    # The harness evaluates; the numbers win or lose on their own. These
    # legs assert structural integrity (complete table, finite numbers,
    # reproducible config) and PRINT the per-cell verdicts. A bad student
    # failing bars is a passing gate with failing cells -- that is the
    # Phase-2 contract (numbers + diagnosis, never silent tuning).
    import json as _js
    srep_path = os.path.join(FIXDIR, "student_run.json")
    if not os.path.isfile(srep_path):
        print("SKIP student legs (no student_run.json -- train first)")
    else:
        srep = _js.load(open(srep_path))
        cfg_ok = all(k in srep for k in
                     ("tag", "width", "steps", "lr", "seed", "patch",
                      "train", "heldout", "torch", "device", "weights",
                      "frames"))
        check("student-config", bool(cfg_ok),
              f"tag={srep.get('tag')} w={srep.get('width')} "
              f"steps={srep.get('steps')} seed={srep.get('seed')} "
              f"patch={srep.get('patch')} torch={srep.get('torch')}")
        fr = srep.get("frames", {})
        complete = (sorted(fr) == sorted(srep.get("train", []) +
                                         srep.get("heldout", []))
                    and all(sorted(fr[n].get("strata", {})) == STRATA
                            and all(np.isfinite(v) and v >= 0
                                    for v in fr[n]["strata"].values())
                            for n in fr))
        check("student-table-complete", bool(complete),
              f"{len(fr)} frames x {len(STRATA)} strata, all finite")
        npass, ncell = 0, 0
        for n in sorted(fr):
            verdicts = []
            for k in STRATA:
                ncell += 1
                ok = fr[n]["strata"][k] <= BARS[k]
                npass += ok
                verdicts.append(f"{k} {fr[n]['strata'][k]:.3f}"
                                f"{'PASS' if ok else 'FAIL'}")
            print(f"student {n} [{fr[n].get('split')}]: "
                  + ", ".join(verdicts), flush=True)
        print(f"student cells passing bars: {npass}/{ncell} "
              f"(harness verdict, not a gate failure)", flush=True)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
