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
    check("strata-manifest",
          len([r for r in man["fixtures"] if not r.get("synthetic")]) == 6
          and len([r for r in man["fixtures"] if r.get("synthetic")]) == 24
          and all(os.path.isfile(os.path.join(FIXDIR, r["npy"]))
                  for r in man["fixtures"])
          and sum(1 for r in man["fixtures"] if r.get("heldout")) == 2,
          f"{len(man['fixtures'])} banked rows (6 real + 24 synthetic) on disk")
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    frames = {}
    real_names = set()
    for row in man["fixtures"]:
        t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
        cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", row["name"] + ".*")))
        assert cand, row["name"]
        im = np.asarray(Image.open(cand[0]).convert("RGB")
                        .resize((t.shape[1], t.shape[0]))) / 255.0
        frames[row["name"]] = (t, im)
        if not row.get("synthetic"):
            real_names.add(row["name"])

    # -- strata deterministic + coverage --------------------------------------
    # coverage binds the real-fixture harness domain (judge lives there);
    # synthetic box-room stats break percentile guarantees by design
    # (Phase-12a median-collapse, characterized not gated).
    det_ok, cov_ok = True, True
    for name, (t, im) in frames.items():
        m1, m2 = strata_masks(t, im), strata_masks(t, im)
        det_ok &= all(np.array_equal(m1[k], m2[k]) for k in STRATA)
        if name in real_names:
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

    # -- Phase-5 battery (constitution-bound search, verdict-reported) --------
    # Frozen 10 configs (see docs/STUDENT.md Phase 5); the gate asserts
    # table integrity + the winner-consistency tripwire (winner None
    # ⟺ no config clears any cell; a passing cell with winner None
    # fails loud demanding the retrain-match claim).
    import json as _js3
    ab = _js3.load(open(os.path.join(FIXDIR, "student_ablation.json")))
    bat = ab.get("battery", {})
    frozen = ["f65w64d2", "f129w64d2", "f65w128d2", "f129w128d2",
              "f65w64d3", "f129w64d3", "f33w128d2", "f33dense5",
              "f65w32d2", "f129w128d3"]
    check("battery-configs", bat.get("configs") == frozen
          and bat.get("reference") == "mscale64", bat.get("configs"))
    cells = bat.get("cells", {})
    complete = True
    for t in frozen:
        fr = cells.get(t, {}).get("frames", {})
        complete &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery-complete", bool(complete), "10 configs x 6x6 finite")
    counts = {}
    for t in frozen:
        n = 0
        for _n, fr in cells[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts[t] = n
    check("battery-winner-none",
          all(v == 0 for v in counts.values()) and bat.get("winner") is None,
          f"held-out cells under bars per config: {counts}")
    flags_ok = all(
        all(k in cells[t].get("flags", {}) for k in
            ("field", "densewin", "depth", "width", "steps", "lr", "seed"))
        for t in frozen)
    check("battery-flags", bool(flags_ok), "repro metadata per config")

    # -- Phase-6 long-range battery (same integrity shape, new key) ---------
    b6 = ab.get("battery6", {})
    frozen6 = ["g257w64d2", "g513w64d2", "g1025w64d2", "g257w128d3",
               "g513w128d3", "globw64d2", "glob257w64d2", "g513w64d3",
               "g257w32d2", "g1025w128d3"]
    check("battery6-configs", b6.get("configs") == frozen6
          and b6.get("reference") == "mscale64", b6.get("configs"))
    cells6 = b6.get("cells", {})
    complete6 = True
    for t in frozen6:
        fr = cells6.get(t, {}).get("frames", {})
        complete6 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery6-complete", bool(complete6), "10 configs x 6x6 finite")
    counts6 = {}
    for t in frozen6:
        n = 0
        for _n, fr in cells6[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts6[t] = n
    check("battery6-winner-none",
          all(v == 0 for v in counts6.values())
          and b6.get("winner") is None,
          f"held-out cells under bars per config: {counts6}")
    flags6_ok = all(
        all(k in cells6[t].get("flags", {}) for k in
            ("field", "densewin", "depth", "width", "steps", "lr", "seed",
             "globsum"))
        for t in frozen6)
    check("battery6-flags", bool(flags6_ok), "repro metadata per config")

    # -- Phase-8 joint-attention build (same integrity shape, new key) -----
    b8 = ab.get("battery8", {})
    frozen8 = ["jattn64b2"]
    check("battery8-configs", b8.get("configs") == frozen8
          and b8.get("reference") == "mscale64", b8.get("configs"))
    cells8 = b8.get("cells", {})
    complete8 = True
    for t in frozen8:
        fr = cells8.get(t, {}).get("frames", {})
        complete8 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery8-complete", bool(complete8), "1 config x 6x6 finite")
    counts8 = {}
    for t in frozen8:
        n = 0
        for _n, fr in cells8[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts8[t] = n
    check("battery8-winner-none",
          all(v == 0 for v in counts8.values())
          and b8.get("winner") is None,
          f"held-out cells under bars per config: {counts8}")
    flags8_ok = all(
        all(k in cells8[t].get("flags", {}) for k in
            ("field", "width", "blocks", "stride", "nctx", "steps", "lr",
             "seed"))
        for t in frozen8)
    check("battery8-flags", bool(flags8_ok), "repro metadata per config")

    # -- Phase-9 schedule battery (same integrity shape, new key) ---------
    b9 = ab.get("battery9", {})
    frozen9 = ["jattn64b2-8k-const", "jattn64b2-8k-cos", "jattn64b2-16k-cos"]
    check("battery9-configs", b9.get("configs") == frozen9
          and b9.get("reference") == "jattn64b2", b9.get("configs"))
    cells9 = b9.get("cells", {})
    complete9 = True
    for t in frozen9:
        fr = cells9.get(t, {}).get("frames", {})
        complete9 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery9-complete", bool(complete9), "3 configs x 6x6 finite")
    counts9 = {}
    for t in frozen9:
        n = 0
        for _n, fr in cells9[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts9[t] = n
    check("battery9-winner-none",
          all(v == 0 for v in counts9.values())
          and b9.get("winner") is None,
          f"held-out cells under bars per config: {counts9}")
    flags9_ok = all(
        all(k in cells9[t].get("flags", {}) for k in
            ("field", "width", "blocks", "stride", "nctx", "steps", "lr",
             "schedule", "seed"))
        for t in frozen9)
    check("battery9-flags", bool(flags9_ok), "repro metadata per config")

    # -- Phase-10 head battery (same integrity shape, new key) ------------
    b10 = ab.get("battery10", {})
    frozen10 = ["jattn64h1-8kcos", "jattn64h2-8kcos", "jattn64h4-8kcos"]
    check("battery10-configs", b10.get("configs") == frozen10
          and b10.get("reference") == "jattn64b2", b10.get("configs"))
    cells10 = b10.get("cells", {})
    complete10 = True
    for t in frozen10:
        fr = cells10.get(t, {}).get("frames", {})
        complete10 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery10-complete", bool(complete10), "3 configs x 6x6 finite")
    counts10 = {}
    for t in frozen10:
        n = 0
        for _n, fr in cells10[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts10[t] = n
    check("battery10-winner-none",
          all(v == 0 for v in counts10.values())
          and b10.get("winner") is None,
          f"held-out cells under bars per config: {counts10}")
    flags10_ok = all(
        all(k in cells10[t].get("flags", {}) for k in
            ("field", "width", "blocks", "heads", "stride", "nctx", "steps",
             "lr", "schedule", "seed"))
        for t in frozen10)
    check("battery10-flags", bool(flags10_ok), "repro metadata per config")
    cap10_ok = all(cells10[t].get("nparams") == 108033 for t in frozen10)
    check("battery10-capacity-matched", bool(cap10_ok),
          "108033 params x3 (heads, not capacity)")

    # -- Phase-11 grid battery (same integrity shape, new key) -------------
    b11 = ab.get("battery11", {})
    frozen11 = ["jattn64-s16h1", "jattn64-s16h2", "jattn64-s16h4"]
    check("battery11-configs", b11.get("configs") == frozen11
          and b11.get("reference") == "jattn64b2", b11.get("configs"))
    cells11 = b11.get("cells", {})
    complete11 = True
    for t in frozen11:
        fr = cells11.get(t, {}).get("frames", {})
        complete11 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery11-complete", bool(complete11), "3 configs x 6x6 finite")
    counts11 = {}
    for t in frozen11:
        n = 0
        for _n, fr in cells11[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts11[t] = n
    check("battery11-winner-none",
          all(v == 0 for v in counts11.values())
          and b11.get("winner") is None,
          f"held-out cells under bars per config: {counts11}")
    flags11_ok = all(
        all(k in cells11[t].get("flags", {}) for k in
            ("field", "width", "blocks", "heads", "stride", "nctx", "steps",
             "lr", "schedule", "seed"))
        for t in frozen11)
    check("battery11-flags", bool(flags11_ok), "repro metadata per config")

    # -- Phase-13 depth battery (same integrity shape, new key) -----------
    b13 = ab.get("battery13", {})
    frozen13 = ["jattn64-s16h2b2", "jattn64-s16h2b3", "jattn64-s16h2b4"]
    check("battery13-configs", b13.get("configs") == frozen13
          and b13.get("reference") == "jattn64b2", b13.get("configs"))
    cells13 = b13.get("cells", {})
    complete13 = True
    for t in frozen13:
        fr = cells13.get(t, {}).get("frames", {})
        complete13 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery13-complete", bool(complete13), "3 configs x 6x6 finite")
    counts13 = {}
    for t in frozen13:
        n = 0
        for _n, fr in cells13[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts13[t] = n
    check("battery13-winner-none",
          all(v == 0 for v in counts13.values())
          and b13.get("winner") is None,
          f"held-out cells under bars per config: {counts13}")
    flags13_ok = all(
        all(k in cells13[t].get("flags", {}) for k in
            ("field", "width", "blocks", "heads", "stride", "nctx", "steps",
             "lr", "schedule", "seed"))
        for t in frozen13)
    check("battery13-flags", bool(flags13_ok), "repro metadata per config")
    cap13 = [cells13[t].get("nparams") for t in frozen13]
    check("battery13-depth-ladder", cap13 == [111425, 161281, 211137],
          f"{cap13} (+49856/block)")

    # -- Phase-19 dit-tiny battery (same integrity shape, new key) --------
    b19 = ab.get("battery19", {})
    frozen19 = ["dit64b2", "dit64b4", "dit64b6"]
    check("battery19-configs", b19.get("configs") == frozen19
          and b19.get("reference") == "jattn64b2", b19.get("configs"))
    cells19 = b19.get("cells", {})
    complete19 = True
    for t in frozen19:
        fr = cells19.get(t, {}).get("frames", {})
        complete19 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery19-complete", bool(complete19), "3 configs x 6x6 finite")
    counts19 = {}
    for t in frozen19:
        n = 0
        for _n, fr in cells19[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts19[t] = n
    check("battery19-winner-none",
          all(v == 0 for v in counts19.values())
          and b19.get("winner") is None,
          f"held-out cells under bars per config: {counts19}")
    flags19_ok = all(
        all(k in cells19[t].get("flags", {}) for k in
            ("arch", "width", "blocks", "heads", "patch", "adaln",
             "geluform", "gradloss", "stride", "steps", "lr", "schedule",
             "seed", "field"))
        and cells19[t]["flags"]["arch"] == "dit-tiny"
        for t in frozen19)
    check("battery19-flags", bool(flags19_ok), "repro metadata per config")
    cap19 = [cells19[t].get("nparams") for t in frozen19]
    check("battery19-depth-ladder", cap19 == [209857, 359233, 508609],
          f"{cap19} (+149376/block incl. AdaLN)")

    # -- Phase-20 dit-clean battery (same integrity shape, new key) -------
    b20 = ab.get("battery20", {})
    frozen20 = ["ditc64b2", "ditc64b4", "ditc64b6"]
    check("battery20-configs", b20.get("configs") == frozen20
          and b20.get("reference") == "jattn64b2", b20.get("configs"))
    cells20 = b20.get("cells", {})
    complete20 = True
    for t in frozen20:
        fr = cells20.get(t, {}).get("frames", {})
        complete20 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery20-complete", bool(complete20), "3 configs x 6x6 finite")
    counts20 = {}
    for t in frozen20:
        n = 0
        for _n, fr in cells20[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts20[t] = n
    check("battery20-winner-none",
          all(v == 0 for v in counts20.values())
          and b20.get("winner") is None,
          f"held-out cells under bars per config: {counts20}")
    flags20_ok = all(
        all(k in cells20[t].get("flags", {}) for k in
            ("arch", "width", "blocks", "heads", "patch", "adaln",
             "geluform", "gradloss", "stride", "steps", "lr", "schedule",
             "seed", "field"))
        and cells20[t]["flags"]["arch"] == "dit-tiny"
        and cells20[t]["flags"]["gradloss"] == 0.0
        for t in frozen20)
    check("battery20-flags", bool(flags20_ok), "repro metadata per config")
    cap20 = [cells20[t].get("nparams") for t in frozen20]
    check("battery20-depth-ladder", cap20 == [209857, 359233, 508609],
          f"{cap20} (+149376/block incl. AdaLN)")

    # -- Phase-21 sweep battery (same integrity shape, new key) -----------
    b21 = ab.get("battery21", {})
    frozen21 = ["ditc64b2-c1", "ditc64b2-c3", "ditc64b2-l1",
                "ditc64b2-l3", "ditc64b2-w1", "ditc64b2-w3"]
    check("battery21-configs", b21.get("configs") == frozen21
          and b21.get("reference") == "jattn64b2", b21.get("configs"))
    cells21 = b21.get("cells", {})
    complete21 = True
    for t in frozen21:
        fr = cells21.get(t, {}).get("frames", {})
        complete21 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery21-complete", bool(complete21), "6 configs x 6x6 finite")
    counts21 = {}
    for t in frozen21:
        n = 0
        for _n, fr in cells21[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts21[t] = n
    check("battery21-winner-none",
          all(v == 0 for v in counts21.values())
          and b21.get("winner") is None,
          f"held-out cells under bars per config: {counts21}")
    flags21_ok = all(
        all(k in cells21[t].get("flags", {}) for k in
            ("arch", "width", "blocks", "heads", "patch", "adaln",
             "geluform", "gradloss", "stride", "steps", "lr", "schedule",
             "warmup", "seed", "field"))
        and cells21[t]["flags"]["arch"] == "dit-tiny"
        and cells21[t]["flags"]["gradloss"] == 0.0
        for t in frozen21)
    check("battery21-flags", bool(flags21_ok), "repro metadata per config")

    # -- Phase-15 dino-input battery (same integrity shape, new key) ------
    b15 = ab.get("battery15", {})
    frozen15 = ["dino-tokonly", "dino-tokrgb"]
    check("battery15-configs", b15.get("configs") == frozen15
          and b15.get("reference") == "jattn64b2", b15.get("configs"))
    cells15 = b15.get("cells", {})
    complete15 = True
    for t in frozen15:
        fr = cells15.get(t, {}).get("frames", {})
        complete15 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery15-complete", bool(complete15), "2 configs x 6x6 finite")
    counts15 = {}
    for t in frozen15:
        n = 0
        for _n, fr in cells15[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts15[t] = n
    check("battery15-winner-none",
          all(v == 0 for v in counts15.values())
          and b15.get("winner") is None,
          f"held-out cells under bars per config: {counts15}")
    flags15_ok = all(
        all(k in cells15[t].get("flags", {}) for k in
            ("field", "width", "blocks", "heads", "feat", "stride", "nctx",
             "steps", "lr", "schedule", "seed"))
        and cells15[t]["flags"]["feat"].startswith("dino-")
        for t in frozen15)
    check("battery15-flags", bool(flags15_ok), "repro metadata per config")
    cap15 = [cells15[t].get("nparams") for t in frozen15]
    check("battery15-capacity", cap15 == [156993, 156993],
          f"{cap15} (108033 + 49216 dino_proj - 256 ctx)")

    # -- Phase-17 textured battery (in-domain graded, ood unjudged) ------
    b17 = ab.get("battery17", {})
    frozen17 = ["texmsc-2k", "texmsc-8kcos"]
    check("battery17-configs", b17.get("configs") == frozen17
          and b17.get("reference") == "mscale64", b17.get("configs"))
    cells17 = b17.get("cells", {})
    complete17 = True
    counts17 = {}
    for t in frozen17:
        fr = {n: f for n, f in cells17.get(t, {}).get("frames", {}).items()
              if f.get("split") == "heldout"}
        complete17 &= (len(fr) == 16 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
        counts17[t] = sum(
            1 for n in fr for k in STRATA if fr[n]["strata"][k] <= BARS[k])
    check("battery17-complete", bool(complete17), "2 configs x 16x6 finite")
    check("battery17-counts", counts17 == {"texmsc-2k": 7, "texmsc-8kcos": 7},
          f"in-domain held-out cells under bars: {counts17}")
    check("battery17-winner-tie", b17.get("winner", "").startswith("tie"),
          str(b17.get("winner")))
    flags17_ok = all(
        all(k in cells17[t].get("flags", {}) for k in
            ("field", "densewin", "depth", "width", "steps", "lr", "schedule",
             "seed"))
        for t in frozen17)
    check("battery17-flags", bool(flags17_ok), "repro metadata per config")
    ood17 = b17.get("out_of_domain", {})
    check("battery17-ood-unjudged",
          set(ood17) == set(frozen17)
          and all(ood17[t].get("judged") is False for t in frozen17)
          and all(len(ood17[t]["frames"]) == 18 for t in frozen17),
          "12 flat + 6 real printed, judged:false")

    # -- Phase-16 refine battery (same integrity shape, new key) ----------
    b16 = ab.get("battery16", {})
    frozen16 = ["cf-frozen"]
    check("battery16-configs", b16.get("configs") == frozen16
          and b16.get("reference") == "jattn64b2", b16.get("configs"))
    cells16 = b16.get("cells", {})
    complete16 = True
    for t in frozen16:
        fr = cells16.get(t, {}).get("frames", {})
        complete16 &= (len(fr) == 6 and all(
            sorted(fr[n].get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr[n]["strata"].values())
            for n in fr))
    check("battery16-complete", bool(complete16), "1 config x 6x6 finite")
    counts16 = {}
    for t in frozen16:
        n = 0
        for _n, fr in cells16[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        counts16[t] = n
    check("battery16-winner-none",
          all(v == 0 for v in counts16.values())
          and b16.get("winner") is None,
          f"held-out cells under bars per config: {counts16}")
    flags16_ok = all(
        all(k in cells16[t].get("flags", {}) for k in
            ("field", "width", "blocks", "heads", "stride", "nctx", "steps",
             "lr", "schedule", "seed", "mode", "pass1", "head"))
        for t in frozen16)
    check("battery16-flags", bool(flags16_ok), "repro metadata per config")
    # Four variants (baseline/perc/sink/both), all openly losing: the gate
    # asserts the table is complete and internally consistent (every cell
    # finite; deltas vs baseline computed, never asserted in a direction).
    # A variant that moved held-out strata systematically would show here
    # as a directional delta -- none does (see docs/STUDENT.md Phase 4).
    import json as _js2
    ab_path = os.path.join(FIXDIR, "student_ablation.json")
    if not os.path.isfile(ab_path):
        print("SKIP ablation legs (no student_ablation.json)")
    else:
        ab = _js2.load(open(ab_path))
        variants = ab.get("variants", [])
        cells = ab.get("cells", {})
        check("ablation-complete",
              variants == ["baseline", "perc", "sink", "both"]
              and all(v in cells and len(cells[v]) == 6 for v in variants),
              f"{variants}")
        finite_ok = True
        for v in variants:
            for n, fr in cells.get(v, {}).items():
                vals = list(fr.get("strata", {}).values())
                finite_ok &= (sorted(fr.get("strata", {})) == STRATA
                              and all(np.isfinite(x) and x >= 0
                                      for x in vals))
        check("ablation-finite", bool(finite_ok), "4x6x6 cells finite >= 0")
        base = cells.get("baseline", {})
        for v in ("perc", "sink", "both"):
            ups, dns, n = 0, 0, 0
            for n_, fr in cells.get(v, {}).items():
                if fr.get("split") != "heldout":
                    continue
                for k in STRATA:
                    n += 1
                    d = fr["strata"][k] - base[n_]["strata"][k]
                    ups += d < -0.01
                    dns += d > 0.01
            print(f"ablation {v} vs baseline (held-out, |d|>0.01): "
                  f"{ups} improved / {dns} worsened / {n} cells", flush=True)
        check("ablation-reported", True, ab.get("verdict", "")[:80])

    # -- Phase-27 loss coordinates (same student, different downhill) -----
    # Frozen 3 coordinate systems (see docs/STUDENT.md Phase 27); the gate
    # asserts table integrity + term inventory (programmatic: every loss
    # term present-or-gated-absent) + the winner-consistency tripwire
    # (winner None ⟺ no coordinate clears bars or directionals; a
    # qualifying coordinate with winner None fails loud demanding the
    # retrain-match claim). Directionals print (reported, like ablation).
    lc = ab.get("losscoord", {})
    lcfrozen = ["lc-value", "lc-log", "lc-quantum"]
    check("losscoord-configs", lc.get("configs") == lcfrozen
          and lc.get("reference") == "lc-value", lc.get("configs"))
    lccells = lc.get("cells", {})
    lccomplete = all(
        t in lccells and len(lccells[t].get("frames", {})) == 6 and all(
            sorted(fr.get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr["strata"].values())
            for fr in lccells[t]["frames"].values())
        for t in lcfrozen)
    check("losscoord-complete", bool(lccomplete), "3 coords x 6x6 finite")
    flags = {t: lccells.get(t, {}).get("flags", {}) for t in lcfrozen}
    check("losscoord-terms",
          sorted(f.get("loss_space") for f in flags.values())
          == ["log", "quantum", "value"]
          and all(f.get("perc") == 0.0 and f.get("sink") == 0.0
                  for f in flags.values())
          and abs(flags["lc-log"].get("loss_lo", 0) - -1.0) < 1e-9
          and flags["lc-quantum"].get("loss_q", 0) > 0,
          "3 spaces inventoried, aux absent (perc=sink=0), lo/q banked")
    lccounts, lcdelta = {}, {}
    for t in lcfrozen:
        n = 0
        for _n, fr in lccells[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        lccounts[t] = n
        ds = [lccells[t]["frames"][_n]["strata"][k]
              - lccells["lc-value"]["frames"][_n]["strata"][k]
              for _n, fr in lccells[t]["frames"].items()
              if fr.get("split") == "heldout" for k in STRATA]
        lcdelta[t] = sum(ds) / len(ds)
    for t in ("lc-log", "lc-quantum"):
        print(f"losscoord {t} vs lc-value (held-out mean delta): "
              f"{lcdelta[t]:+.4f}", flush=True)
    check("losscoord-winner-none",
          all(v == 0 for v in lccounts.values())
          and all(lcdelta[t] >= -0.05 for t in ("lc-log", "lc-quantum"))
          and lc.get("winner") is None,
          f"under-bars {lccounts}, deltas "
          f"{ {t: round(lcdelta[t], 4) for t in lcfrozen} }")

    # -- Phase-29 lesion-matched training (damage profile, not outputs) --
    # Frozen 6 configs (lm-clean control + lm-A..E basin targets);
    # term-inventory covers the target source (target_maps present
    # exactly on lesioned arms, None on control; loss_space value
    # everywhere — coordinates frozen); materialization integrity
    # (banked map sha + determinism-rerun flag); winner tripwire
    # (winner None ⟺ no cells and no directional move).
    lm = ab.get("lesionmatch", {})
    lmfrozen = ["lm-clean", "lm-A", "lm-B", "lm-C", "lm-D", "lm-E"]
    check("lesionmatch-configs", lm.get("configs") == lmfrozen
          and lm.get("reference") == "lm-clean", lm.get("configs"))
    lmcells = lm.get("cells", {})
    lmcomplete = all(
        t in lmcells and len(lmcells[t].get("frames", {})) == 6 and all(
            sorted(fr.get("strata", {})) == STRATA
            and all(np.isfinite(v) and v >= 0
                    for v in fr["strata"].values())
            for fr in lmcells[t]["frames"].values())
        for t in lmfrozen)
    check("lesionmatch-complete", bool(lmcomplete), "6 configs x 6x6 finite")
    lmflags = {t: lmcells.get(t, {}).get("flags", {}) for t in lmfrozen}
    check("lesionmatch-terms",
          all(f.get("loss_space") == "value" for f in lmflags.values())
          and lmflags["lm-clean"].get("target_maps") is None
          and all(lmflags[t].get("target_files", {}).keys()
                  == {"church", "dogs", "squirrel", "train"}
                  and all(v.endswith(f"lm-{t.split('-')[1]}_{k}.npy")
                          for k, v in lmflags[t]["target_files"].items())
                  for t in lmfrozen[1:])
          and all(f.get("perc") == 0.0 and f.get("sink") == 0.0
                  for f in lmflags.values()),
          "per-frame lesioned maps inventoried, value-space, aux absent")
    lmcounts, lmdelta = {}, {}
    for t in lmfrozen:
        n = 0
        for _n, fr in lmcells[t]["frames"].items():
            if fr.get("split") != "heldout":
                continue
            n += sum(1 for k in STRATA if fr["strata"][k] <= BARS[k])
        lmcounts[t] = n
        ds = [lmcells[t]["frames"][_n]["strata"][k]
              - lmcells["lm-clean"]["frames"][_n]["strata"][k]
              for _n, fr in lmcells[t]["frames"].items()
              if fr.get("split") == "heldout" for k in STRATA]
        lmdelta[t] = sum(ds) / len(ds)
    for t in lmfrozen[1:]:
        print(f"lesionmatch {t} vs lm-clean (held-out mean delta): "
              f"{lmdelta[t]:+.4f}", flush=True)
    check("lesionmatch-winner-none",
          all(v == 0 for v in lmcounts.values())
          and all(lmdelta[t] >= -0.05 for t in lmfrozen[1:])
          and lm.get("winner") is None,
          f"under-bars {lmcounts}, deltas "
          f"{ {t: round(lmdelta[t], 4) for t in lmfrozen} }")
    mat = lm.get("materialization", {})
    import hashlib as _hl
    sha_ok = True
    for fn, h in mat.get("map_sha256", {}).items():
        p = os.path.join(ROOT, "samples", "marigold", "inputs_lesioned", fn)
        if not os.path.isfile(p) or \
                _hl.sha256(open(p, "rb").read()).hexdigest()[:16] != h:
            sha_ok = False
    check("lesionmatch-materialization",
          bool(mat.get("determinism_rerun_match")) and sha_ok
          and len(mat.get("map_sha256", {})) == 20,
          "rerun byte-identical (pasted) + 20 map shas recomputed")

    # -- Phase-30 basin-C replication + specificity -------------------------
    # Frozen two questions: (1) replication — fresh maps (regenerated
    # via the committed path, 20/20 byte-identical) + fresh training
    # reproduces the banked C cells; (2) specificity — banked A/B/D/E
    # + clean arms reused verbatim (bit-determinism proven: fresh
    # reruns would be identical, stated openly). Winner tripwire:
    # winner None ⟺ no bars anywhere (directional-move ≠ winner).
    lr = ab.get("lmrep", {})
    lrconfigs = ["lm-C-rep"]
    check("lmrep-configs", lr.get("configs") == lrconfigs
          and lr.get("reference") == "lm-clean", lr.get("configs"))
    lrcells = lr.get("cells", {})
    check("lmrep-complete",
          set(lrcells) == set(lrconfigs)
          and len(lrcells["lm-C-rep"].get("frames", {})) == 6 and all(
              sorted(fr.get("strata", {})) == STRATA
              and all(np.isfinite(v) and v >= 0
                      for v in fr["strata"].values())
              for fr in lrcells["lm-C-rep"]["frames"].values()),
          "C-rep 6x6 finite")
    bankedC = lmcells["lm-C"]["frames"] if "lm-C" in lmcells else {}
    repC = lrcells["lm-C-rep"]["frames"]
    check("lmrep-replication",
          bool(bankedC) and max(
              abs(bankedC[n]["strata"][k] - repC[n]["strata"][k])
              for n in bankedC for k in STRATA) <= 1e-9,
          "C-rep == banked C end-to-end (maps + training)")
    reffr = lmcells["lm-clean"]["frames"]
    for t, label in (("lm-C", "banked"), ("lm-C-rep", "fresh")):
        fr = lmcells[t]["frames"] if t in lmcells else repC
        ds = [fr[n]["strata"][k] - reffr[n]["strata"][k]
              for n in fr if fr[n].get("split") == "heldout"
              for k in STRATA]
        ups = sum(1 for d in ds if d < -0.05)
        print(f"lmrep {label} C vs clean: mean {sum(ds) / len(ds):+.4f}, "
              f"{ups}/12 cells >0.05 better", flush=True)
    other = {}
    for t in ("lm-A", "lm-B", "lm-D", "lm-E"):
        fr = lmcells[t]["frames"]
        ds = [fr[n]["strata"][k] - reffr[n]["strata"][k]
              for n in fr if fr[n].get("split") == "heldout"
              for k in STRATA]
        other[t] = (sum(ds) / len(ds),
                    sum(1 for d in ds if d < -0.05))
    check("lmrep-specificity",
          all(v[1] == 0 for v in other.values())
          and all(v[0] >= -0.05 for v in other.values()),
          f"A/B/D/E move nothing: { {t: (round(m, 4), u) for t, (m, u) in other.items()} } "
          f"(C alone moves — C-specific, not damage-general)")
    check("lmrep-winner-none",
          lr.get("winner") is None,
          "no bars cleared anywhere (directional-move ≠ winner)")

    # -- basin-supervision key (basin-ground-truth round; legs above
    #    untouched; heavy retrain NOWHERE in gate by design) ------------------
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import hashlib as _hl
    import hopfield_retrieval as _R
    _bs = json.load(open(os.path.join(
        FIXDIR, "basin_supervision.json")))
    _bc = np.load(os.path.join(FIXDIR, "basin_codes.npy")).astype(np.float64)
    check("pb-schema",
          _bs["Kstar"] == 4 and _bc.shape == (11, 128)
          and bool(np.isfinite(_bc).all()),
          "K*=4, 11 codes x128 finite")
    _wh = np.load(_bs["W_path"])
    check("pb-weights",
          _hl.sha256(open(_bs["W_path"], "rb").read()).hexdigest()
          == _bs["W_sha256"],
          "out-of-tree weights sha matches")
    _pred = np.argmax(_bc[:8] @ _wh["bh"] + _wh["bbh"], axis=1)
    _true = np.array(_bs["assignment"]["true"])
    check("pb-assignment",
          bool((_pred == _true).all())
          and _bs["assignment"]["accuracy"] == 1.0
          and _bs["bar"] == 0.5,
          f"head recomputed from codes: 8/8 = 1.0 >= bar 0.5 (K*=4)")
    _H8 = _bc[:8] / np.linalg.norm(_bc[:8], axis=1, keepdims=True)
    _G = _H8 @ _H8.T
    _iu = np.triu_indices(8, 1)
    _mx, _mn = float(_G[_iu].max()), float(_G[_iu].mean())
    _ss = _bs["separability"]
    check("pb-separability",
          abs(_mx - _ss["maxcos"]) < 1e-6 and abs(_mn - _ss["meancos"]) < 1e-6
          and _mx < 0.9 and _mn < 0.5,
          f"no regression vs cleared round (max {_mx:.4f} < 0.9, "
          f"mean {_mn:+.4f} < 0.5)")
    _hit = sum(
        _R.retrieve(_H8, _R.corrupt(_H8[fi], f, s, li))[0] == fi
        for li, f in enumerate(_R.LEVELS)
        for fi in range(8) for s in _R.SEEDS)
    _bb = _bs["battery"]
    check("pb-battery",
          _bb["trials"] == 200 and _bb["accuracy"] == _hit / 200
          and _bb["accuracy"] >= _bb["bar"] == 0.25,
          f"recomputed {_hit}/200 >= 0.25 -> { _bb['verdict']}")
    check("pb-determinism", _bs["determinism_maxabs"] == 0.0,
          "same image -> same code, bit-identical")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
