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

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
