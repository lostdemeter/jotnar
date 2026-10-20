"""Training-harness gate (docs/TRAINING.md constitution): base contract +
backfill parity, CPU-fast, no full training runs.

Legs:
- determinism: backfilled trainer, same smallest config twice (CPU) ->
  metrics agree within 1e-6 (house float=eps bar; run-to-run CPU floor
  measured 2.4e-08, so the bar has 100x margin, not wallpaper).
- parity (transitive, stated): hand-rolled vs harness run measured
  max-abs-diff 1.25e-09 on the smallest config (paste in TRAINING.md);
  this gate banks the harness numbers below and fails on any drift
  (>1e-6 on any stratum) — hand == banked at backfill, banked == now
  in-gate, so science changes go red here.
- budget: the profiled shares table (parsed from docs/TRAINING.md, not
  restated) sums to ~100%; the cut slice measured live (feature-cache
  hit vs multiscale recompute on a real frame, hit must win big).
- base units: schedules/stride/cadence/cache/schema/timer (no subprocess).
- neighbors: backfilled module API preserved for its importers
  (train_refine_student uses load_frame + _define_model); the tracked
  banked metrics file is guarded + restored around every run (the only
  existing-results change allowed is none).
Usage: python3 tests/test_train_harness.py (CPU, seconds).
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []
BANKED = os.path.join(ROOT, "samples", "marigold", "attn_run.json")

# parity anchor: harness run --tag harness-base-new --steps 2 --width 8
# --blocks 1 --heads 1 (== hand-rolled run at max-abs-diff 1.25e-09;
# run-to-run CPU floor 2.4e-08; gate bar 1e-6).
ANCHOR = {
 "church": {"far": 0.45662815577209104, "near": 0.46097311024274645,
            "nonplanar": 0.4802704354932343, "planar": 0.4901228885447923,
            "smooth": 0.4605422536593082, "tex": 0.4500165292195822},
 "dogs": {"far": 0.4746984752134483, "near": 0.44106634260889716,
          "nonplanar": 0.39708760553583905, "planar": 0.5168612517752813,
          "smooth": 0.5422687224562206, "tex": 0.3818575261892547},
 "squirrel": {"far": 0.6397716859597515, "near": 0.5725682160768967,
              "nonplanar": 0.514499316384126, "planar": 0.6528977828173066,
              "smooth": 0.7122332967033539, "tex": 0.48715411941118525},
 "train": {"far": 0.5118659764331993, "near": 0.48631532030653646,
           "nonplanar": 0.5275407032924357, "planar": 0.47465707810429647,
           "smooth": 0.5443146552101543, "tex": 0.4678472736548855},
 "geo_webcam_0_combined": {"far": 0.2665270286700224,
                           "near": 0.2718371287928156,
                           "nonplanar": 0.32001508654847344,
                           "planar": 0.24506680594685998,
                           "smooth": 0.20014214730642818,
                           "tex": 0.34507113575523446},
 "geo_webcam_1_combined": {"far": 0.4738386866632039,
                           "near": 0.44717154124513403,
                           "nonplanar": 0.4912697832192319,
                           "planar": 0.4690683571770808,
                           "smooth": 0.47191558840703857,
                           "tex": 0.42592861341136096},
}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def run_small(tag):
    """Backfilled trainer, smallest config, CPU (hermetic, GPU stays free)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "scripts/train_attn_student.py", "--tag", tag,
         "--steps", "2", "--width", "8", "--blocks", "1", "--heads", "1"],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    assert r.returncode == 0, f"trainer rc={r.returncode}: {r.stderr[-500:]}"
    rep = json.load(open(BANKED))
    return rep, r.stdout


def maxdiff(a, b):
    mx = 0.0
    for f in a:
        for k in a[f]:
            mx = max(mx, abs(a[f][k] - b[f][k]))
    return mx


def main():
    import train_base as BASE

    guard = open(BANKED).read()
    try:
        # -- determinism: same config twice --------------------------------
        rep_a, _ = run_small("harness-gate-A")
        fa = {f: v["strata"] for f, v in rep_a["frames"].items()}
        rep_b, _ = run_small("harness-gate-B")
        fb = {f: v["strata"] for f, v in rep_b["frames"].items()}
        d = maxdiff(fa, fb)
        check("harness-determinism", d <= 1e-6,
              f"same config twice max-abs-diff={d:.2e} (bar 1e-6)")

        # -- parity anchor (transitive): banked == now ----------------------
        d = maxdiff(fa, ANCHOR)
        check("harness-parity", d <= 1e-6,
              f"banked-anchor max-abs-diff={d:.2e} (hand==banked 1.25e-09)")

        # -- base units ------------------------------------------------------
        import torch
        opt = torch.optim.Adam(
            [torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        import math
        ok = (BASE.apply_schedule(opt, "const", 0.01, 5, 100) == 0.01
              and abs(BASE.apply_schedule(opt, "cosine", 0.01, 50, 100)
                        - 0.01 * 0.5 * (1 + math.cos(math.pi / 2))) < 1e-12
              and abs(BASE.apply_schedule(opt, "linear", 0.01, 50, 100)
                        - 0.005) < 1e-12
              and abs(BASE.apply_schedule(opt, "warmcos", 0.01, 0, 100, 400)
                        - 0.01 / 400) < 1e-15)
        check("harness-schedules", ok, "const/cosine/linear/warmcos values")
        try:
            BASE.apply_schedule(opt, "nope", 0.01, 0, 1)
            check("harness-sched-loud", False, "unknown schedule must raise")
        except ValueError:
            check("harness-sched-loud", True, "unknown schedule raises")
        check("harness-stride",
              BASE.stride_index(20, 7, "cpu").tolist() == [0, 7, 14]
              and BASE.cadence(0, 100, 50) and BASE.cadence(99, 100, 50)
              and not BASE.cadence(7, 100, 50), "stride-7 + cadence rule")
        try:
            BASE.bank_metrics({"tag": "t"}, "/tmp/harness_nonexist.json")
            check("harness-schema-loud", False, "missing keys must raise")
        except ValueError:
            check("harness-schema-loud", True, "schema fail-loud")
        t = BASE.PhaseTimer()
        t.add("forward", 1.0)
        t.add("backward", 3.0)
        s = t.summary()
        check("harness-timer",
              abs(sum(v["share"] for v in s.values()) - 1.0) < 1e-12,
              "phase shares sum to 1")

        # -- budget: shares sum sanely (parsed from the spec, not restated) --
        spec = open(os.path.join(ROOT, "docs", "TRAINING.md")).read()
        shares = [float(x) for x in re.findall(r"(\d+\.\d)%", spec.split(
            "Instrumented shares", 1)[1].split("```", 2)[1])]
        check("harness-budget-sums", abs(sum(shares) - 100.0) < 1.0,
              f"profiled shares {shares} sum={sum(shares):.1f}%")

        # -- budget: the cut slice measured (cache hit vs recompute) ---------
        import time as _t
        import numpy as np
        import train_student as TS
        _, im3, _ = TS.load_frame("church")
        Hh, Ww, _ = im3.shape
        c = BASE.FeatureCache("/tmp/harness_cache", "test-mscale", 1, "33-3")
        t0 = _t.time()
        a = TS.multiscale_features(im3, 33, 3, False)
        t_re = _t.time() - t0
        t0 = _t.time()
        b = c.get("church", lambda: TS.multiscale_features(im3, 33, 3, False))
        t_miss = _t.time() - t0
        t0 = _t.time()
        b2 = c.get("church", lambda: TS.multiscale_features(im3, 33, 3, False))
        t_hit = _t.time() - t0
        check("harness-cut-cache",
              (a == b).all() and (a == b2).all() and t_hit * 5 < t_re,
              f"recompute={t_re:.2f}s miss={t_miss:.2f}s hit={t_hit:.3f}s "
              f"(same bytes, hit wins {t_re / max(t_hit, 1e-9):.0f}x)")

        # -- neighbors: importer API preserved --------------------------------
        import train_attn_student as JA
        check("harness-api",
              callable(JA.load_frame) and callable(JA._define_model),
              "load_frame + _define_model still importable")
        import train_refine_student as JR
        check("harness-refine-import", hasattr(JR, "main"),
              "refine (JA importer) still imports")
    finally:
        open(BANKED, "w").write(guard)
        for tag in ("harness-gate-A", "harness-gate-B"):
            import shutil
            shutil.rmtree(os.path.expanduser(f"~/student_runs/{tag}"),
                          ignore_errors=True)

    check("harness-guard-restored", open(BANKED).read() == guard,
          "tracked banked metrics byte-identical after runs")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
