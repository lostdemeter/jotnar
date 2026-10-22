"""Option-scan gate (docs/OPTION_SCAN.md constitution): CPU-only, seconds.

Legs (every status claim machine-checked — branches, gates, banked
tables, never memory):
- temporal-unrun: no temporal/video-student branch, gate, or banked
  table; machinery (chain/temporal.py) + data (consecutive shard
  frames) + /dev/video0 present.
- arrangement-never-run: no arrange branch/gate/table in student
  scope (the VELOCITY hit is teacher-RLM, out of scope by shape).
- ordinal-unrun: no pairwise/rank/ordinal terms in trainers.
- scale-unswept: shard frame count far below 10M.
- faithful-partial: B2 banked (dithyp64b2), B4/B6 artifacts absent.
- textured-signal: texmsc banks present (run-with-signal).
- constraints-current: B1/B2/B3/A5/A6 cards + sizes in PATTERN_GAPS.
- ranking-complete: every frozen candidate id appears in the ranked
  spec section (no unranked stragglers).
- no-implementation tripwire: named component files absent
  (scoping discipline — trainers/emitters/programs unadded).
Usage: python3 tests/test_option_scan.py
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []
CANDS = ["temporal-video", "ordinal-currency", "textured-real",
         "btrack-adaptivity", "faithful-ladder-hypersim", "scale-10M",
         "a5a6-perf", "arrangement-battery"]
SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=ROOT, shell=True).stdout


def main():
    branches = sh("git branch -a")
    tests = " ".join(os.path.basename(p)
                     for p in glob.glob(os.path.join(ROOT, "tests",
                                                     "test_*.py")))
    trains = " ".join(os.path.basename(p)
                      for p in glob.glob(os.path.join(ROOT, "scripts",
                                                      "train_*.py")))
    marig = " ".join(os.path.basename(p)
                     for p in glob.glob(os.path.join(ROOT, "samples",
                                                     "marigold", "*.json")))

    # -- temporal-video: RAN empty since the scan (update, not rewrite) -----
    # The scan correctly reported unrun at its base; the temporal round
    # has since run to an empty verdict (+0.0110 coin-flip, control-compared).
    # Temporal extended the hypersim gate in place (no dedicated gate file
    # by design) — so "ran" here means branch + trainer + banked cells.
    import json as _js
    _tab = {}
    try:
        _ab = _js.load(open(os.path.join(
            ROOT, "samples", "marigold", "student_ablation.json")))
        _tt = _ab.get("temporal", {})
        for _k in ("cells", "winner"):
            if _k in _tt:
                _tab[_k] = _tt[_k]
    except (OSError, ValueError):
        pass
    _tcells = _tab.get("cells", {})
    check("scan-temporal-ran",
          "temporal" in branches
          and "temporal" in trains
          and set(_tcells) >= {"temp-pair", "temp-single"},
          "branch + trainer + banked cells exist (ran since scan)")
    check("scan-temporal-verdict-empty",
          _tab.get("winner") is None
          and len(_tcells) == 2,
          "banked winner None (empty verdict, coin-flip delta in spec)")
    check("scan-temporal-apparatus",
          os.path.isfile(os.path.join(ROOT, "chain", "temporal.py"))
          and len(glob.glob(os.path.join(
              SHARD, "ai_001_010", "rgb_cam_00_fr*.png"))) >= 10
          and os.path.exists("/dev/video0"),
          "machinery + consecutive frames + webcam in hand")

    # -- arrangement: never-run verified ------------------------------------
    check("scan-arrangement-never-run",
          "arrange" not in branches and "arrange" not in tests
          and "arrange" not in trains
          and not [m for m in marig.split() if "arrange" in m],
          "no branch/gate/trainer/table")

    # -- ordinal: unrun ------------------------------------------------------
    hits = []
    for p in glob.glob(os.path.join(ROOT, "scripts", "train_*.py")):
        src = open(p).read()
        if re.search(r"pairwise|ordinal|rank.?loss|argsort", src):
            hits.append(os.path.basename(p))
    check("scan-ordinal-unrun", hits == [], f"trainers clean {hits}")

    # -- scale 10M+: unswept --------------------------------------------------
    nframes = len(glob.glob(os.path.join(SHARD, "*", "rgb_cam_00_fr*.png")))
    check("scan-scale-unswept", 0 < nframes < 100000,
          f"shard frames={nframes} (10M+ nowhere)")

    # -- faithful ladder: B2 run, B4/B6 absent ---------------------------------
    check("scan-faithful-partial",
          "attn_run_dithyp64b2.json" in marig
          and "attn_run_dithyp64b4.json" not in marig
          and "attn_run_dithyp64b6.json" not in marig,
          "B2 banked, B4/B6 artifacts absent")

    # -- textured: run-with-signal ----------------------------------------------
    check("scan-textured-signal",
          "student_run_texmsc-2k.json" in marig
          and "student_run_texmsc-8kcos.json" in marig,
          "texmsc banks present")

    # -- constraints: prices current --------------------------------------------
    pg = open(os.path.join(ROOT, "docs", "PATTERN_GAPS.md")).read()
    check("scan-constraints-current",
          all(f"### {c}" in pg for c in
              ("A5 —", "A6 —", "B1 —", "B2 —", "B3 —"))
          and "- Size: M (advisor surgery" in pg
          and "- Size: L (protocol design" in pg
          and "- Size: S (static) / M (dynamic" in pg,
          "A5/A6/B1/B2/B3 cards + sizes intact")

    # -- ranking completeness: every frozen id ranked in spec --------------------
    spec = open(os.path.join(ROOT, "docs", "OPTION_SCAN.md")).read()
    ranked = spec.split("## Ranked backlog", 1)[1] \
        if "## Ranked backlog" in spec else ""
    missing = [c for c in CANDS if c not in ranked]
    check("scan-ranking-complete", missing == [],
          f"unranked stragglers: {missing}")

    # -- no-implementation tripwire ----------------------------------------------
    # NOTE (post-main-merge update): scripts/train_temporal_student.py
    # exists via the merged temporal round (legitimate main-side work,
    # not this branch) — removed from this list with reason recorded.
    absent = ["scripts/train_video_student.py",
              "scripts/train_ordinal_student.py",
              "scripts/train_arrange_student.py",
              "tests/test_temporal_student.py",
              "programs/temporal_student.py"]
    present = [p for p in absent
               if os.path.exists(os.path.join(ROOT, p))]
    check("scan-no-impl", present == [], f"unadded: {present}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
