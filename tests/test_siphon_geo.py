"""Siphon install IN-LISTING gate (teacher side, generic ball + program).

Builder program (scripts/siphon_geo.py): 28 Qwen2-7B layers plus the
yarn-ball body at L27 output (explicit country-row routing per
addr_sep). Ball data from yarnball_bank (keys x8, Paris value at
mirror-gain-1 x |x27|, 2 null stores, ledger). Two runs: zero-dose
base (geo-base for fork-correct hold grading -- geo FP16 base blanks
Germany/Italy where the mirror reads Berlin/Rome) then gain-1.
Gates: Germany blank(paris45)->' Paris'(paris1) INSTALL; Italy +
Japan bit-identical holds (Paris-ranks unchanged to the integer).
SKIPs without snapshot/GPU/nvcc. Slow (~50min: 2x mine+compile).
Usage: python3 tests/test_siphon_geo.py
"""
import json
import os
import shutil
import subprocess
import sys

FAIL = []
COUNTRIES = ["Germany", "Italy", "Japan"]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(root), "phi-core")))
    sys.path.insert(0, root)
    import torch
    from chain.qwen7b import SNAP, snapshot_ok
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        sys.exit(0)
    if shutil.which("nvcc") is None or not torch.cuda.is_available():
        print("SKIP (needs nvcc + GPU)")
        sys.exit(0)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(root), "phi-core")) \
        + ":" + root + ":" + env.get("PYTHONPATH", "")
    arms = {}
    for gain, tag, work in ((0.0, "base", "/tmp/gen7b_siphonT"),
                            (1.0, "ball", "/tmp/gen7b_siphonT1")):
        env["QWEN_WORKDIR"] = work
        out = f"/tmp/siphon_geoT_{tag}.json"
        r = subprocess.run(
            [sys.executable, os.path.join(root, "scripts", "siphon_geo.py"),
             "--gain", str(gain), "--smax", "8", "--out-json", out],
            capture_output=True, text=True, cwd=root, env=env, timeout=3600)
        if r.returncode != 0:
            check(f"siphon-geo-run-{tag}", False, r.stderr[-300:])
            print("FAILURES:", FAIL)
            sys.exit(1)
        arms[tag] = json.load(open(out))["arms"]
    g0, g1 = arms["base"], arms["ball"]
    check("siphon-geo-install",
          g1["Germany"]["paris_rank"] == 1,
          f"blank({g0['Germany']['paris_rank']})->"
          f"{g1['Germany']['top']!r}({g1['Germany']['paris_rank']})")
    for c in ("Italy", "Japan"):
        check(f"siphon-geo-hold-{c}",
              g1[c]["top"] == g0[c]["top"] and
              g1[c]["paris_rank"] == g0[c]["paris_rank"],
              f"{g0[c]}->{g1[c]} (bit-identical)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
