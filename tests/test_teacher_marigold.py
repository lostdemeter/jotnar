"""Marigold v2 teacher gate (student Phase 0): fixtures exist, behave.

SKIP-clean without the sibling checkout, checkpoints, or GPU (gold
stays green machineless): checkpoint presence (Qwen-Image-Edit-2509
+ depth/Log-stage2 + depth embeddings), banked fixture manifest
(samples/marigold/teacher_manifest.json) with files on disk,
seed-determinism leg (same --seed twice on one example ->
byte-identical npy; reruns the teacher CLI, needs GPU + weights),
output shape/dtype legs on the banked .npy files (no GPU needed).
The seeded-vs-unseeded difference (VAE sampling) is documented in
docs/STUDENT.md -- bit-exact is the wrong bar unseeded, the right
bar seeded; this gate pins the seeded half.
Usage: python3 tests/test_teacher_marigold.py (seconds when skipped).
"""
import json
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []

SIBLING = os.path.normpath(os.path.join(ROOT, "..", "marigold-v2"))
FIXDIR = os.path.join(ROOT, "samples", "marigold")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def have_gpu():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def main():
    if not os.path.isdir(SIBLING):
        print("SKIP (no marigold-v2 sibling checkout)")
        return
    qw = os.path.join(SIBLING, "assets", "checkpoints", "Qwen-Image-Edit-2509")
    st2 = os.path.join(SIBLING, "assets", "checkpoints", "Marigold-V2",
                       "depth", "Log-stage2")
    emb = os.path.join(SIBLING, "assets", "checkpoints", "Marigold-V2",
                       "qwen_text_embeddings",
                       "qwen_edit_2509_qwen_depth_realimg512_prompt_embeds.pt")
    check("teacher-qwen-present", os.path.isdir(qw), qw)
    check("teacher-stage2-present", os.path.isdir(st2), st2)
    check("teacher-embeds-present", os.path.isfile(emb), emb)
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    man_path = os.path.join(FIXDIR, "teacher_manifest.json")
    if not os.path.isfile(man_path):
        print("SKIP (no banked fixtures yet -- see docs/STUDENT.md Phase 0)")
        return
    man = json.load(open(man_path))
    ok = True
    for row in man["fixtures"]:
        p = os.path.join(FIXDIR, row["npy"])
        if not (os.path.isfile(p) and os.path.getsize(p) > 0):
            ok = False
    check("teacher-fixtures-present", ok,
          f"{len(man['fixtures'])} rows in manifest")

    import numpy as np
    for row in man["fixtures"]:
        a = np.load(os.path.join(FIXDIR, row["npy"]))
        check(f"teacher-shape-{row['name']}",
              tuple(a.shape) == tuple(row["shape"]) and str(a.dtype) == row["dtype"],
              f"{a.shape} {a.dtype}")
    if FAIL:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    # -- seed-determinism leg (needs GPU + weights; reruns teacher CLI) -----
    if not have_gpu():
        print("SKIP teacher-seed-deterministic (no GPU here)")
    else:
        venv_py = os.path.expanduser("~/marigold-env/bin/python")
        if not os.path.isfile(venv_py):
            print("SKIP teacher-seed-deterministic (no ~/marigold-env)")
        else:
            _determinism(venv_py, man)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)




def _determinism(venv_py, man):
    import glob as _g
    import shutil as _sh
    seed = str(man.get("seed", 2025))
    # Sanitized env: our cuda_env.sh exports (PYTHONPATH/LD_LIBRARY_PATH
    # into the user-local stack) poison the teacher venv with mixed trees
    # (measured: subprocess fails under the sourced env, passes clean).
    # Pass through only what's needed; drop the rest.
    keep = {}
    for k in ("HOME", "USER", "LOGNAME", "TMPDIR", "CUDA_VISIBLE_DEVICES",
              "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
              "HF_HUB_OFFLINE", "http_proxy", "https_proxy", "HTTP_PROXY",
              "HTTPS_PROXY", "no_proxy"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep["PATH"] = (os.path.dirname(venv_py) + ":/usr/bin:/bin")
    outs = []
    try:
        for i in range(2):
            od = f"/tmp/opencode/mg_det{i}"
            _sh.rmtree(od, ignore_errors=True)
            r = subprocess.run(
                [venv_py, os.path.join(SIBLING, "scripts", "infer.py"),
                 "--image_dir", os.path.join(FIXDIR, "inputs"),
                 "--output_dir", od, "--seed", seed,
                 "--width", "1024", "--height", "672"],
                capture_output=True, text=True, cwd=SIBLING, timeout=1800,
                env=keep)
            if r.returncode != 0:
                check("teacher-seed-deterministic", False,
                      (r.stderr or r.stdout)[-300:])
                break
            outs.append(od)
        if len(outs) == 2:
            a = sorted(_g.glob(outs[0] + "/images/predictions_npy/*.npy"))
            b = sorted(_g.glob(outs[1] + "/images/predictions_npy/*.npy"))
            same = (len(a) == len(b) == len(man["fixtures"]) and all(
                open(x, "rb").read() == open(y, "rb").read()
                for x, y in zip(a, b)))
            check("teacher-seed-deterministic", same,
                  f"{len(a)} vs {len(b)} npy files, byte-identical")
    finally:
        for od in outs:
            _sh.rmtree(od, ignore_errors=True)


if __name__ == "__main__":
    main()
