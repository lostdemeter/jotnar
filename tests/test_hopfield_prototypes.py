"""Hopfield-prototypes gate (docs/HOPFIELD.md constitution): CPU-fast.

Heavy reruns are NOT routine: this gate checks banked-prototype
integrity (schema, frozen 3 points x 8 frames x 3072), banked
determinism evidence (same image twice bit-identical, exactly 0.0),
capacity-check legs (harvested dim vs banked math: 3072 >= 512 ->
PASS, N_exp recomputed from scripts/hopfield_capacity), checksums
pre==post (7 weight files, read-only proof), and value sanity
(finite, nonzero, correct shapes). A rerun path exists nowhere here
by design (GPU hook sessions are worker territory with pasted proof).
Usage: python3 tests/test_hopfield_prototypes.py (CPU, seconds).
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FAIL = []
FIXDIR = os.path.join(ROOT, "samples", "marigold")
BANK = os.path.join(FIXDIR, "hopfield_prototypes.json")
POINTS = ("transformer_blocks.29", "transformer_blocks.59", "norm_out")
FRAMES = ("ai_018_001_fr0000", "ai_018_001_fr0001", "ai_021_001_fr0000",
          "ai_021_001_fr0001", "ai_022_001_fr0000", "ai_022_001_fr0001",
          "ai_023_005_fr0000", "ai_023_005_fr0001")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    rep = json.load(open(BANK))
    check("hproto-schema",
          rep.get("points") == list(POINTS) and rep.get("frames") == list(FRAMES)
          and rep.get("dim") == 3072
          and set(rep.get("prototypes", {})) == set(POINTS)
          and all(set(rep["prototypes"][p]) == set(FRAMES) for p in POINTS),
          "3 frozen points x 8 frozen frames x 3072")
    ok = True
    for p in POINTS:
        for f in FRAMES:
            v = rep["prototypes"][p][f]
            if len(v) != 3072 or not all(isinstance(x, float) for x in v):
                ok = False
    import math
    finite = all(math.isfinite(x)
                 for p in POINTS for f in FRAMES
                 for x in rep["prototypes"][p][f])
    nzero = all(any(x != 0.0 for x in rep["prototypes"][p][f])
                for p in POINTS for f in FRAMES)
    check("hproto-values", ok and finite and nzero,
          "all float, finite, nonzero, len 3072")
    check("hproto-shapes",
          rep.get("shapes") == {p: [1, 1728, 3072] for p in POINTS},
          "captures [1,1728,3072] per point (mean-pooled over seq)")

    # -- determinism evidence (banked, bar exactly 0.0) -------------------
    det = rep.get("determinism_maxabs", {})
    check("hproto-determinism",
          set(det) == set(POINTS) and all(d == 0.0 for d in det.values()),
          f"frame0 x2 maxabs per point: {[det[p] for p in POINTS]}")

    # -- capacity check vs banked math --------------------------------------
    import hopfield_capacity as H
    cap = json.load(open(os.path.join(FIXDIR, "hopfield_capacity.json")))
    n_exp, _ = H.capacity_exp(3072)
    banked_cap = rep.get("capacity", {})
    check("hproto-capacity",
          rep["dim"] >= 512 and n_exp >= 80 and cap.get("K") == 20
          and set(banked_cap) == set(POINTS)
          and all(v.get("verdict") == "PASS" for v in banked_cap.values()),
          f"dim 3072>=512, N_exp(3072)={n_exp:.2g}>=80 (K=20): PASS x3")

    # -- checksums pre==post (read-only proof) -------------------------------
    sha = rep.get("sha256", {})
    check("hproto-checksums",
          sha.get("pre") and sha.get("pre") == sha.get("post")
          and len(sha["pre"]) == 7,
          "7 weight files, pre==post (nothing written)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
