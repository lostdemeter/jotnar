"""Teacher-lesion Phase-A gate (docs/DEBUGGER.md constitution): CPU-fast.

Heavy reruns are NOT routine: this gate checks banked-table
integrity (schema, 60+9+5 rows present, fires sane, checksums
pre==post), baseline-recompute-from-banked (mean AbsRel
re-derived from banked cells), delta arithmetic re-derived
in-gate (per-target % vs baseline; nominations == frozen <5%
criterion applied mechanically). A rerun path exists nowhere
here by design (GPU teacher runs are worker/integrator
territory with the banked evidence pasted).
Usage: python3 tests/test_teacher_lesion.py (CPU, seconds).
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FAIL = []
FIXDIR = os.path.join(ROOT, "samples", "marigold")
BAR_X = 5.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def cell_mean(frames):
    def vals(fr):
        return fr["strata"].values() if "strata" in fr else fr.values()
    vs = [v for fr in frames.values() for v in vals(fr) if v is not None]
    return sum(vs) / len(vs)


def main():
    base = json.load(open(os.path.join(FIXDIR, "teacher_baseline_A.json")))
    rep = json.load(open(os.path.join(FIXDIR, "teacher_lesion_A.json")))
    rows = rep.get("rows", [])

    # -- schema: 60 blocks + 9 vae stages + 5 families ----------------------
    blocks = [r for r in rows if r["target"].startswith("transformer_blocks.")]
    vae = [r for r in rows if r["target"].startswith("decoder.")
           or r["target"] == "post_quant_conv"]
    fams = [r for r in rows if r["target"].startswith("*")]
    check("tlesion-schema",
          len(blocks) == 60 and len(vae) == 9 and len(fams) == 5
          and all(set(r) >= {"target", "fires", "frames"} for r in rows),
          f"{len(blocks)} blocks + {len(vae)} vae + {len(fams)} fams")
    check("tlesion-fires",
          all(r["fires"].get(r["target"], 0) == 8 for r in blocks)
          and all(sum(r["fires"].values()) == 480 for r in fams),
          "blocks fired 8/8 frames; families 60x8")

    # -- baseline recompute + delta arithmetic re-derived --------------------
    bmean = cell_mean(base["frames"])
    check("tlesion-baseline",
          0 < bmean < 2.0 and len(base["frames"]) == 8,
          f"banked baseline mean AbsRel={bmean:.4f} over 8 frames")
    noms, table = [], {}
    for r in rows:
        m = cell_mean(r["frames"])
        d = 100.0 * (m - bmean) / bmean
        table[r["target"]] = (m, d)
        if d < BAR_X:
            noms.append(r["target"])
    check("tlesion-nominations",
          sorted(noms) == sorted(rep.get("nominations", [])),
          f"criterion delta<{BAR_X}% -> {len(noms)} nominations")
    worst = max(table.values(), key=lambda t: t[1])
    print(f"tlesion max delta: {max(table, key=lambda k: table[k][1])} "
          f"{worst[1]:+.2f}% (mean {worst[0]:.4f})", flush=True)
    best = min(table.values(), key=lambda t: t[1])
    print(f"tlesion min delta: {min(table, key=lambda k: table[k][1])} "
          f"{best[1]:+.2f}% (mean {best[0]:.4f})", flush=True)

    # -- checksums pre==post (read-only proof) ---------------------------------
    sha = rep.get("sha256", {})
    check("tlesion-checksums",
          sha.get("pre") and sha.get("pre") == sha.get("post")
          and len(sha["pre"]) == 7,
          "7 weight files, pre==post (nothing written)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
