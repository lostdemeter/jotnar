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

    # -- Holography subset-curve legs (banked integrity + verdict) -------
    # Same integrity shape (not a fork): 15 subset rows with frozen
    # subsets, linear-fit R2 + cliff + seed-agreement recomputed,
    # verdict equality. Heavy reruns NOT routine.
    HP = os.path.join(FIXDIR, "teacher_lesion_H.json")
    HB = os.path.join(FIXDIR, "teacher_baseline_H.json")
    if not (os.path.isfile(HP) and os.path.isfile(HB)):
        print("SKIP holography legs (no banked H battery yet)")
    else:
        import random as _rd
        import numpy as _np
        hrep = json.load(open(HP))
        hbase = json.load(open(HB))
        hrows = hrep.get("rows", [])
        check("holo-schema",
              len(hrows) == 15
              and all(set(r) >= {"target", "fires", "frames", "subset"}
                      for r in hrows)
              and all(len(r["frames"]) == 8 for r in hrows),
              "15 subset rows x 8 frames with frozen subsets")
        check("holo-subsets",
              all(sorted(r["subset"])
                  == sorted(_rd.Random(
                      28000 + int(r["target"].split("_")[1]) * 100
                      + int(r["target"].split("_")[2])).sample(
                          range(60), int(r["target"].split("_")[1])))
                  for r in hrows),
              "banked subsets == frozen RNG scheme re-derived")
        check("holo-finite",
              all(v is not None and _np.isfinite(v)
                  for r in hrows for fr in r["frames"].values()
                  for v in fr.values()),
              "0 nulls, all finite")

        def _cm(frames):
            vs = []
            for fr in frames.values():
                d = fr["strata"] if "strata" in fr else fr
                vs += [v for v in d.values() if v is not None]
            return sum(vs) / len(vs)

        hbmean = _cm(hbase["frames"])
        hmeans = [hbmean]
        hagree = {}
        for s in (5, 10, 20, 30, 40):
            vs = [_cm(next(r for r in hrows
                           if r["target"] == f"subset_{s}_{i}")["frames"])
                  for i in (0, 1, 2)]
            hmeans.append(sum(vs) / 3)
            hagree[s] = max(vs) / min(vs)
        xs = _np.array([0, 5, 10, 20, 30, 40], dtype=float)
        ys = _np.array(hmeans)
        A = _np.stack([xs, _np.ones_like(xs)], 1)
        (ha, hb2), _, _, _ = _np.linalg.lstsq(A, ys, rcond=None)
        pred = ha * xs + hb2
        r2 = 1 - float(((ys - pred) ** 2).sum()) / float(
            ((ys - ys.mean()) ** 2).sum())
        steps = [hmeans[i + 1] / hmeans[i]
                 for i in range(len(hmeans) - 1)]
        check("holo-linear",
              abs(r2 - hrep.get("linear", {}).get("r2", -1)) < 1e-9,
              f"R2={r2:.4f} (banked match; holographic needs >0.8)")
        check("holo-cliff",
              abs(max(steps) - hrep.get("max_step", -1)) < 1e-9
              and max(steps) > 2.0,
              f"max step x{max(steps):.2f} (>2x = cliff)")
        check("holo-seeds",
              all(v < 1.5 for v in hagree.values()),
              f"seed max/min per size "
              f"{ {s: round(v, 3) for s, v in hagree.items()} } (<1.5)")
        rv = ("holographic"
              if (r2 > 0.8 and max(steps) <= 2.0) else "modular")
        check("holo-verdict",
              hrep.get("verdict", {}).get("name") == rv,
              f"banked={hrep.get('verdict', {}).get('name')} "
              f"recomputed={rv}")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
