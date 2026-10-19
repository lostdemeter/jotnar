"""Patch-shuffle probe gate (student Phase 14): structure dependence.

SKIP-clean without assets/GPU (gold stays green machineless):
shuffle determinism (same seed identical incl. index map;
roundtrip unshuffle(shuffle(x))==x; tile-multiset preserved),
banked integrity (schema, rel-deltas finite >= 0, controls small),
verdict legs recomputed from banked numbers (teacher/dav2 ratio,
cross-system spread — reported with the frozen rule, never
asserted beyond it), mscale64 live spot recompute (1 fixture,
1 patch, CPU) vs banked. Teacher/DAV2 reruns are NOT routine
(their preds live in /tmp at measure time; banked scalars rule).
Usage: python3 tests/test_shuffle_probe.py (~1 min with assets).
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FAIL = []
FIXDIR = os.path.join(ROOT, "samples", "marigold")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import numpy as np
    from shuffle_probe import shuffle_rgb, unshuffle, base_images

    # -- determinism legs (always) --------------------------------------
    base = base_images()
    det_ok, rt_ok, ms_ok = True, True, True
    for p in (16, 64):
        a, ia = shuffle_rgb(base["dogs"], p)
        b, ib = shuffle_rgb(base["dogs"], p)
        det_ok &= bool((a == b).all()) and bool((ia == ib).all())
        c, _ = shuffle_rgb(base["dogs"], p, seed=8)
        det_ok &= bool(not (a == c).all())
        rt_ok &= bool((unshuffle(a, ia) == base["dogs"]).all())
        ms_ok &= (sorted(a.reshape(-1, 3).tobytes()) == sorted(
            base["dogs"].reshape(-1, 3).tobytes()))
    check("shuffle-deterministic", bool(det_ok), "seed 7 stable, seed 8 differs")
    check("shuffle-roundtrip", bool(rt_ok), "unshuffle(shuffle(x))==x")
    check("shuffle-multiset", bool(ms_ok), "tiles preserved, only relocated")

    bank_path = os.path.join(FIXDIR, "shuffle_probe.json")
    if not os.path.isfile(bank_path):
        print("SKIP banked legs (no shuffle_probe.json yet)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)
    bank = json.load(open(bank_path))
    check("shuffle-banked-schema",
          bank.get("patch") == [16, 64] and bank.get("seed") == 7
          and set(bank.get("systems", {})) == {"teacher", "dav2", "mscale64", "jattn64b2"},
          "2 sizes + 4 systems")
    fin_ok = True
    for sys_name, sd in bank["systems"].items():
        for p in (16, 64):
            cells = sd["deltas"][f"p{p}"]
            fin_ok &= (len(cells) == 6 and all(
                len(fr) == 6 and all(
                    isinstance(c, dict) and np.isfinite(c["rel"]) and c["rel"] >= 0
                    and np.isfinite(c["raw"]) and c["raw"] >= 0
                    for c in fr.values())
                for fr in cells.values()))
    check("shuffle-banked-finite", bool(fin_ok), "4x2x6x6 raw+rel finite >= 0")
    check("shuffle-controls",
          bank.get("controls", {}).get("dav2_nn_roundtrip", 9) < 0.1
          and bank.get("teacher_runs", {}).get("p16_twice_identical") is True,
          "nn-roundtrip small, teacher twice-identical")

    # -- verdict legs (recomputed from banked; frozen rule) --------------
    def mean_rel(sys_name, p):
        vals = [c["rel"] for fr in bank["systems"][sys_name]["deltas"][f"p{p}"].values()
                for c in fr.values()]
        return float(np.mean(vals))

    t16, d16 = mean_rel("teacher", 16), mean_rel("dav2", 16)
    ratio = t16 / d16
    check("shuffle-ratio-measured", True, f"teacher/dav2 @p16 = {ratio:.2f} (rule needs >=3)")
    spreads = {}
    for p in (16, 64):
        ms = [mean_rel(s, p) for s in ("teacher", "dav2", "mscale64", "jattn64b2")]
        spreads[p] = max(ms) / min(ms)
    check("shuffle-spread-measured", True,
          f"max/min p16={spreads[16]:.2f} p64={spreads[64]:.2f} (rule needs <=2)")
    m16a = mean_rel("mscale64", 16)
    m64a = mean_rel("mscale64", 64)
    check("shuffle-locality-signature", True,
          f"mscale64 p16={m16a:.2f} p64={m64a:.2f} (halving = 33px field inside 64px patches)")

    # -- mscale64 live spot (CPU) -----------------------------------------
    try:
        import torch
        import train_student as T
        sd = torch.load(os.path.expanduser("~/student_runs/mscale64/weights.pt"),
                        map_location="cpu", weights_only=True)
        mnet = torch.nn.Sequential(
            torch.nn.Linear(177, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1)).float().eval()
        mnet.load_state_dict(sd)
        import test_strata_marigold as H
        im = base["church"]
        sh, sidx = shuffle_rgb(im, 16)
        with torch.no_grad():
            def pred(x):
                return mnet(torch.tensor(
                    T.multiscale_features(x.astype(np.float64) / 255.0, 33, 3, False),
                    dtype=torch.float32)).numpy().reshape(672, 1024)
            ri, back = pred(im), unshuffle(pred(sh), sidx)
        t = np.load(os.path.join(FIXDIR, "church.npy")).astype(np.float64)
        masks = H.strata_masks(t, im.astype(np.float64) / 255.0)
        worst = 0.0
        for k in H.STRATA:
            got = float(H.align_mae(back, ri, masks[k]))
            want = bank["systems"]["mscale64"]["deltas"]["p16"]["church"][k]["raw"]
            worst = max(worst, abs(got - want) / max(want, 1e-300))
        check("shuffle-mscale64-spot", worst < 1e-9, f"worst rel dev {worst:.1e}")
    except (ImportError, OSError, FileNotFoundError) as e:
        print(f"SKIP shuffle-mscale64-spot ({e})")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
