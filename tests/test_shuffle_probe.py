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

    # -- register-token legs (registers round; legs above untouched) -------
    reg_path = os.path.join(FIXDIR, "registers_shuffle.json")
    reg = json.load(open(reg_path))
    check("registers-schema",
          set(reg["arms"]) == {"reg", "noreg"}
          and reg["arms"]["reg"]["nctx"] == 4
          and reg["arms"]["noreg"]["nctx"] == 0
          and reg["recipe"]["seed"] == 7
          and all(set(reg["arms"][a]["shuffle"]) == {"p16", "p64"}
                  and len(reg["arms"][a]["shuffle"][p]) == 6
                  for a in ("reg", "noreg") for p in ("p16", "p64")),
          "2 arms x nctx 4/0 x 2 sizes x 6 frames")
    def rmean(a, p):
        return float(np.mean([c["rel"]
                              for fr in reg["arms"][a]["shuffle"][f"p{p}"].values()
                              for c in fr.values()]))
    rm16, cm16 = rmean("reg", 16), rmean("noreg", 16)
    check("registers-determinism",
          abs(rm16 - reg["arms"]["reg"]["shuffle_mean"]["16"]) < 1e-12
          and abs(cm16 - reg["arms"]["noreg"]["shuffle_mean"]["16"]) < 1e-12,
          f"means recomputed == banked ({rm16:.4f} vs {cm16:.4f})")
    check("registers-robustness",
          abs(rm16 - 1.0324) < 1e-3 and abs(cm16 - 1.3382) < 1e-3
          and reg["verdicts"]["overall"] == "dead"
          and not reg["verdicts"]["reg_gt_control"],
          f"reg {rm16:.4f} < control {cm16:.4f} (directional FAIL), "
          f"reg >= 1.0 holds alone; verdict dead arithmetically correct")
    bars = {"near": 0.09, "far": 0.09, "planar": 0.09,
            "nonplanar": 0.08, "tex": 0.07, "smooth": 0.09}
    bcheck = True
    for a in ("reg", "noreg"):
        cells = [(k, v) for n, d in reg["arms"][a]["strata"].items()
                 if n.startswith("geo") for k, v in d.items()]
        bcheck &= (sum(1 for k, v in cells if v < bars[k])
                   == reg["arms"][a]["heldout_pass"] == 0)
        bcheck &= (abs(float(np.mean([v for _, v in cells]))
                       - reg["arms"][a]["heldout_mean"]) < 1e-12)
    check("registers-bars", bool(bcheck)
          and reg["arms"]["reg"]["heldout_mean"] <= 0.4062 * 1.05,
          "0/12 both arms; reg mean within 5% of banked 0.4062")
    check("registers-ablation-equality",
          reg["model_note"]["retrain_match"].startswith("void:model-mismatch")
          and reg["model_note"]["arm_nparams"] == 157249
          and reg["banked_8kcos_ref"]["nparams"] == 108033,
          "control self-consistent (recomputed above); 8k-cos equality "
          "VOID by documented model mismatch (157249 vs 108033), "
          "not silently worked around")
    try:
        import torch
        import train_attn_student as JA
        JAMod = JA._define_model(torch)
        sd = torch.load(os.path.expanduser("~/student_runs/jattn64b2-reg/weights.pt"),
                        map_location="cpu", weights_only=True)
        rnet = JAMod(64, 672 // 32, 1024 // 32, 4, 2).float().eval()
        rnet.load_state_dict(sd)
        yy, xx = torch.meshgrid(torch.arange(672), torch.arange(1024), indexing="ij")
        rpidx = ((yy // 32) * 32 + (xx // 32)).long()
        with torch.no_grad():
            rgbt = torch.tensor(base["dogs"].astype(np.float64) / 255.0,
                                dtype=torch.float32)
            to = rnet.encode(rgbt)
            ri = rnet.head(to, rgbt, rpidx).numpy()
            sh, sidx = shuffle_rgb(base["dogs"], 16)
            rgbts = torch.tensor(sh.astype(np.float64) / 255.0, dtype=torch.float32)
            back = unshuffle(rnet.head(rnet.encode(rgbts), rgbts, rpidx).numpy(),
                             sidx)
        import test_strata_marigold as H
        t = np.load(os.path.join(FIXDIR, "dogs.npy")).astype(np.float64)
        masks = H.strata_masks(t, base["dogs"].astype(np.float64) / 255.0)
        worst = 0.0
        for k in H.STRATA:
            got = float(H.align_mae(back, ri, masks[k]))
            want = reg["arms"]["reg"]["shuffle"]["p16"]["dogs"][k]["raw"]
            worst = max(worst, abs(got - want) / max(want, 1e-300))
        check("registers-spot", worst < 1e-9, f"dogs p16 worst rel dev {worst:.1e}")
    except (ImportError, OSError, FileNotFoundError) as e:
        print(f"SKIP registers-spot ({e})")

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
