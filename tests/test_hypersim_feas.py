"""Hypersim feasibility gate (student Phase 22): CPU-only, fast.

Disk/state legs (shard present, out-of-tree), GT-audit
determinism (1 frame recompute vs banked), strata-transfer legs
(coverage counts recomputed), re-derived bars integrity (scaffold
minima/5 recomputed), GO/NO-GO verdict leg reading the banked
audit (frozen rule, reported never asserted beyond it).
SKIP-clean without downloads (gold stays green).
Usage: python3 tests/test_hypersim_feas.py (seconds when skipped).
"""
import glob
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FAIL = []
SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")
BANK = os.path.join(ROOT, "samples", "marigold", "hypersim_audit.json")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import numpy as np
    from PIL import Image

    have = os.path.isdir(SHARD) and len(
        glob.glob(os.path.join(SHARD, "*", "depth_plane_cam_00_fr*.png"))) > 0
    check("hypersim-present", True,
          f"extract dir {'present' if have else 'absent'} "
          f"(out-of-tree, documented)")
    if not os.path.isfile(BANK):
        print("SKIP banked legs (no hypersim_audit.json yet)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)
    bank = json.load(open(BANK))
    check("hypersim-banked-schema",
          len(bank.get("frames", [])) == 24
          and all(k in bank for k in ("coverage_30_70_counts",
                                      "scaffold_minima", "bars_fresh_div5",
                                      "verdict")),
          "24 frames + coverage + scaffold + bars + verdict")
    if not have:
        print("SKIP recompute legs (no downloads here)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    fr0 = bank["frames"][0]
    scene, fname = fr0["name"].split("/")
    rp = os.path.join(SHARD, scene, fname)
    dp = os.path.join(SHARD, scene,
                      fname.replace("rgb_cam_00", "depth_plane_cam_00"))
    rgb = np.asarray(Image.open(rp).convert("RGB")).astype(np.float64) / 255.0
    depth = np.asarray(Image.open(dp)).astype(np.float64) / 1000.0
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.0)
    check("hypersim-determinism",
          abs((1.0 - valid.mean()) - fr0["invalid_rate"]) < 1e-12
          and abs(float(np.median(depth[valid])) - fr0["depth_p50"]) < 1e-9,
          f"{fr0['name']} recompute matches")
    masks = H.strata_masks(depth, rgb)
    cov = {k: float(masks[k].mean()) for k in H.STRATA}
    c = float(depth[valid].mean())
    worst = 0.0
    want_sc = bank["frames"][0]["scaffold"]
    for k in H.STRATA:
        m = masks[k]
        ok30 = 0.30 < cov[k] < 0.70
        worst = max(worst, 0.0 if ok30 else 1.0)
        if int(m.sum()) == 0:
            continue
        A = np.stack([np.full(m.sum(), c), np.ones(m.sum())], 1)
        (a, b), _, _, _ = np.linalg.lstsq(A, depth[m].ravel(), rcond=None)
        got = float(np.abs(a * c + b - depth[m]).mean())
        want = want_sc[k]
        if want is not None:
            worst = max(worst, abs(got - want) / max(want, 1e-300))
    check("hypersim-strata-recompute", worst < 1e-9,
          "coverage + per-frame scaffold reproduce")
    v = bank["verdict"]
    go = (v["invalid_mean"] < 0.05 and v["invalid_worst"] < 0.20
          and v["coverage_ok_rules"] >= 5 and v["bars_computable"])
    check("hypersim-verdict", go == v["go"],
          f"frozen rule recomputed: {'GO' if go else 'NO-GO'}")

    # -- scale-training verdict legs (banked integrity + winner) --------
    # No retraining here: single-seed result (determinism NOT
    # re-verified — stated openly, unlike retrain-matched batteries).
    ab = json.load(open(os.path.join(ROOT, "samples", "marigold",
                                     "student_ablation.json")))
    if "hyper" not in ab:
        print("SKIP hyper legs (no banked hyper battery yet)")
    else:
        hb, hc = ab["hyper"], ab["hyper"]["cells"]
        check("hyper-configs",
              hb.get("configs") == ["hyper-mscale", "hyper-jattn"]
              and hb.get("reference") == "mscale64",
              "2 banked archs, no new shapes")
        BARS = {"near": 0.09, "far": 0.09, "planar": 0.09,
                "nonplanar": 0.08, "tex": 0.07, "smooth": 0.09}
        complete, counts = True, {}
        for t in hb["configs"]:
            fr = {n: f for n, f in hc[t]["frames"].items()
                  if f.get("split") == "heldout"}
            complete &= (len(fr) == 60 and all(
                len(fr[n].get("strata", {})) == 6
                and all(v is None or (np.isfinite(v) and v >= 0)
                        for v in fr[n]["strata"].values())
                for n in fr))
            counts[t] = sum(
                1 for n in fr for k in BARS
                if fr[n]["strata"][k] is not None
                and fr[n]["strata"][k] <= BARS[k])
        check("hyper-complete", bool(complete), "2x60x6 finite-or-null")
        check("hyper-winner-none",
              all(v == 0 for v in counts.values())
              and hb.get("winner") is None,
              f"held-out cells under fresh bars: {counts}")
        check("hyper-determinism-note", True,
              "single-seed, retrain NOT verified (stated, not green-washed)")
        tr = hb.get("transfer", {})
        check("hyper-transfer-unjudged",
              set(tr) == {"hyper-mscale", "hyper-jattn"}
              and all(tr[t].get("judged") is False for t in tr),
              "transfer reported-not-graded (incl. grid-lock note)")

    # -- faithful-on-Hypersim verdict legs (banked integrity + winner) --
    # No retraining here: single-seed result (determinism NOT
    # re-verified — stated openly, hyper precedent). Same integrity
    # shape as the hyper legs above (not a fork): 60 heldout frames x
    # 6 strata finite-or-null, winner tripwire vs Phase-22 fresh bars,
    # transfer reported-not-graded.
    if "dithyp" not in ab:
        print("SKIP dithyp legs (no banked faithful-on-Hypersim battery yet)")
    else:
        dh, dc = ab["dithyp"], ab["dithyp"]["cells"]
        check("dithyp-configs",
              dh.get("configs") == ["dithyp64b2"]
              and dh.get("reference") == "ditc64b2",
              "1 banked arch (data swap only), no new shapes")
        BARS_FRESH = {"near": 0.068, "far": 0.019, "planar": 0.046,
                      "nonplanar": 0.123, "tex": 0.132, "smooth": 0.117}
        dcomplete, dcounts = True, {}
        for t in dh["configs"]:
            fr = {n: f for n, f in dc[t]["frames"].items()
                  if f.get("split") == "heldout"}
            dcomplete &= (len(fr) == 60 and all(
                len(fr[n].get("strata", {})) == 6
                and all(v is None or (np.isfinite(v) and v >= 0)
                        for v in fr[n]["strata"].values())
                for n in fr))
            dcounts[t] = sum(
                1 for n in fr for k in BARS_FRESH
                if fr[n]["strata"][k] is not None
                and fr[n]["strata"][k] <= BARS_FRESH[k])
        check("dithyp-complete", bool(dcomplete), "1x60x6 finite-or-null")
        check("dithyp-winner-none",
              all(v == 0 for v in dcounts.values())
              and dh.get("winner") is None,
              f"held-out cells under fresh bars: {dcounts}")
        check("dithyp-determinism-note", True,
              "single-seed, retrain NOT verified (stated, not green-washed)")
        dtr = dh.get("transfer", {})
        check("dithyp-transfer-unjudged",
              set(dtr) == {"fixtures"}
              and all(dtr[t].get("judged") is False for t in dtr)
              and "grid-lock" in dtr["fixtures"].get("note", ""),
              "transfer reported-not-graded (grid-lock note)")

    # -- Phase-25 telemetry legs (banked integrity + frozen verdicts) --
    # Reads banked telemetry only (no GPU): schema, smoke-match, and a
    # from-scratch recompute of the frozen interpretation rules (the
    # banked verdict field must equal the recompute — integrity).
    TP = os.path.join(ROOT, "samples", "marigold",
                      "dithyp64b2_telemetry.json")
    if not os.path.isfile(TP):
        print("SKIP telemetry legs (no banked Phase-25 telemetry yet)")
    else:
        tel = json.load(open(TP))
        smps = tel.get("samples", [])
        ok_schema = (
            tel.get("base_tag") == "dithyp64b2"
            and tel.get("instrument", {}).get("every") == 500
            and len(tel.get("loss_trace", [])) == 8000
            and len(smps) == 17
            and all(set(s) >= {"it", "loss", "blocks"} for s in smps)
            and all(set(b) >= {"act_in", "act_out", "grad", "upd",
                               "mod"} for s in smps for b in s["blocks"])
            and all(set(b["mod"]) >= {"sh1", "sc1", "g1", "sh2",
                                      "sc2", "g2"} for s in smps
                    for b in s["blocks"]))
        check("telemetry-schema", ok_schema,
              f"{len(smps)} samples, trace {len(tel.get('loss_trace', []))}")
        sm = tel.get("smoke", {})
        check("telemetry-smoke",
              bool(sm.get("ok")) and sm.get("maxdiff", 1) <= 1e-4
              and abs(sm.get("banked_0", 0) - 0.4014) < 1e-4,
              f"hooks-delta maxdiff={sm.get('maxdiff'):.2e} (tol 1e-4)")
        ratios = [s["blocks"][0]["grad"] / s["blocks"][1]["grad"]
                  for s in smps if s["blocks"][1]["grad"] > 0]
        att = sum(ratios) / len(ratios)
        allg = [s["blocks"][b]["grad"] for s in smps for b in (0, 1)]
        health = max(allg) / min(allg)
        endvars = [s["blocks"][b]["mod"][c]["var"]
                   for s in smps[-1:] for b in (0, 1)
                   for c in ("sh1", "sc1", "g1", "sh2", "sc2", "g2")]
        firstvars = [s["blocks"][b]["mod"][c]["var"]
                     for s in smps[:1] for b in (0, 1)
                     for c in ("sh1", "sc1", "g1", "sh2", "sc2", "g2")]
        sab = (all(v < 1e-6 for v in endvars)
               and all(f0 > 100 * fe for f0, fe in zip(firstvars,
                                                       endvars)))
        hu = tel.get("eval_summary", {}).get("heldout_under")
        name = ("attenuation" if att > 10.0 else
                "self-sabotage" if sab else
                "landscape" if (0 < health < 10.0
                                and all(v >= 1e-6 for v in endvars)
                                and hu == 0)
                else "none-of-the-above")
        check("telemetry-verdict",
              tel.get("verdict", {}).get("name") == name
              and abs(tel["verdict"]["numbers"]["att_ratio"] - att) < 1e-9,
              f"banked={tel.get('verdict', {}).get('name')} "
              f"recomputed={name} (att={att:.3f} health={health:.1f})")
        check("telemetry-overhead",
              0 <= tel.get("overhead", {}).get("frac", 1) < 0.10,
              f"hook overhead frac={tel.get('overhead', {}).get('frac'):.4f}")

    # -- Phase-26 alignment-surgery legs (banked integrity + verdicts) --
    # Same integrity shape (not a fork): 60 heldout frames x 6 strata
    # finite-or-null, winner tripwire vs Phase-22 fresh bars, transfer
    # reported-not-graded. Plus scale-boundedness legs on banked
    # telemetry (Phase-25 blowup numbers as explicit comparison) and
    # the round-verdict recompute (bounded+descended =
    # confirmed-progressing, else deeper-pathology).
    if "ditshift" not in ab:
        print("SKIP ditshift legs (no banked surgery battery yet)")
    else:
        ds, dsc = ab["ditshift"], ab["ditshift"]["cells"]
        check("ditshift-configs",
              ds.get("configs") == ["ditshift64b2"]
              and ds.get("reference") == "dithyp64b2",
              "1 banked arch (loss line only), no new shapes")
        BARS_FRESH = {"near": 0.068, "far": 0.019, "planar": 0.046,
                      "nonplanar": 0.123, "tex": 0.132, "smooth": 0.117}
        scomplete, scounts = True, {}
        for t in ds["configs"]:
            fr = {n: f for n, f in dsc[t]["frames"].items()
                  if f.get("split") == "heldout"}
            scomplete &= (len(fr) == 60 and all(
                len(fr[n].get("strata", {})) == 6
                and all(v is None or (np.isfinite(v) and v >= 0)
                        for v in fr[n]["strata"].values())
                for n in fr))
            scounts[t] = sum(
                1 for n in fr for k in BARS_FRESH
                if fr[n]["strata"][k] is not None
                and fr[n]["strata"][k] <= BARS_FRESH[k])
        check("ditshift-complete", bool(scomplete), "1x60x6 finite-or-null")
        check("ditshift-winner-none",
              all(v == 0 for v in scounts.values())
              and ds.get("winner") is None,
              f"held-out cells under fresh bars: {scounts}")
        check("ditshift-determinism-note", True,
              "single-seed, retrain NOT verified (stated, not green-washed)")
        str_ = ds.get("transfer", {})
        check("ditshift-transfer-unjudged",
              set(str_) == {"fixtures"}
              and all(str_[t].get("judged") is False for t in str_)
              and "grid-lock" in str_["fixtures"].get("note", ""),
              "transfer reported-not-graded (grid-lock note)")

    # -- Phase-26 scale-boundedness legs (banked telemetry) --------------
    # Frozen bounds: act_out < 1e4, act_in < 1e3, end mod vars < 1e2,
    # final loss < 0.40 (descent past the 0.40-0.74 pin). Phase-25
    # recurrence numbers alongside every assertion (6.6e8 / 14415 /
    # final 0.6183). Round verdict recomputed: bounded+descended =
    # mechanism-confirmed-and-progressing, else deeper-pathology.
    TP26 = os.path.join(ROOT, "samples", "marigold",
                        "ditshift64b2_telemetry.json")
    if not os.path.isfile(TP26):
        print("SKIP surgery-telemetry legs (no banked Phase-26 telemetry)")
    else:
        st = json.load(open(TP26))
        smps = st.get("samples", [])
        mx_out = max(b["act_out"] for x in smps for b in x["blocks"])
        mx_in = max(b["act_in"] for x in smps for b in x["blocks"])
        endvars = [x["blocks"][b]["mod"][c]["var"]
                   for x in smps[-1:] for b in (0, 1)
                   for c in ("sh1", "sc1", "g1", "sh2", "sc2", "g2")]
        mx_mod = max(endvars)
        fin = st["loss_trace"][-1]
        check("surgery-acts-bounded",
              mx_out < 1e4 and mx_in < 1e3,
              f"act_out max={mx_out:.1f} (<1e4; P25 6.6e8) "
              f"act_in max={mx_in:.1f} (<1e3)")
        check("surgery-mods-bounded",
              mx_mod < 1e2,
              f"end mod var max={mx_mod:.4f} (<1e2; P25 14415)")
        check("surgery-descent",
              fin < 0.40,
              f"final loss={fin:.4f} (<0.40 past pin; P25 0.6183)")
        bounded = mx_out < 1e4 and mx_in < 1e3 and mx_mod < 1e2
        rv = ("mechanism-confirmed-and-progressing"
              if (bounded and fin < 0.40) else "deeper-pathology")
        check("surgery-verdict",
              st.get("round_verdict", {}).get("name") == rv,
              f"banked={st.get('round_verdict', {}).get('name')} "
              f"recomputed={rv}")
        sk = st.get("smoke", {})
        check("surgery-smoke",
              bool(sk.get("ok")) and sk.get("it0_diff", 1) <= 1e-4,
              f"it0 match diff={sk.get('it0_diff'):.2e} (tol 1e-4; "
              f"prefix-past-it0 invalid across schedules, stated)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
