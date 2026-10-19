"""Edge-concentration gate (student Phase 18): CPU-only, fast.

Edge-map determinism (same banked depth -> identical bands),
band partition integrity (disjoint, covering, on all 6 real
fixtures), recompute-vs-banked spot (mscale64/church full
forward + bands vs banked, tol 1e-9), verdict legs reading the
banked table (edge/interior per system vs the frozen 2x rule —
reported, never asserted beyond it). No training, no GPU.
Usage: python3 tests/test_edge_concentration.py (~1 min, CPU).
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
FIXDIR = os.path.join(ROOT, "samples", "marigold")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def main():
    import numpy as np
    import torch
    from edge_analysis import edge_bands, sobel_mag
    import test_strata_marigold as H
    import train_student as T
    from PIL import Image

    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    real = [r for r in man["fixtures"] if not r.get("synthetic")]

    # -- determinism + partition integrity (always) ----------------------
    det_ok, part_ok = True, True
    for row in real:
        t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
        b1, b2 = edge_bands(t), edge_bands(t)
        det_ok &= all(bool((b1[k] == b2[k]).all()) for k in b1)
        part_ok &= (not (b1["edge"] & b1["halo"]).any()
                    and not (b1["edge"] & b1["interior"]).any()
                    and not (b1["halo"] & b1["interior"]).any()
                    and bool((b1["edge"] | b1["halo"] | b1["interior"]).all()))
    check("edge-deterministic", bool(det_ok), "same depth identical bands")
    check("edge-partition", bool(part_ok), "disjoint + covering x6")

    bank_path = os.path.join(FIXDIR, "edge_analysis.json")
    if not os.path.isfile(bank_path):
        print("SKIP banked legs (no edge_analysis.json yet)")
        print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
        sys.exit(1 if FAIL else 0)
    bank = json.load(open(bank_path))
    check("edge-banked-schema",
          set(bank.get("systems", {})) == {"mscale64", "jattn64b2", "texmsc-2k", "texmsc-8kcos"}
          and all(len(bank["systems"][t]) == 6 for t in bank["systems"]),
          "4 systems x 6 fixtures")

    # -- recompute-vs-banked spot (mscale64/church, CPU) ------------------
    sd = torch.load(os.path.expanduser("~/student_runs/mscale64/weights.pt"),
                    map_location="cpu", weights_only=True)
    net = torch.nn.Sequential(
        torch.nn.Linear(177, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 1)).float().eval()
    net.load_state_dict(sd)
    row = next(r for r in real if r["name"] == "church")
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", "church.*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    with torch.no_grad():
        p = net(torch.tensor(T.multiscale_features(im, 33, 3, False),
                             dtype=torch.float32)).reshape(-1).double().numpy()
    Hh, Ww = tuple(row["shape"])
    bands = edge_bands(t)
    worst = max(abs(float(H.align_mae(p.reshape(Hh, Ww), t, bands[k]))
                    - bank["systems"]["mscale64"]["church"][k])
                for k in bands)
    check("edge-recompute-spot", worst < 1e-9, f"worst dev {worst:.1e}")

    # -- verdict legs (frozen 2x rule, reported) --------------------------
    for tag in ("mscale64", "jattn64b2", "texmsc-2k", "texmsc-8kcos"):
        e = np.mean([bank["systems"][tag][n]["edge"] for n in bank["systems"][tag]])
        i = np.mean([bank["systems"][tag][n]["interior"] for n in bank["systems"][tag]])
        check(f"edge-ratio-{tag}", True, f"edge/interior = {e / i:.2f} (rule needs >=2)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
