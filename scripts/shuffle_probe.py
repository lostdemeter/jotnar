"""Patch-shuffle probe (student Phase 14): measure, don't train.

Scramble fixture inputs in P×P patches (seed 7, deterministic;
texture preserved, global structure destroyed) and run three
systems on intact-vs-shuffled pairs: seeded teacher (existing CLI,
GPU), DAV2 geometric replica (sibling, read-only, CPU), banked
students mscale64 + jattn64b2 (CPU). Metric: aligned-MAE deltas
intact→shuffled per stratum, same harness alignment, masks from
intact (same masks all arms). No training, no new deps
(PIL/numpy shuffle only). Predictions stay in /tmp; scalars banked
to samples/marigold/shuffle_probe.json.
Usage: python3 scripts/shuffle_probe.py [--skip-teacher]
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
SIBLING = os.path.normpath(os.path.join(ROOT, "..", "marigold-v2"))
DAV2DIR = "/home/balrog/Documents/OpenCode/dav2_revisited/dav2_reverse"
TMP = "/tmp/opencode/shuffle"
PATCHES = [16, 64]
SHUFFLE_SEED = 7
TW, TH = 1024, 672


def shuffle_rgb(im, patch, seed=SHUFFLE_SEED):
    """im uint8 (H,W,3) -> (scrambled copy, source-index).

    Pools keyed by tile shape (ragged edges permuted within-pool);
    deterministic in `seed`. sidx[i,j] = (si,sj) with
    out[i,j] = im[si,sj] — the exact inverse map (see unshuffle).
    Texture preserved, global structure destroyed."""
    import numpy as np
    H, W, _ = im.shape
    rng = np.random.RandomState(seed)
    rows, cols = [], []
    y = 0
    while y < H:
        rows.append((y, min(y + patch, H)))
        y += patch
    x = 0
    while x < W:
        cols.append((x, min(x + patch, W)))
        x += patch
    pools = {}
    for (y0, y1) in rows:
        for (x0, x1) in cols:
            pools.setdefault((y1 - y0, x1 - x0), []).append((y0, y1, x0, x1))
    out = np.empty_like(im)
    sidx = np.empty((H, W, 2), dtype=np.int32)
    for tiles in pools.values():
        perm = rng.permutation(len(tiles))
        src = [im[y0:y1, x0:x1] for (y0, y1, x0, x1) in tiles]
        for (y0, y1, x0, x1), s in zip(tiles, perm):
            out[y0:y1, x0:x1] = src[s]
            sy0, sy1, sx0, sx1 = tiles[s]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            sidx[y0:y1, x0:x1, 0] = yy - y0 + sy0
            sidx[y0:y1, x0:x1, 1] = xx - x0 + sx0
    return out, sidx


def unshuffle(arr, sidx):
    """Exact inverse of shuffle_rgb: out[si,sj] = arr[i,j]."""
    import numpy as np
    out = np.empty_like(arr)
    out[sidx[..., 0], sidx[..., 1]] = arr
    return out


def base_images():
    """6 fixtures as uint8 (1024,672,3), load_frame resize semantics."""
    import numpy as np
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    out = {}
    for row in man["fixtures"]:
        cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", row["name"] + ".*")))
        im = np.asarray(Image.open(cand[0]).convert("RGB").resize((TW, TH)))
        out[row["name"]] = im
    return out


def main():
    import numpy as np
    import torch
    import test_strata_marigold as H
    skip_teacher = "--skip-teacher" in sys.argv

    base = base_images()
    bank = {"patch": PATCHES, "seed": SHUFFLE_SEED, "systems": {},
            "metric": "align_mae(unshuffle(pred_shuffled), pred_intact) per stratum; "
                      "rel = raw / std(intact over mask) for cross-system scale comparability"}
    # permutation maps depend on grid only: same seed -> same map
    import numpy as np
    sidx = {p: shuffle_rgb(np.zeros((TH, TW, 3), np.uint8), p)[1] for p in PATCHES}

    def rel_delta(back, ref, mask):
        """(raw, rel): rel normalizes by the system's own intact variation."""
        import test_strata_marigold as H
        raw = float(H.align_mae(back, ref, mask))
        s = float(ref.ravel()[np.ascontiguousarray(mask).ravel()].std())
        rel = raw / s if s > 1e-12 else None
        return raw, rel

    # -- banked intact references ---------------------------------------
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    teacher_intact = {}
    for row in man["fixtures"]:
        teacher_intact[row["name"]] = np.load(
            os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    masks_full = {}
    for n, im in base.items():
        t = teacher_intact[n]
        masks_full[n] = H.strata_masks(t, im.astype(np.float64) / 255.0)

    # -- teacher shuffled arms (GPU, existing CLI) -----------------------
    if "--skip-teacher" in sys.argv:
        import glob as _g
        teacher_shuf = {}
        for p in PATCHES:
            fs = sorted(_g.glob(os.path.join(
                TMP, f"p{p}", "out0/images/predictions_npy/*.npy")))
            assert len(fs) == 6, (p, len(fs), "rerun without --skip-teacher")
            for f in fs:
                n = os.path.basename(f)[:-4]
                teacher_shuf[(n, p)] = np.load(f).astype(np.float64)
        print("teacher preds reused from /tmp (no rerun)", flush=True)
        import glob as _g2
        a = sorted(_g2.glob(os.path.join(TMP, "p16/out0/images/predictions_npy/*.npy")))
        b = sorted(_g2.glob(os.path.join(TMP, "p16/out1/images/predictions_npy/*.npy")))
        ident = (len(a) == len(b) == 6 and all(
            open(x, "rb").read() == open(y, "rb").read() for x, y in zip(a, b)))
        print("teacher p16 seeded-twice byte-identical x6 (reverified):", ident, flush=True)
        bank.setdefault("teacher_runs", {})["p16_twice_identical"] = bool(ident)
    if not skip_teacher:
        venv_py = os.path.expanduser("~/marigold-env/bin/python")
        keep = {k: os.environ[k] for k in
                ("HOME", "USER", "LOGNAME", "TMPDIR", "CUDA_VISIBLE_DEVICES",
                 "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
                 "HF_HUB_OFFLINE", "http_proxy", "https_proxy", "HTTP_PROXY",
                 "HTTPS_PROXY", "no_proxy") if k in os.environ}
        keep["PATH"] = os.path.dirname(venv_py) + ":/usr/bin:/bin"
        from PIL import Image
        teacher_shuf = {}
        for p in PATCHES:
            inp = os.path.join(TMP, f"p{p}", "inputs")
            shutil.rmtree(os.path.join(TMP, f"p{p}"), ignore_errors=True)
            os.makedirs(inp)
            for n, im in base.items():
                sh, _ = shuffle_rgb(im, p)
                Image.fromarray(sh).save(os.path.join(inp, n + ".png"))
            reps = 2 if p == PATCHES[0] else 1
            for r in range(reps):
                od = os.path.join(TMP, f"p{p}", f"out{r}")
                rr = subprocess.run(
                    [venv_py, os.path.join(SIBLING, "scripts", "infer.py"),
                     "--image_dir", inp, "--output_dir", od,
                     "--seed", "2025", "--width", str(TW), "--height", str(TH)],
                    capture_output=True, text=True, cwd=SIBLING, timeout=1800,
                    env=keep)
                assert rr.returncode == 0, (rr.stderr or rr.stdout)[-500:]
            if reps == 2:
                a = sorted(glob.glob(os.path.join(TMP, f"p{p}", "out0/images/predictions_npy/*.npy")))
                b = sorted(glob.glob(os.path.join(TMP, f"p{p}", "out1/images/predictions_npy/*.npy")))
                ident = (len(a) == len(b) == 6 and all(
                    open(x, "rb").read() == open(y, "rb").read() for x, y in zip(a, b)))
                print(f"teacher p{p} seeded-twice byte-identical x6:", ident, flush=True)
                bank.setdefault("teacher_runs", {})[f"p{p}_twice_identical"] = bool(ident)
            for n in base:
                teacher_shuf[(n, p)] = np.load(os.path.join(
                    TMP, f"p{p}", "out0/images/predictions_npy", n + ".npy")).astype(np.float64)
    deltas = {}
    for (n, p), pred in teacher_shuf.items():
        assert pred.shape == teacher_intact[n].shape, (pred.shape, n)
        back = unshuffle(pred, sidx[p])
        per = {}
        for k in H.STRATA:
            raw, rel = rel_delta(back, teacher_intact[n], masks_full[n][k])
            per[k] = {"raw": raw, "rel": rel}
        deltas.setdefault(f"p{p}", {})[n] = per
    bank["systems"]["teacher"] = {"res": "1024x672", "deltas": deltas}

    # -- DAV2 replica (sibling, read-only, CPU) ---------------------------
    sys.path.insert(0, DAV2DIR)
    import geo_depth
    from PIL import Image
    dav = geo_depth.GeometricDepthAnythingV2(device="cpu")
    dav_int, dav_shape = {}, None
    with torch.no_grad():
        for n, im in base.items():
            t = dav(geo_depth.preprocess(im.astype(np.float64) / 255.0)).cpu().numpy()[0]
            dav_int[n] = t.astype(np.float64)
        dav_shape = next(iter(dav_int.values())).shape
        print("dav2 out shape:", dav_shape, flush=True)
        # resampling-error control: NN roundtrip on banked depth, no system
        rs_err = max(
            float(H.align_mae(
                np.asarray(Image.fromarray(
                    teacher_intact[n].astype(np.float32), mode="F").resize(
                    (dav_shape[1], dav_shape[0]), Image.NEAREST).resize(
                    (TW, TH), Image.NEAREST)).astype(np.float64),
                teacher_intact[n], np.ones_like(teacher_intact[n], bool)))
            for n in base)
        print(f"dav2 resampling control (roundtrip align_mae): {rs_err:.2e}",
              flush=True)
        bank.setdefault("controls", {})["dav2_nn_roundtrip"] = rs_err

        def dav_up(t):
            return np.asarray(Image.fromarray(
                t.astype(np.float32), mode="F").resize(
                (TW, TH), Image.NEAREST)).astype(np.float64)

        dav_d = {}
        for p in PATCHES:
            for n, im in base.items():
                sh, _ = shuffle_rgb(im, p)
                t = dav(geo_depth.preprocess(
                    sh.astype(np.float64) / 255.0)).cpu().numpy()[0].astype(np.float64)
                back = unshuffle(dav_up(t), sidx[p])
                ref = dav_up(dav_int[n])
                m = H.strata_masks(ref, im.astype(np.float64) / 255.0)
                per = {}
                for k in H.STRATA:
                    raw, rel = rel_delta(back, ref, m[k])
                    per[k] = {"raw": raw, "rel": rel}
                dav_d.setdefault(f"p{p}", {})[n] = per
    bank["systems"]["dav2"] = {"res": f"{dav_shape[1]}x{dav_shape[0]}+nnup",
                               "deltas": dav_d}

    # -- banked students (CPU) --------------------------------------------
    import train_student as T
    sd = torch.load(os.path.expanduser("~/student_runs/mscale64/weights.pt"),
                    map_location="cpu", weights_only=True)
    mnet = torch.nn.Sequential(
        torch.nn.Linear(177, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 64), torch.nn.SiLU(),
        torch.nn.Linear(64, 1)).float().eval()
    mnet.load_state_dict(sd)
    import train_attn_student as JA
    JAMod = JA._define_model(torch)
    sdj = torch.load(os.path.expanduser("~/student_runs/jattn64b2/weights.pt"),
                     map_location="cpu", weights_only=True)
    jnet = JAMod(64, TH // 32, TW // 32, 4, 2).float().eval()
    jnet.load_state_dict(sdj)
    yy, xx = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing="ij")
    pidx = ((yy // 32) * (TW // 32) + (xx // 32)).long()

    def student_pred(which, im01):
        with torch.no_grad():
            if which == "mscale64":
                x = torch.tensor(T.multiscale_features(im01, 33, 3, False),
                                 dtype=torch.float32)
                return mnet(x).reshape(-1).double().numpy().reshape(TH, TW)
            rgbt = torch.tensor(im01.astype(np.float32))
            to = jnet.encode(rgbt)
            return jnet.head(to, rgbt, pidx).numpy()

    for tag in ("mscale64", "jattn64b2"):
        ref = {n: student_pred(tag, base[n].astype(np.float64) / 255.0) for n in base}
        dd = {}
        for p in PATCHES:
            for n in base:
                sh, _ = shuffle_rgb(base[n], p)
                pr = student_pred(tag, sh.astype(np.float64) / 255.0)
                back = unshuffle(pr, sidx[p])
                per = {}
                for k in H.STRATA:
                    raw, rel = rel_delta(back, ref[n], masks_full[n][k])
                    per[k] = {"raw": raw, "rel": rel}
                dd.setdefault(f"p{p}", {})[n] = per
        bank["systems"][tag] = {"res": "1024x672", "deltas": dd}

    json.dump(bank, open(os.path.join(FIXDIR, "shuffle_probe.json"), "w"), indent=1)
    print("wrote samples/marigold/shuffle_probe.json", flush=True)
    for sys_name, sd_ in bank["systems"].items():
        for p in PATCHES:
            vals = [c["rel"] for fr in sd_["deltas"][f"p{p}"].values()
                    for c in fr.values() if c["rel"] is not None]
            print(f"{sys_name} p{p}: mean-rel-delta {np.mean(vals):.4f} "
                  f"over {len(vals)} cells", flush=True)


if __name__ == "__main__":
    main()
