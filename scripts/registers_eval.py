"""Registers eval (CPU, deterministic single forward, no training).

Loads a jattn arm's weights.pt, recomputes strata cells (trainer-eval
mirror) + shuffle deltas p16/p64 (probe student-path mirror, seed 7).
Usage: ~/marigold-env/bin/python scripts/registers_eval.py TAG NCTX
Prints JSON {tag, nctx, strata, shuffle} to stdout.
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
TH, TW = 672, 1024


def main():
    import numpy as np
    import torch
    import train_attn_student as JA
    from shuffle_probe import shuffle_rgb, unshuffle, base_images
    import test_strata_marigold as H

    tag, nctx = sys.argv[1], int(sys.argv[2])
    JAMod = JA._define_model(torch)
    sd = torch.load(os.path.expanduser(f"~/student_runs/{tag}/weights.pt"),
                     map_location="cpu", weights_only=True)
    nrows, ncols = TH // 32, TW // 32
    net = JAMod(64, nrows, ncols, nctx, 2).float().eval()
    net.load_state_dict(sd)
    yy, xx = torch.meshgrid(torch.arange(TH), torch.arange(TW), indexing="ij")
    pidx = ((yy // 32) * (TW // 32) + (xx // 32)).long()

    def pred(im01):
        with torch.no_grad():
            rgbt = torch.tensor(im01.astype(np.float32))
            return net.head(net.encode(rgbt), rgbt, pidx).numpy()

    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    base = base_images()
    names = ["church", "dogs", "squirrel", "train",
             "geo_webcam_0_combined", "geo_webcam_1_combined"]
    strata = {}
    for n in names:
        t = np.load(os.path.join(FIXDIR, n + ".npy")).astype(np.float64)
        im = base[n]
        masks = H.strata_masks(t, im.astype(np.float64) / 255.0)
        p = pred(im.astype(np.float64) / 255.0)
        strata[n] = {k: float(H.align_mae(p, t, masks[k])) for k in H.STRATA}
    sidx = {}
    shuf = {}
    for p in (16, 64):
        per = {}
        for n in names:
            im = base[n]
            sh, idx = shuffle_rgb(im, p)
            sidx[p] = True
            back = unshuffle(pred(sh.astype(np.float64) / 255.0), idx)
            ref = pred(im.astype(np.float64) / 255.0)
            t = np.load(os.path.join(FIXDIR, n + ".npy")).astype(np.float64)
            masks = H.strata_masks(t, im.astype(np.float64) / 255.0)
            cell = {}
            for k in H.STRATA:
                raw = float(H.align_mae(back, ref, masks[k]))
                s = float(ref[masks[k]].std())
                cell[k] = {"raw": raw, "rel": raw / s if s > 1e-12 else None}
            per[n] = cell
        shuf[f"p{p}"] = per
    print(json.dumps({"tag": tag, "nctx": nctx, "strata": strata,
                      "shuffle": shuf}))


if __name__ == "__main__":
    main()
