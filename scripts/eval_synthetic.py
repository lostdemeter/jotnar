"""Synthetic-scene diagnosis (student Phase 12a): measure, don't train.

Teacher noise-vs-bias: seeded-twice teacher preds (outA/outB, 320x256)
vs analytic depth via affine-invariant align_mae (existing harness).
Students vs analytic GT: mscale64 at native 320x256 through the
extracted multiscale_features front-end (banked-match verified on
fixtures first); jattn64b2 sidecar at PIL-bilinear-upscaled 1024x672
with nearest-GT (grid-locked pos tables can't run off-grid — blur
confound stated; architecture finding, not a number to bank
against mscale64). Writes samples/marigold/synthetic_eval.json
(scalars only; scenes regenerate from seeds, predictions stay in
/tmp). No training, no emitter changes.
Usage: python3 scripts/eval_synthetic.py
"""
import glob
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
SYN = "/tmp/opencode/synth320"


def main():
    import numpy as np
    import torch
    import synth_depth as S
    import test_strata_marigold as H
    import train_student as T

    def EvaL(gt, pred, rgb01):
        """Teacher-native log space: masks on log-GT, align vs log-GT.

        Starved masks (median collapse on box-room stats — see spec)
        record None and stay out of means, counted openly."""
        masks = H.strata_masks(gt, rgb01)
        per = {}
        for k in H.STRATA:
            if int(masks[k].sum()) == 0:
                per[k] = None
            else:
                with np.errstate(invalid="ignore"):
                    per[k] = float(H.align_mae(pred, gt, masks[k]))
        per["_full"] = float(H.align_mae(pred, gt, np.ones_like(gt, bool)))
        per["_starved"] = [k for k in H.STRATA if per[k] is None]
        return per

    out = {"scenes": {}, "teacher": {"scenes": {}}, "students": {}}

    # -- scene catalog --------------------------------------------------
    for i in range(S.N_SCENES):
        _, _, m = S.render_scene(i)
        out["scenes"][f"s{i:02}"] = m

    # -- teacher bias (outA; outB already proved byte-identical) ---------
    # GT in log space: the teacher trains relative-LOG-depth
    # (training_relative_log_depth config), so affine-invariant
    # comparison belongs in log space; z-space residuals confound
    # the log compression with teacher error (measured, then fixed).
    for i in range(S.N_SCENES):
        rgb, depth, meta = S.render_scene(i)
        gt = np.log(depth)
        pred = np.load(os.path.join(
            SYN, "outA/images/predictions_npy/s%02d.npy" % i)).astype(np.float64)
        assert pred.shape == depth.shape, (pred.shape, depth.shape)
        out["teacher"]["scenes"][f"s{i:02}"] = {
            "strata": EvaL(gt, pred, rgb.astype(np.float64) / 255.0)}

    # -- mscale64 front-end banked-match on fixtures (required gate) -----
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    banked = json.load(open(os.path.join(FIXDIR, "student_run.json")))
    sd = torch.load(os.path.expanduser("~/student_runs/mscale64/weights.pt"),
                    map_location="cpu", weights_only=True)
    W, Din = 64, 177
    layers = [torch.nn.Linear(Din, W), torch.nn.SiLU(),
              torch.nn.Linear(W, W), torch.nn.SiLU(), torch.nn.Linear(W, 1)]
    net = torch.nn.Sequential(*layers).float().eval()
    net.load_state_dict(sd)
    worst = 0.0
    for row in man["fixtures"]:
        x, im3, t = T.load_frame(row["name"], 1, True, 33, 3, False)
        x2 = T.multiscale_features(im3, 33, 3, False)
        worst = max(worst, float(abs(x - x2).max()))
    print("front-end refactor max dev vs load_frame:", worst)
    assert worst == 0.0, worst
    worst = 0.0
    with torch.no_grad():
        for row in man["fixtures"]:
            x, im3, t = T.load_frame(row["name"], 1, True, 33, 3, False)
            p = net(torch.tensor(x, dtype=torch.float32)).reshape(-1).double().numpy()
            Hh, Ww = tuple(row["shape"])
            masks = H.strata_masks(t.reshape(Hh, Ww), im3.reshape(Hh, Ww, 3))
            for k in H.STRATA:
                got = H.align_mae(p.reshape(Hh, Ww), t.reshape(Hh, Ww), masks[k])
                want = banked["frames"][row["name"]]["strata"][k]
                worst = max(worst, abs(got - want))
    print("mscale64 eval-path max dev vs banked student_run.json:", worst)
    assert worst < 1e-9, worst

    # -- mscale64 vs analytic GT, native 320x256 (log space) ------------
    msc = {}
    with torch.no_grad():
        for i in range(S.N_SCENES):
            rgb, depth, meta = S.render_scene(i)
            gt = np.log(depth)
            im = rgb.astype(np.float64) / 255.0
            x = torch.tensor(T.multiscale_features(im, 33, 3, False),
                             dtype=torch.float32)
            p = net(x).reshape(-1).double().numpy().reshape(gt.shape)
            msc[f"s{i:02}"] = {"strata": EvaL(gt, p, im)}
    out["students"]["mscale64"] = {"res": "320x256 native", "scenes": msc}

    # -- jattn64b2 sidecar (upscaled 1024x672, blur stated) --------------
    from PIL import Image
    import train_attn_student as JA
    JAMod = JA._define_model(torch)
    sdj = torch.load(os.path.expanduser("~/student_runs/jattn64b2/weights.pt"),
                     map_location="cpu", weights_only=True)
    jat = {}
    with torch.no_grad():
        for i in range(S.N_SCENES):
            rgb, depth, meta = S.render_scene(i)
            im_big = np.asarray(Image.fromarray(rgb).resize(
                (1024, 672), Image.BILINEAR)) / 255.0
            gt_big = np.log(np.asarray(Image.fromarray(depth).resize(
                (1024, 672), Image.NEAREST)))
            Hh, Ww = 672, 1024
            netj = JAMod(64, Hh // 32, Ww // 32, 4, 2).float().eval()
            netj.load_state_dict(sdj)
            rgbt = torch.tensor(im_big, dtype=torch.float32)
            to = netj.encode(rgbt)
            yy, xx = torch.meshgrid(torch.arange(Hh), torch.arange(Ww),
                                    indexing="ij")
            pidx = ((yy // 32) * (Ww // 32) + (xx // 32)).long()
            p = netj.head(to, rgbt, pidx).numpy()
            jat[f"s{i:02}"] = {"strata": EvaL(gt_big, p, im_big)}
    out["students"]["jattn64b2"] = {"res": "1024x672 upscale sidecar",
                                    "scenes": jat}

    json.dump(out, open(os.path.join(FIXDIR, "synthetic_eval.json"), "w"), indent=1)
    print("wrote samples/marigold/synthetic_eval.json")

    # -- console tables ---------------------------------------------------
    for who in ("teacher",):
        vals = out[who]["scenes"]
        full = [vals[s]["strata"]["_full"] for s in sorted(vals)]
        print(f"{who}: scene-full mean {np.mean(full):.4f} "
              f"best {min(full):.4f} worst {max(full):.4f}")
    for tag in ("mscale64", "jattn64b2"):
        vals = out["students"][tag]["scenes"]
        full = [vals[s]["strata"]["_full"] for s in sorted(vals)]
        print(f"{tag}: scene-full mean {np.mean(full):.4f} "
              f"best {min(full):.4f} worst {max(full):.4f}")


if __name__ == "__main__":
    main()
