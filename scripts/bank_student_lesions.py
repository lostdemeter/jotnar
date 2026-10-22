"""Bank student lesion tables (one-shot measurement, CPU-only).

Builds mscale64 + jattn64b2 from banked configs with banked weights
(READ-ONLY out-of-tree, weights_only, eval + no_grad — no training,
no weight writes), runs chain.debugger.TorchSession lesion tables,
banks samples/marigold/lesion_<tag>.json (the gate verifies them).
Church fixture input, full-frame depth output, deterministic.
Usage: python3 scripts/bank_student_lesions.py
"""
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")


def build_mscale(torch):
    import train_student as T
    torch.manual_seed(7)
    layers = [torch.nn.Linear(177, 64), torch.nn.SiLU(),
              torch.nn.Linear(64, 64), torch.nn.SiLU(),
              torch.nn.Linear(64, 1)]
    net = torch.nn.Sequential(*layers).float().eval()
    net.load_state_dict(torch.load(os.path.join(OUTBASE, "mscale64",
                                                "weights.pt"),
                                   map_location="cpu", weights_only=True))
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    row = next(r for r in man["fixtures"] if r["name"] == "church")
    import numpy as np
    from PIL import Image
    import glob as _g
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(_g.glob(os.path.join(FIXDIR, "inputs", "church.*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    x = torch.tensor(T.multiscale_features(im, 33, 3, False),
                     dtype=torch.float32)

    def run():
        return {"depth": net(x).reshape(-1)}

    return net, run, "church full-frame multiscale-177"


def build_jattn(torch):
    import train_attn_student as JA
    JointAttn = JA._define_model(torch)
    torch.manual_seed(7)
    im, _ = JA.load_frame("church")
    H0, W0 = im.shape[:2]
    nrows, ncols = H0 // 32, W0 // 32
    net = JointAttn(64, nrows, ncols, 4, 2, 1).float().eval()
    res = net.load_state_dict(torch.load(
        os.path.join(OUTBASE, "jattn64b2", "weights.pt"),
        map_location="cpu", weights_only=True), strict=False)
    assert sorted(res.missing_keys) == ["dino_proj.bias",
                                        "dino_proj.weight"], res
    assert res.unexpected_keys == [], res
    rgb = torch.tensor(im, dtype=torch.float32)
    yy, xx = torch.meshgrid(torch.arange(H0), torch.arange(W0),
                            indexing="ij")
    pidx = ((yy // 32) * ncols + (xx // 32)).long()

    def run():
        return {"depth": net.head(net.encode(rgb), rgb, pidx).reshape(-1)}

    return net, run, "church rgb-path encode+head (Phase-8 shape)"


def main():
    import numpy as np
    import torch
    from chain import debugger as D
    for tag, build in (("mscale64", build_mscale),
                       ("jattn64b2", build_jattn)):
        t0 = time.time()
        net, run, desc = build(torch)
        sess = D.TorchSession(net, run)
        rows = sess.lesion_table()
        ctl = sess.control()
        rep = {"tag": tag, "input": desc, "outputs": ["depth"],
               "leaves": sess.leaves, "rows": rows, "control": ctl,
               "torch": torch.__version__,
               "finite": bool(all(np.isfinite(r["worst"])
                                  for r in rows))}
        outp = os.path.join(FIXDIR, f"lesion_{tag}.json")
        json.dump(rep, open(outp, "w"), indent=1)
        print(f"{tag}: {len(rows)} rows control={ctl['deltas']} "
              f"({time.time() - t0:.0f}s) wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
