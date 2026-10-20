"""Hypersim scale training (student Phase 23): torch, outside listings.

Two banked architectures, fresh weights, exact GT (no alignment):
- mscale: mscale64-EXACT mixer (177 dims, W64/d2), stride-7 sample.
- jattn: jattn SHAPE fresh at Hypersim grid (W64/B2/H1/stride-32/C4,
  24x32 grid; pos tables per-grid — grid-lock forbids weight reuse).
Loss: log-GT MAE (predict log depth). Minibatch 4 scenes/step,
deterministic rotation (no RNG in-loop). Eval: exp + AbsRel per
harness stratum vs metric GT (masks & valid-pixel intersect).
Transfer (secondary, ungraded): mscale on real fixtures (aligned
cells vs banked teacher depth); jattn grid-locked (documented).
Weights out-of-tree (~/student_runs/<tag>/); metrics ->
samples/marigold/hyper_run_<tag>.json.
Usage: python3 scripts/train_hyper.py --arch mscale --tag hyper-mscale
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

SHARD = os.path.expanduser("~/student_runs/hypersim_shard00/extract/test")
FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")
TRAIN_SCENES = ["ai_001_010", "ai_003_007", "ai_005_001", "ai_005_010",
                "ai_008_005", "ai_008_007", "ai_010_003", "ai_010_004",
                "ai_011_007", "ai_013_010", "ai_014_006", "ai_015_010"]
HELD_SCENES = ["ai_018_001", "ai_021_001", "ai_022_001",
               "ai_022_003", "ai_022_006", "ai_023_005"]
NFR = 10


def scene_frames(scene):
    """Sorted available frame ids with both rgb+depth present (ai_011_007
    ships even-frames-only — never assume contiguity)."""
    import glob as _g
    import re as _re
    have = set()
    for p in _g.glob(os.path.join(SHARD, scene, "rgb_cam_00_fr*.png")):
        m = _re.search(r"fr(\d+)\.png$", p)
        if m and os.path.isfile(os.path.join(
                SHARD, scene, f"depth_plane_cam_00_fr{m.group(1)}.png")):
            have.add(int(m.group(1)))
    return sorted(have)


def load_pair(scene, fr):
    import numpy as np
    from PIL import Image
    base = os.path.join(SHARD, scene)
    import numpy as np
    from PIL import Image
    base = os.path.join(SHARD, scene)
    rgb = np.asarray(Image.open(
        os.path.join(base, f"rgb_cam_00_fr{fr:04}.png")).convert("RGB"))
    depth = np.asarray(Image.open(
        os.path.join(base, f"depth_plane_cam_00_fr{fr:04}.png"))
        ).astype(np.float64) / 1000.0
    valid = np.isfinite(depth) & (depth > 0) & (depth <= 65.0)
    return rgb.astype(np.float64) / 255.0, depth, valid


def build_mlp(W=64, Din=177, depth=2):
    import torch
    layers = [torch.nn.Linear(Din, W), torch.nn.SiLU()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(W, W), torch.nn.SiLU()]
    layers += [torch.nn.Linear(W, 1)]
    return torch.nn.Sequential(*layers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mscale", "jattn"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mb", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch
    import train_student as T
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} arch={args.arch}",
          flush=True)
    nvidia = ""
    try:
        import subprocess as _sp
        nvidia = _sp.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=30).stdout.strip()
    except Exception:
        pass
    print(f"gpu: {nvidia}", flush=True)

    train_idx = [(s, scene_frames(s)[k])
                 for s in TRAIN_SCENES for k in range(NFR)]
    held_idx = [(s, scene_frames(s)[k])
                for s in HELD_SCENES for k in range(NFR)]
    assert all(len(scene_frames(s)) >= NFR for s in TRAIN_SCENES + HELD_SCENES)
    print(f"frames: {len(train_idx)} train + {len(held_idx)} heldout "
          f"(first {NFR} available/scene)", flush=True)

    if args.arch == "mscale":
        net = build_mlp().to(device).float()
        din = 177
    else:
        import train_attn_student as JA
        JAMod = JA._define_model(torch)
        net = JAMod(64, 24, 32, 4, 2, 1).to(device).float()
        din = None
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    sidx_cache = {}
    t0 = time.time()
    for it in range(args.steps):
        opt.zero_grad()
        tot = 0.0
        mb = [train_idx[(it * args.mb + k) % len(train_idx)]
              for k in range(args.mb)]
        for (s, f) in mb:
            im, depth, valid = load_pair(s, f)
            key = depth.shape
            if key not in sidx_cache:
                sidx_cache[key] = torch.arange(
                    0, depth.size, 7, device=device)
            sidx = sidx_cache[key]
            gt = torch.tensor(np.log(depth).astype(np.float32),
                              device=device).reshape(-1)[sidx]
            vm = torch.tensor(valid.reshape(-1)[sidx.cpu().numpy()],
                              device=device)
            if args.arch == "mscale":
                x = torch.tensor(T.multiscale_features(im, 33, 3, False),
                                 dtype=torch.float32, device=device)[sidx]
                p = net(x).reshape(-1).double()
            else:
                rgbt = torch.tensor(im.astype(np.float32), device=device)
                Hh, Ww, _ = im.shape
                yy, xx = torch.meshgrid(torch.arange(Hh, device=device),
                                        torch.arange(Ww, device=device),
                                        indexing="ij")
                to = net.encode(rgbt)
                p = net.head(to, rgbt,
                             ((yy // 32) * 32 + (xx // 32)).long()
                             ).reshape(-1)[sidx].double()
            l = (p[vm] - gt.double()[vm]).abs().mean()
            l.backward()
            tot += float(l)
        opt.step()
        if it % 50 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {tot / args.mb:.4f}", flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    rep = {"tag": args.tag, "arch": args.arch,
           "flags": {"field": 33, "densewin": 3, "depth": 2, "width": 64,
                     "steps": args.steps, "lr": args.lr, "seed": args.seed}
           if args.arch == "mscale" else
           {"field": "full-frame", "width": 64, "blocks": 2, "heads": 1,
            "stride": 32, "nctx": 4, "steps": args.steps, "lr": args.lr,
            "seed": args.seed},
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, idxs in (("train", train_idx), ("heldout", held_idx)):
            for (s, f) in idxs:
                im, depth, valid = load_pair(s, f)
                if args.arch == "mscale":
                    x = torch.tensor(T.multiscale_features(im, 33, 3, False),
                                     dtype=torch.float32, device=device)
                    p = torch.exp(net(x).reshape(-1)).double().cpu().numpy()
                else:
                    rgbt = torch.tensor(im.astype(np.float32), device=device)
                    Hh, Ww, _ = im.shape
                    yy, xx = torch.meshgrid(torch.arange(Hh, device=device),
                                            torch.arange(Ww, device=device),
                                            indexing="ij")
                    to = net.encode(rgbt)
                    pz = net.head(to, rgbt, ((yy // 32) * 32 + (xx // 32)
                                             ).long()).cpu().numpy()
                    p = np.exp(pz)
                Hh, Ww = depth.shape
                masks = H.strata_masks(depth, im)
                per = {}
                for k in H.STRATA:
                    m = masks[k] & valid
                    per[k] = (None if int(m.sum()) == 0 else float(
                        np.abs(p.reshape(Hh, Ww)[m] - depth[m]).mean()
                        / depth[m].mean()))
                rep["frames"][f"{s}_fr{f:04}"] = {"split": split,
                                                  "strata": per}
    outp = os.path.join(FIXDIR, f"hyper_run_{args.tag}.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
