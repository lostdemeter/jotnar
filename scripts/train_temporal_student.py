"""Two-frame temporal student (student Phase 28): parallax as information.

mscale64-EXACT front end per frame (177-d field-33/densewin-3) +
CONCAT[feat_t, feat_t+1] (354-d) + the same MLP mixer (W64/d2/SiLU).
--mode single = single-frame control (177-d, reference frames only;
same scenes/splits/seed/schedule — the ONLY delta is the second
frame). Cost-volume/flow-warp REPLACED with reason (no poses/flow
GT in shard; concat keeps the comparison airtight — see
constitution). Same aligned-MAE stride-7 doctrine both arms.
Weights out-of-tree; metrics -> temporal_run_<tag>.json.
Usage: python3 scripts/train_temporal_student.py --mode pair --tag temp-pair
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pair", "single"], default="pair")
    ap.add_argument("--tag", default="temp-pair")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mb", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch
    import train_hyper as TH
    import train_student as T
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} mode={args.mode}",
          flush=True)

    def pairs(idxs):
        out = []
        for s in idxs:
            frs = TH.scene_frames(s)[:TH.NFR]
            out += [((s, frs[i]), (s, frs[i + 1]))
                    for i in range(len(frs) - 1)]
        return out

    train_pairs = pairs(TH.TRAIN_SCENES)
    held_pairs = pairs(TH.HELD_SCENES)
    print(f"pairs: {len(train_pairs)} train + {len(held_pairs)} heldout",
          flush=True)

    # preload stride-sampled single-frame features for every unique
    # frame (values identical to per-step compute, order fixed).
    uniq = []
    for pr in train_pairs + held_pairs:
        for k in pr:
            if k not in uniq:
                uniq.append(k)
    F, TT, VV = {}, {}, {}
    shapes = set()
    t0 = time.time()
    for k in uniq:
        im, depth, valid = TH.load_pair(*k)
        shapes.add(depth.shape)
        f = T.multiscale_features(im, 33, 3, False)
        F[k] = torch.tensor(f.reshape(-1, 177), dtype=torch.float32)
        TT[k] = torch.tensor(depth.reshape(-1).astype(np.float64))
        VV[k] = torch.tensor(valid.reshape(-1))
    assert len(shapes) == 1, f"frame shapes vary: {shapes} (stop: grid math)"
    npx = next(iter(TT.values())).numel()
    sidx = torch.arange(0, npx, 7)
    print(f"preload: {len(uniq)} frames in {time.time() - t0:.0f}s", flush=True)
    Fs = {k: v[sidx] for k, v in F.items()}
    Ts = {k: v[sidx] for k, v in TT.items()}
    Vs = {k: v[sidx] for k, v in VV.items()}
    del F, TT, VV

    Din = 177 if args.mode == "single" else 354
    layers = [torch.nn.Linear(Din, args.width), torch.nn.SiLU()]
    for _ in range(args.depth - 1):
        layers += [torch.nn.Linear(args.width, args.width), torch.nn.SiLU()]
    layers += [torch.nn.Linear(args.width, 1)]
    net = torch.nn.Sequential(*layers).to(device).float()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} din={Din}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    t0 = time.time()
    for it in range(args.steps):
        opt.zero_grad()
        ps, ts = [], []
        mb = [train_pairs[(it * args.mb + k) % len(train_pairs)]
              for k in range(args.mb)]
        for (ka, kb) in mb:
            fa = Fs[ka].float()
            x = fa if args.mode == "single" else torch.cat(
                [fa, Fs[kb].float()], dim=-1)
            ps.append(net(x).reshape(-1).double())
            ts.append(Ts[ka].double())
        p = torch.cat(ps)
        Tall = torch.cat(ts)
        with torch.no_grad():
            A = torch.stack([p.detach(), torch.ones_like(p)], 1)
            ab, _, _, _ = torch.linalg.lstsq(A, Tall)
            a, b = ab[0], ab[1]
        loss = (a * p + b - Tall).abs().mean()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {loss.item():.4f}", flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    rep = {"tag": args.tag, "arch": "mscale-temporal", "mode": args.mode,
           "width": args.width, "depth": args.depth, "field": 33,
           "densewin": 3, "steps": args.steps, "lr": args.lr,
           "seed": args.seed, "mb": args.mb,
           "train": [f"{a[0]}/{a[1]:04}+{b[1]:04}" for a, b in train_pairs],
           "heldout": [f"{a[0]}/{a[1]:04}+{b[1]:04}" for a, b in held_pairs],
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, idxs in (("train", train_pairs), ("heldout", held_pairs)):
            for (ka, kb) in idxs:
                im, depth, valid = TH.load_pair(*ka)
                fa = T.multiscale_features(im, 33, 3, False)
                if args.mode == "single":
                    x = fa
                else:
                    imb, _, _ = TH.load_pair(*kb)
                    x = np.concatenate(
                        [fa, T.multiscale_features(imb, 33, 3, False)], -1)
                p = net(torch.tensor(x, dtype=torch.float32,
                                     device=device)).reshape(-1).double()
                p = p.cpu().numpy()
                masks = H.strata_masks(depth, im)
                per = {}
                for c in H.STRATA:
                    m = masks[c] & valid
                    per[c] = (None if int(m.sum()) == 0 else H.align_mae(
                        p.reshape(depth.shape), depth, m))
                rep["frames"][f"{ka[0]}_fr{ka[1]:04}+{kb[1]:04}"] = {
                    "split": split, "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: (None if v is None else round(v, 4))
                               for k, v in fr["strata"].items()}, flush=True)
    outp = os.path.join(FIXDIR, f"temporal_run_{args.tag}.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
