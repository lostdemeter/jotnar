"""DiT-faithful on Hypersim GT (student Phase 24): torch, data swap only.

Phase-20 package EXACTLY (LN+GELU-form+AdaLN+patchify+unfused, B2,
W64/H4/s16, shifted-log z + aligned-MAE, NO aux, 8k-cosine) on the
Phase-23 Hypersim splits. Data is the ONLY variable (see
docs/STUDENT.md Phase-24 constitution, FROZEN pre-run).
Assembly: model from train_dit_student, scenes/frames from
train_hyper, plumbing from train_base. No existing file's science
touched. Weights out-of-tree; metrics -> attn_run_<tag>.json.
Usage: python3 scripts/train_dit_hypersim.py --tag dithyp64b2
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

import train_base as BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dithyp64b2")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--schedule", choices=["const", "cosine"], default="cosine")
    ap.add_argument("--grad-w", type=float, default=0.0)
    ap.add_argument("--aux-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mb", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    import torch
    import train_dit_student as DIT
    import train_hyper as TH
    DiTTiny = DIT._define_model(torch)
    BASE.seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)
    if device == "cuda":
        BASE.log_gpu_mem()

    train_idx = [(s, f) for s in TH.TRAIN_SCENES for f in TH.scene_frames(s)[:TH.NFR]]
    held_idx = [(s, f) for s in TH.HELD_SCENES for f in TH.scene_frames(s)[:TH.NFR]]
    assert len(train_idx) == 120 and len(held_idx) == 60, (len(train_idx), len(held_idx))
    print(f"frames: {len(train_idx)} train + {len(held_idx)} heldout", flush=True)

    # preload all frames to device (dit mirrors fixtures this way; 180
    # frames fit GPU — values identical to per-step loading, order fixed).
    frames = {}
    shapes = set()
    for s, f in train_idx + held_idx:
        im, depth, valid = TH.load_pair(s, f)
        shapes.add(depth.shape)
        frames[(s, f)] = (torch.tensor(im.astype(np.float32), device=device),
                          torch.tensor(depth.astype(np.float32), device=device),
                          torch.tensor(valid, device=device))
    assert len(shapes) == 1, f"frame shapes vary: {shapes} (stop: grid math)"
    H0, W0 = next(iter(shapes))
    print(f"frame shape: {H0}x{W0}", flush=True)
    nrows, ncols = H0 // args.stride, W0 // args.stride
    yy, xx = torch.meshgrid(torch.arange(H0, device=device),
                            torch.arange(W0, device=device), indexing="ij")
    pidx = ((yy // args.stride) * ncols + (xx // args.stride)).long()
    sidx = BASE.stride_index(H0 * W0, 7, device)

    # log-normalization constants from the train split (dit procedure;
    # valid pixels only — Hypersim has invalid (0/NaN/>65), fixtures
    # had none; log of those is unimplementable, stated necessity).
    tall = np.concatenate([frames[k][1].cpu().numpy()[frames[k][2].cpu().numpy()]
                           for k in train_idx])
    lo = float(tall.min())
    lt = np.log(np.maximum(tall - lo + 1.0, 1e-9))
    mu = float(lt.mean())
    sigma = float(lt.std())
    print(f"lognorm lo={lo:.4f} mu={mu:.4f} sigma={sigma:.4f}", flush=True)

    def to_z(t_np):
        return (np.log(np.maximum(t_np - lo + 1.0, 1e-9)) - mu) / sigma

    Z = {k: torch.tensor(to_z(frames[k][1].cpu().numpy()),
                         dtype=torch.float64, device=device)
         for k in frames}

    net = DiTTiny(args.width, nrows, ncols, args.blocks,
                  args.heads).to(device).float()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} grid={nrows}x{ncols} seq={nrows * ncols}",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    kx = torch.tensor(DIT.SOBEL_X, dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = torch.tensor(DIT.SOBEL_Y, dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def grad_mag(f2d):
        f = f2d.unsqueeze(0).unsqueeze(0).float()
        gx = torch.nn.functional.conv2d(f, kx, padding=1)
        gy = torch.nn.functional.conv2d(f, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze()

    t0 = time.time()
    for it in range(args.steps):
        BASE.apply_schedule(opt, args.schedule, args.lr, it, args.steps)
        opt.zero_grad()
        ps = []
        mb = [train_idx[(it * args.mb + k) % len(train_idx)]
              for k in range(args.mb)]
        Tm = torch.cat([Z[k].reshape(-1)[sidx.cpu()] for k in mb]).to(device)
        for k in mb:
            rgb, _, _ = frames[k]
            to = net.encode(rgb)
            p = net.head(to, rgb, pidx).reshape(-1)[sidx].double()
            ps.append(p)
        p = torch.cat(ps)
        with torch.no_grad():
            A = torch.stack([p.detach(), torch.ones_like(p)], 1)
            ab, _, _, _ = torch.linalg.lstsq(A, Tm)
            a, b = ab[0], ab[1]
        loss = (a * p + b - Tm).abs().mean()
        aux = torch.zeros((), device=device)
        if args.grad_w != 0 and (it % args.aux_every == 0
                                 or it == args.steps - 1):
            terms = []
            for k in mb:
                rgb, _, _ = frames[k]
                to = net.encode(rgb)
                pf = net.head(to, rgb, pidx).double()
                pa = a * pf + b
                zt = Z[k].reshape(rgb.shape[0], rgb.shape[1])
                terms.append((grad_mag(pa.reshape(zt.shape))
                              - grad_mag(zt.float())).abs().mean())
            aux = sum(terms) / len(terms)
        (loss + args.grad_w * aux).backward()
        opt.step()
        if BASE.cadence(it, args.steps, 500):
            print(f"iter {it}: loss {loss.item():.4f} aux {float(aux):.4f}",
                  flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)
    assert bool(torch.isfinite(loss).all()), f"final loss non-finite: {loss.item()}"

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    rep = {"tag": args.tag, "arch": "dit-tiny", "data": "hypersim",
           "width": args.width, "blocks": args.blocks, "heads": args.heads,
           "patch": 16, "adaln": "const-vector",
           "geluform": "silu(x*1.702)/1.702", "gradloss": args.grad_w,
           "lognorm": {"lo": lo, "mu": mu, "sigma": sigma},
           "stride": args.stride, "nctx": 0, "steps": args.steps,
           "lr": args.lr, "schedule": args.schedule, "mb": args.mb,
           "seed": args.seed, "field": "full-frame",
           "train": [f"{s}/fr{f:04}" for s, f in train_idx],
           "heldout": [f"{s}/fr{f:04}" for s, f in held_idx],
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, idxs in (("train", train_idx), ("heldout", held_idx)):
            for k in idxs:
                rgb, depth, valid = frames[k]
                to = net.encode(rgb)
                pz = net.head(to, rgb, pidx).cpu().numpy()
                zl = np.clip(np.asarray(rep["lognorm"]["sigma"]) * pz
                             + np.asarray(rep["lognorm"]["mu"]),
                             np.log(1e-9), np.log(1e4))
                p = np.exp(zl) + np.asarray(rep["lognorm"]["lo"]) - 1.0
                t = depth.cpu().numpy()
                im = rgb.cpu().numpy()
                masks = H.strata_masks(t, im)
                vm = valid.cpu().numpy()
                per = {}
                for c in H.STRATA:
                    m = masks[c] & vm
                    per[c] = (None if int(m.sum()) == 0 else H.align_mae(
                        p.reshape(t.shape), t, m))
                rep["frames"][f"{k[0]}_fr{k[1]:04}"] = {"split": split,
                                                        "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: (None if v is None else round(v, 4))
                               for k, v in fr["strata"].items()}, flush=True)
    outp = os.path.join(FIXDIR, f"attn_run_{args.tag}.json")
    BASE.bank_metrics(rep, outp)

    # transfer-report probe (ungraded): fixtures need 42x64 tables,
    # this grid is 48x64 — record the lock, don't grade it.
    try:
        probe = torch.zeros(672, 1024, 3, device=device)
        net.encode(probe)
        print("transfer-probe: UNEXPECTED — fixture grid encoded", flush=True)
        rep_transfer = {"judged": False, "note": "fixture grid encoded?!",
                        "grids": f"{nrows}x{ncols} vs 42x64"}
    except Exception as e:
        print(f"transfer-probe: grid-locked ({type(e).__name__})", flush=True)
        rep_transfer = {"judged": False, "note": f"grid-locked ({type(e).__name__})",
                        "grids": f"{nrows}x{ncols} vs 42x64"}
    json.dump(rep_transfer, open(os.path.join(
        OUTBASE, args.tag, "transfer.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
