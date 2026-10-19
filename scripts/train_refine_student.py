"""Two-stage coarse-to-fine student (student Phase 16): torch, outside listings.

Pass 1 = frozen banked jattn64b2 (front-end reuse, never retrained):
per frame, image-token global code (672x64) + coarse depth (own full
forward, no grad). Pass 2 = trained refinement MLP (68->128->128->1,
SILU) on CONCAT[code (nearest-cell GATHER) + coarse depth + RGB].
Modes: frozen (head only) isolates information usability; joint
(front-end warm-started from banked weights, all train) runs second.
Deployed ops: GATHER/CONCAT/MATMUL/ADD/SILU (all tabled Phase-8
precedent); no new machinery. Weights out-of-tree; metrics ->
samples/marigold/attn_run_<tag>.json (tracked attn_run.json never
touched). Usage: python3 scripts/train_refine_student.py --tag
cf-frozen --mode frozen
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
    ap.add_argument("--tag", default="cf-frozen")
    ap.add_argument("--mode", choices=["frozen", "joint"], default="frozen")
    ap.add_argument("--pass1", default="jattn64b2")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--schedule", choices=["const", "cosine"], default="const")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", nargs="+",
                    default=["church", "dogs", "squirrel", "train"])
    ap.add_argument("--heldout", nargs="+",
                    default=["geo_webcam_0_combined", "geo_webcam_1_combined"])
    args = ap.parse_args()

    import numpy as np
    import torch
    import train_attn_student as JA
    JointAttn = JA._define_model(torch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} mode={args.mode}", flush=True)

    frames = {}
    for n in args.train + args.heldout:
        im, t = JA.load_frame(n)
        frames[n] = (torch.tensor(im, dtype=torch.float32, device=device),
                     torch.tensor(t, dtype=torch.float64, device=device))
    H0, W0 = frames[args.train[0]][0].shape[:2]
    nrows, ncols = H0 // 32, W0 // 32
    yy, xx = torch.meshgrid(torch.arange(H0, device=device),
                            torch.arange(W0, device=device), indexing="ij")
    pidx = ((yy // 32) * ncols + (xx // 32)).long()
    sidx = torch.arange(0, H0 * W0, 7, device=device)

    front = JointAttn(64, nrows, ncols, 4, 2, 1).to(device).float()
    # banked Phase-8 weights predate dino_proj (unused in rgb path):
    # strict=False, then prove nothing else is missing/unexpected.
    res = front.load_state_dict(torch.load(
        os.path.join(OUTBASE, args.pass1, "weights.pt"),
        map_location=device, weights_only=True), strict=False)
    assert sorted(res.missing_keys) == ["dino_proj.bias", "dino_proj.weight"], res
    assert res.unexpected_keys == [], res
    print(f"pass1 {args.pass1} weights loaded (dino_proj fresh, rest exact)", flush=True)
    head = torch.nn.Sequential(
        torch.nn.Linear(68, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 1)).to(device).float()

    def pass1_feats(n):
        """frozen (code_broadcast HxWx64, coarse HxW) for frame n."""
        rgb, _ = frames[n]
        with torch.no_grad():
            to = front.encode(rgb)
            coarse = front.head(to, rgb, pidx)
            code = to[pidx]
        return code, coarse

    if args.mode == "frozen":
        front.eval()
        feats = {n: pass1_feats(n) for n in args.train + args.heldout}
        params = list(head.parameters())
    else:
        front.train()
        feats = {}
        params = list(front.parameters()) + list(head.parameters())
    nparams = sum(p.numel() for p in params)
    nparams_all = (sum(p.numel() for p in front.parameters())
                   + sum(p.numel() for p in head.parameters()))
    print(f"trainable={nparams} total={nparams_all}", flush=True)
    opt = torch.optim.Adam(params, lr=args.lr)

    def feat_sample(n):
        rgb, _ = frames[n]
        if args.mode == "frozen":
            code, coarse = feats[n]
            f = torch.cat([code, coarse.unsqueeze(-1),
                           rgb.float()], dim=-1).reshape(-1, 68)[sidx]
            return f.detach()
        to = front.encode(rgb)
        coarse = front.head(to, rgb, pidx)
        code = to[pidx]
        return torch.cat([code, coarse.unsqueeze(-1),
                          rgb.float()], dim=-1).reshape(-1, 68)[sidx]

    Tcat = torch.cat([frames[n][1].reshape(-1)[sidx] for n in args.train])
    t0 = time.time()
    import math as _math
    for it in range(args.steps):
        if args.schedule == "cosine":
            lr_t = args.lr * 0.5 * (1.0 + _math.cos(_math.pi * it / args.steps))
            for pg in opt.param_groups:
                pg["lr"] = lr_t
        opt.zero_grad()
        ps = []
        for n in args.train:
            p = head(feat_sample(n).float()).reshape(-1).double()
            ps.append(p)
        p = torch.cat(ps)
        with torch.no_grad():
            A = torch.stack([p.detach(), torch.ones_like(p)], 1)
            ab, _, _, _ = torch.linalg.lstsq(A, Tcat)
            a, b = ab[0], ab[1]
        loss = (a * p + b - Tcat).abs().mean()
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {loss.item():.4f}", flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save({"front": front.state_dict(), "head": head.state_dict()},
               os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "mode": args.mode, "pass1": args.pass1,
           "head": "68-128-128-1", "width": 64, "blocks": 2, "heads": 1,
           "stride": 32, "nctx": 4, "steps": args.steps,
           "lr": args.lr, "schedule": args.schedule, "seed": args.seed,
           "field": "full-frame",
           "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams_all, "frames": {}}
    front.eval()
    head.eval()
    with torch.no_grad():
        for split, names in (("train", args.train), ("heldout", args.heldout)):
            for n in names:
                rgb, t = frames[n]
                to = front.encode(rgb)
                coarse = front.head(to, rgb, pidx)
                code = to[pidx]
                f = torch.cat([code, coarse.unsqueeze(-1),
                               rgb.float()], dim=-1)
                p = head(f.float()).cpu().numpy().reshape(-1)
                t = t.cpu().numpy()
                Hh, Ww = next(r["shape"] for r in man["fixtures"]
                              if r["name"] == n)
                masks = H.strata_masks(t.reshape(Hh, Ww),
                                       rgb.cpu().numpy().reshape(Hh, Ww, 3))
                per = {k: H.align_mae(p.reshape(Hh, Ww), t.reshape(Hh, Ww), masks[k])
                       for k in H.STRATA}
                rep["frames"][n] = {"split": split, "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: round(v, 4) for k, v in fr["strata"].items()},
              flush=True)
    outp = os.path.join(FIXDIR, f"attn_run_{args.tag}.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
