"""DPT-tiny torch training on analytic GT (Phase 21 Weibull... no — Phase: analytic training).

Res-general DPT-tiny (grid derived from image dims; same math as the
built listing / gate twin: patch-16 tokens, 6 plain-ViT blocks H4,
reassemble strided/broadcast, RefineNet GATHER-form fusion, head).
Trains toward analytic log-GT (direct regression, NO alignment —
exact truth needs none). Assembly lowering is a later round,
explicitly not this one. Weights out-of-tree; metrics + configs
banked in-repo via attn_run_<tag>.json analogues (dpt_run_<tag>).
Usage: python3 scripts/train_dpt_torch.py --tag dpt-gt --steps 2000
"""
import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")


def _define_model(torch):
    """Model classes (torch lives here and in main only, never module top)."""

    def geluform(x):
        return torch.nn.functional.silu(x * 1.702) / 1.702

    class Block(torch.nn.Module):
        def __init__(self, W, Hh=4):
            super().__init__()
            self.Hh = Hh
            self.n1 = torch.nn.LayerNorm(W, elementwise_affine=False)
            self.q = torch.nn.Linear(W, W, bias=False)
            self.k = torch.nn.Linear(W, W, bias=False)
            self.v = torch.nn.Linear(W, W, bias=False)
            self.o = torch.nn.Linear(W, W, bias=False)
            self.n2 = torch.nn.LayerNorm(W, elementwise_affine=False)
            self.m1 = torch.nn.Linear(W, 4 * W, bias=False)
            self.m2 = torch.nn.Linear(4 * W, W, bias=False)

        def forward(self, x):
            S, W = x.shape
            Dh = W // self.Hh
            h = self.n1(x)
            q = self.q(h).view(S, self.Hh, Dh).transpose(0, 1)
            k = self.k(h).view(S, self.Hh, Dh).transpose(0, 1)
            v = self.v(h).view(S, self.Hh, Dh).transpose(0, 1)
            a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(Dh), -1)
            o = (a @ v).transpose(0, 1).reshape(S, W)
            x = x + self.o(o)
            h = self.n2(x)
            x = x + self.m2(geluform(self.m1(h)))
            return x

    class DPTTiny(torch.nn.Module):
        def __init__(self, W, nrows, ncols, B, Hh=4, patch=16):
            super().__init__()
            self.W, self.patch = W, patch
            self.tok = torch.nn.Linear(3 * patch * patch, W, bias=False)
            self.row = torch.nn.Embedding(nrows, W)
            self.col = torch.nn.Embedding(ncols, W)
            self.blocks = torch.nn.ModuleList([Block(W, Hh) for _ in range(B)])
            self.rproj = torch.nn.ModuleList(
                [torch.nn.Linear(W, W, bias=False) for _ in range(4)])
            # fusion: 4 levels x 2 RCUs x 2 convs (full 3x3, replicate)
            self.fuse = torch.nn.ModuleList()
            for _ in range(4):
                self.fuse.append(torch.nn.ModuleList([
                    torch.nn.ModuleList([
                        torch.nn.Conv2d(W, W, 3, padding=1, bias=False,
                                        padding_mode="replicate")
                        for _ in range(2)])
                    for _ in range(2)]))
            self.hp = torch.nn.Linear(W, 32, bias=False)
            self.hl = torch.nn.Linear(32, 1, bias=False)

        def forward(self, rgb):
            Hh, Ww, _ = rgb.shape
            P, W = self.patch, self.W
            nrows, ncols = Hh // P, Ww // P
            t = rgb.unfold(0, P, P).unfold(1, P, P)
            toks = t.permute(0, 1, 3, 4, 2).reshape(-1, 3 * P * P)
            pos = (self.row.weight.repeat_interleave(ncols, dim=0)
                   + self.col.weight.repeat(nrows, 1))
            h = self.tok(toks) + pos
            outs = {}
            for bi, b in enumerate(self.blocks):
                h = b(h)
                if bi in (1, 3, 4, 5):
                    outs[bi] = h
            # reassemble to grids (strided/broadcast, mirror listing)
            g = {}
            g[(nrows // 2, ncols // 2)] = outs[1].view(
                nrows, ncols, W)[::2, ::2].reshape(-1, W)
            g[(nrows, ncols)] = outs[3]
            big8 = outs[4].view(nrows, ncols, W).repeat_interleave(
                2, 0).repeat_interleave(2, 1)
            g[(2 * nrows, 2 * ncols)] = big8.reshape(-1, W)
            big4 = outs[5].view(nrows, ncols, W).repeat_interleave(
                4, 0).repeat_interleave(4, 1)
            g[(4 * nrows, 4 * ncols)] = big4.reshape(-1, W)
            order = sorted(g, key=lambda k: k[0] * k[1])
            projs = [self.rproj[i](g[k]) for i, k in enumerate(order)]
            y = None
            for li in range(4):
                f = projs[li] if li == 0 else None
                if li > 0:
                    gh, gw = order[li]
                    ph, pw = order[li - 1][0], order[li - 1][1]
                    up = y.view(ph, pw, W).repeat_interleave(
                        gh // ph, 0).repeat_interleave(gw // pw, 1)
                    f = projs[li] + up.reshape(-1, W)
                fh, fw = order[li]
                f4 = f.view(fh, fw, W).permute(2, 0, 1).unsqueeze(0)
                for rcu in self.fuse[li]:
                    h2 = rcu[1](torch.nn.functional.silu(rcu[0](f4)))
                    f4 = f4 + h2
                y = f4[0].permute(1, 2, 0).reshape(-1, W)
            hp = self.hp(y)
            gh, gw = order[3]
            big = hp.view(gh, gw, -1).repeat_interleave(
                Hh // gh, 0).repeat_interleave(Ww // gw, 1)
            return (big @ self.hl.weight.t()).reshape(Hh, Ww)

    return DPTTiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dpt-gt")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", default=None,
                    help="weights .pt to load + eval-only (no training — "
                    "metrics reflect existing weights, e.g. after a crash fix)")
    ap.add_argument("--train", nargs="+", type=int,
                    default=list(range(6)) + list(range(12, 18)))
    ap.add_argument("--heldout", nargs="+", type=int,
                    default=list(range(8, 12)) + list(range(20, 24)))
    args = ap.parse_args()

    import numpy as np
    import torch
    DPTTiny = _define_model(torch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    import synth_depth as S
    scenes = {}
    for i in args.train + args.heldout:
        rgb, depth, meta = S.render_scene(i)
        scenes[i] = (torch.tensor(rgb.astype(np.float32) / 255.0,
                                  device=device),
                     torch.tensor(np.log(depth).astype(np.float32),
                                  device=device),
                     meta)
    H0, W0, _ = scenes[args.train[0]][0].shape
    nrows, ncols = H0 // 16, W0 // 16
    net = DPTTiny(args.width, nrows, ncols, args.blocks,
                  args.heads).to(device).float()
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} grid={nrows}x{ncols}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    t0 = time.time()
    if args.resume:
        net.load_state_dict(torch.load(os.path.expanduser(args.resume),
                                       map_location=device, weights_only=True))
        print(f"resumed {args.resume} (eval-only, no training)", flush=True)
    else:
        for it in range(args.steps):
            opt.zero_grad()
            tot = 0.0
            for n in args.train:
                # per-scene backward (one live graph at a time — memory-safe;
                # grads identical to a joint backward over the sum).
                rgb, gt, _ = scenes[n]
                l = (net(rgb) - gt).abs().mean()
                l.backward()
                tot += float(l)
            loss = tot / len(args.train)
            opt.step()
            if it % 500 == 0 or it == args.steps - 1:
                print(f"iter {it}: loss {loss:.4f}", flush=True)
        print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    rep = {"tag": args.tag, "arch": "dpt-tiny-torch", "width": args.width,
           "blocks": args.blocks, "heads": args.heads, "patch": 16,
           "loss": "logGT-MAE", "steps": args.steps, "lr": args.lr,
           "seed": args.seed, "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    if args.resume:
        rep["resumed_from"] = os.path.expanduser(args.resume)
    net.eval()
    with torch.no_grad():
        for split, idxs in (("train", args.train), ("heldout", args.heldout)):
            for i in idxs:
                rgb, gt, meta = scenes[i]
                p = torch.exp(net(rgb)).cpu().numpy()
                t = np.exp(gt.cpu().numpy())
                im = rgb.cpu().numpy()
                masks = H.strata_masks(t, im)
                per = {}
                for k in H.STRATA:
                    m = masks[k]
                    if int(m.sum()) == 0:
                        per[k] = None
                    else:
                        per[k] = float(np.abs(p - t)[m].mean() / t[m].mean())
                rep["frames"][f"s{i:02}"] = {
                    "split": split, "strata": per,
                    "constr": {a: meta[a] for a in
                               ("count", "occlusion", "range", "textured")}}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: (None if v is None else round(v, 4))
                               for k, v in fr["strata"].items()}, flush=True)
    outp = os.path.join(FIXDIR, f"dpt_run_{args.tag}.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
