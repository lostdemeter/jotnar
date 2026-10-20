"""DPT-tiny training on Hypersim GT (analytic truth, not mimicry).

Res-general DPT-tiny torch mirroring the BUILT listing dataflow
(patch-16 tokens, 6 plain-ViT blocks H4, reassemble strided/
broadcast + 1x1 projs, RefineNet fusion as replicate-pad conv2d —
mathematically identical to the listing GATHER-tap form per the
P1-equivalence probe —, direct head). Assembly lowering is a
later round, explicitly not this one.
Loss: log-GT MAE on stride-7 sample (direct regression, no
alignment — exact truth needs none; gradient aux DROPPED with
reason: aux-pin measured Phases 19-20). Eval: exp + AbsRel per
harness stratum (masks & valid intersect), full-frame at end.
Weights out-of-tree (~/student_runs/<tag>/); metrics ->
samples/marigold/dpt_run_<tag>.json.
Usage: python3 scripts/train_dpt_gt.py --tag dpt-hyper
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

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
            import math
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
                if li == 0:
                    f = projs[li]
                else:
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
    ap.add_argument("--tag", default="dpt-hyper")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mb", type=int, default=2)
    args = ap.parse_args()

    import numpy as np
    import torch
    import train_hyper as TH
    DPTTiny = _define_model(torch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    train_idx = [(s, TH.scene_frames(s)[k])
                 for s in TH.TRAIN_SCENES for k in range(TH.NFR)]
    held_idx = [(s, TH.scene_frames(s)[k])
                for s in TH.HELD_SCENES for k in range(TH.NFR)]
    H0, W0, _ = TH.load_pair(*train_idx[0])[0].shape
    nrows, ncols = H0 // 16, W0 // 16
    net = DPTTiny(args.width, nrows, ncols, args.blocks,
                  args.heads).to(device).float()
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} grid={nrows}x{ncols}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    sidx_cache = {}
    t0 = time.time()
    for it in range(args.steps):
        opt.zero_grad()
        tot = 0.0
        mb = [train_idx[(it * args.mb + k) % len(train_idx)]
              for k in range(args.mb)]
        for (s, f) in mb:
            im, depth, valid = TH.load_pair(s, f)
            key = depth.shape
            if key not in sidx_cache:
                sidx_cache[key] = torch.arange(
                    0, depth.size, 7, device=device)
            sidx = sidx_cache[key]
            rgbt = torch.tensor(im.astype(np.float32), device=device)
            gt = torch.tensor(np.log(depth).astype(np.float32),
                              device=device).reshape(-1)[sidx]
            vm = torch.tensor(valid.reshape(-1)[sidx.cpu().numpy()],
                              device=device)
            p = net(rgbt).reshape(-1)[sidx].double()
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
    BARS = {"near": 0.09, "far": 0.09, "planar": 0.09,
            "nonplanar": 0.08, "tex": 0.07, "smooth": 0.09}
    rep = {"tag": args.tag, "arch": "dpt-tiny-torch", "width": args.width,
           "blocks": args.blocks, "heads": args.heads, "patch": 16,
           "loss": "logGT-MAE-noaux", "steps": args.steps, "lr": args.lr,
           "seed": args.seed,
           "train": [f"{s}/fr{f:04}" for s, f in train_idx],
           "heldout": [f"{s}/fr{f:04}" for s, f in held_idx],
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, idxs in (("train", train_idx), ("heldout", held_idx)):
            for (s, f) in idxs:
                im, depth, valid = TH.load_pair(s, f)
                rgbt = torch.tensor(im.astype(np.float32), device=device)
                p = torch.exp(net(rgbt)).cpu().numpy()
                masks = H.strata_masks(depth, im)
                per = {}
                for k in H.STRATA:
                    m = masks[k] & valid
                    per[k] = (None if int(m.sum()) == 0 else float(
                        np.abs(p[m] - depth[m]).mean() / depth[m].mean()))
                rep["frames"][f"{s}_fr{f:04}"] = {"split": split,
                                                  "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: (None if v is None else round(v, 4))
                               for k, v in fr["strata"].items()}, flush=True)
    outp = os.path.join(FIXDIR := os.path.join(ROOT, "samples", "marigold"),
                        f"dpt_run_{args.tag}.json")
    json.dump(rep, open(outp, "w"), indent=1)
    print(f"wrote {outp}", flush=True)


if __name__ == "__main__":
    main()
