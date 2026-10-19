"""Tiny joint-attention depth student (student Phase 8): torch, outside listings.

Architecture jattn64b2 per docs/STUDENT.md Phase-8 constitution (frozen):
stride-32 cell-center token grid + factored row/col pos (ADD) + C=4
learned const-text tokens (BETA) -> B=2 joint self-attention blocks
(RMSNorm, explicit-softmax attention, SILU MLP) -> nearest-cell
GATHER broadcast head (token + pixel RGB -> MLP -> depth).
Deployed-op citations (README table): MATMUL/ADD/MUL F, RMSNORM F,
SILU F, CONCAT/GATHER F, BMMV F, SOFTMAX_WIDE F, BETA F. No
LAYERNORM/GELU/CONV/GroupNorm/SIN-COS/INTERP anywhere.
Loss: affine-invariant MAE (no-grad lstsq align, stride-7 sample),
deterministic stack (fixed seed). Weights out-of-tree
(~/student_runs/<tag>/); metrics + config -> samples/marigold/attn_run.json.
Usage: python3 scripts/train_attn_student.py --tag jattn64b2
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")


def load_frame(name):
    import numpy as np
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    row = next(r for r in man["fixtures"] if r["name"] == name)
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(__import__("glob").glob(
        os.path.join(FIXDIR, "inputs", name + ".*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    return im, t


def _define_model(torch):
    """Model classes (torch lives here and in main only, never module top)."""
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class Block(torch.nn.Module):
        def __init__(self, W):
            super().__init__()
            self.n1 = RMSNorm(W)
            self.q = torch.nn.Linear(W, W)
            self.k = torch.nn.Linear(W, W)
            self.v = torch.nn.Linear(W, W)
            self.o = torch.nn.Linear(W, W)
            self.n2 = RMSNorm(W)
            self.m1 = torch.nn.Linear(W, 4 * W)
            self.m2 = torch.nn.Linear(4 * W, W)

        def forward(self, x):
            import math
            h = self.n1(x)
            q, k, v = self.q(h), self.k(h), self.v(h)
            a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
            a = torch.softmax(a, dim=-1)
            x = x + self.o(a @ v)
            h = self.n2(x)
            x = x + self.m2(torch.nn.functional.silu(self.m1(h)))
            return x

    class JointAttn(torch.nn.Module):
        def __init__(self, W, nrows, ncols, C, B):
            super().__init__()
            self.W = W
            self.tok = torch.nn.Linear(3, W)
            self.row = torch.nn.Embedding(nrows, W)
            self.col = torch.nn.Embedding(ncols, W)
            self.ctx = torch.nn.Parameter(torch.randn(C, W) * 0.02)
            self.blocks = torch.nn.ModuleList([Block(W) for _ in range(B)])
            self.h1 = torch.nn.Linear(W + 3, W)
            self.h2 = torch.nn.Linear(W, 1)

        def encode(self, rgb):
            """rgb: (H,W,3) tensor -> image-token outs (T,W)."""
            H, W3, _ = rgb.shape
            W = self.W
            nrows, ncols = self.row.num_embeddings, self.col.num_embeddings
            cy = torch.arange(nrows, device=rgb.device) * (H // nrows) + (H // nrows) // 2
            cx = torch.arange(ncols, device=rgb.device) * (W3 // ncols) + (W3 // ncols) // 2
            gy, gx = torch.meshgrid(cy, cx, indexing="ij")
            toks = rgb[gy.reshape(-1), gx.reshape(-1)]  # (T,3)
            pos = (self.row.weight.repeat_interleave(ncols, dim=0)
                   + self.col.weight.repeat(nrows, 1))
            temb = self.tok(toks) + pos
            seq = torch.cat([self.ctx, temb], dim=0)
            for b in self.blocks:
                seq = b(seq)
            return seq[self.ctx.shape[0]:]

        def head(self, tokouts, rgb, idx):
            """tokouts (T,W), rgb (H,W,3), idx (H,W) long -> (H,W) depth."""
            g = tokouts[idx]
            f = torch.cat([g, rgb], dim=-1)
            return self.h2(torch.nn.functional.silu(self.h1(f))).squeeze(-1)

    return JointAttn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="jattn64b2")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--nctx", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", nargs="+",
                    default=["church", "dogs", "squirrel", "train"])
    ap.add_argument("--heldout", nargs="+",
                    default=["geo_webcam_0_combined", "geo_webcam_1_combined"])
    args = ap.parse_args()

    import numpy as np
    import torch
    JointAttn = _define_model(torch)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    frames = {}
    for n in args.train + args.heldout:
        im, t = load_frame(n)
        frames[n] = (torch.tensor(im, dtype=torch.float32, device=device),
                     torch.tensor(t, dtype=torch.float64, device=device))
    H0, W0 = frames[args.train[0]][0].shape[:2]
    nrows, ncols = H0 // args.stride, W0 // args.stride
    yy, xx = torch.meshgrid(torch.arange(H0, device=device),
                            torch.arange(W0, device=device), indexing="ij")
    pidx = ((yy // args.stride) * ncols + (xx // args.stride)).long()
    sidx = torch.arange(0, H0 * W0, 7, device=device)

    net = JointAttn(args.width, nrows, ncols, args.nctx, args.blocks).to(device).float()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} grid={nrows}x{ncols} seq={nrows * ncols + args.nctx}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    Tcat = torch.cat([frames[n][1].reshape(-1)[sidx] for n in args.train])
    t0 = time.time()
    for it in range(args.steps):
        opt.zero_grad()
        ps = []
        for n in args.train:
            rgb, _ = frames[n]
            to = net.encode(rgb)
            p = net.head(to, rgb, pidx).reshape(-1)[sidx].double()
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
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "width": args.width, "blocks": args.blocks,
           "stride": args.stride, "nctx": args.nctx, "steps": args.steps,
           "lr": args.lr, "seed": args.seed, "field": "full-frame",
           "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams, "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, names in (("train", args.train), ("heldout", args.heldout)):
            for n in names:
                rgb, t = frames[n]
                to = net.encode(rgb)
                p = net.head(to, rgb, pidx).cpu().numpy()
                t = t.cpu().numpy()
                Hh, Ww = next(r["shape"] for r in man["fixtures"] if r["name"] == n)
                masks = H.strata_masks(t.reshape(Hh, Ww), rgb.cpu().numpy().reshape(Hh, Ww, 3))
                per = {k: H.align_mae(p.reshape(Hh, Ww), t.reshape(Hh, Ww), masks[k])
                       for k in H.STRATA}
                rep["frames"][n] = {"split": split, "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: round(v, 4) for k, v in fr["strata"].items()},
              flush=True)
    json.dump(rep, open(os.path.join(FIXDIR, "attn_run.json"), "w"), indent=1)
    print("wrote samples/marigold/attn_run.json", flush=True)


if __name__ == "__main__":
    main()
