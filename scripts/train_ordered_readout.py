"""Ordered-readout depth student (torch, outside listings).

Host COPY of train_attn_student.JointAttn (Block/RMSNorm verbatim,
same training loop/recipe) with the ONLY delta marked DELTA below:
per-token discounted-summary readout over a frozen traversal
(raster shared / nearest-first / farthest-first per anchor) replacing
raw block-outs; head unchanged. nctx=0 control config (no ctx
Parameter at all). New params (alpha-logit const 2.0 — no RNG draw;
summary Linear LAST — init-order discipline: host draws bit-identical
to the noreg control, verified pre-training, see spec).
Deployed-op citations: MATMUL/ADD/MUL (cell+mixer), SILU (head),
GATHER (ordering), CONCAT n/a. No LAYERNORM/GELU/CONV anywhere.
Loss: affine-invariant MAE (no-grad lstsq align, stride-7 sample),
deterministic stack (fixed seed). Weights out-of-tree
(~/student_runs/<tag>/); metrics -> OUTDIR/arm_eval.json (NOT
attn_run.json — that primary is never touched by this script).
Usage: ~/marigold-env/bin/python scripts/train_ordered_readout.py
  --tag ordered-near --traversal near [frozen recipe flags]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_base as BASE
from train_attn_student import load_frame

FIXDIR = os.path.join(ROOT, "samples", "marigold")
OUTBASE = os.path.expanduser("~/student_runs")


def build_order(nrows, ncols, kind):
    import numpy as np
    S = nrows * ncols
    rr, cc = np.divmod(np.arange(S), ncols)
    if kind == "raster":
        return np.tile(np.arange(S), (S, 1)).astype(np.int64)
    d2 = (rr[:, None] - rr[None, :]) ** 2 + (cc[:, None] - cc[None, :]) ** 2
    idx = np.tile(np.arange(S), (S, 1))
    if kind == "near":
        key = list(zip(d2.tolist(), idx.tolist()))
        return np.array([sorted(range(S), key=lambda j: (d2[i, j], j))
                         for i in range(S)], dtype=np.int64)
    if kind == "far":
        return np.array([sorted(range(S), key=lambda j: (-d2[i, j], j))
                         for i in range(S)], dtype=np.int64)
    raise ValueError(f"unknown traversal {kind}")


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
        def __init__(self, W, Hh=1):
            super().__init__()
            assert W % Hh == 0, (W, Hh)
            self.Hh = Hh
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
            S, W = x.shape
            Hh, Dh = self.Hh, W // self.Hh
            h = self.n1(x)
            q = self.q(h).view(S, Hh, Dh).transpose(0, 1)
            k = self.k(h).view(S, Hh, Dh).transpose(0, 1)
            v = self.v(h).view(S, Hh, Dh).transpose(0, 1)
            a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(Dh))
            a = torch.softmax(a, dim=-1)
            o = (a @ v).transpose(0, 1).reshape(S, W)
            x = x + self.o(o)
            h = self.n2(x)
            x = x + self.m2(torch.nn.functional.silu(self.m1(h)))
            return x

    class JointAttnReadout(torch.nn.Module):
        # DELTA vs train_attn_student.JointAttn: no ctx Parameter (nctx=0
        # control); traversal ORDER buffer + discounted-summary readout
        # (alpha logit, summary mixer) between blocks and head. Block,
        # tok/row/col/head code verbatim; new params drawn LAST.
        def __init__(self, W, nrows, ncols, B, Hh=1, traversal="raster"):
            super().__init__()
            self.W = W
            self.tok = torch.nn.Linear(3, W)
            self.row = torch.nn.Embedding(nrows, W)
            self.col = torch.nn.Embedding(ncols, W)
            self.blocks = torch.nn.ModuleList([Block(W, Hh) for _ in range(B)])
            self.h1 = torch.nn.Linear(W + 3, W)
            self.h2 = torch.nn.Linear(W, 1)
            self.alpha_logit = torch.nn.Parameter(torch.tensor(2.0))
            self.mix = torch.nn.Linear(W, W)
            order = build_order(nrows, ncols, traversal)
            self.register_buffer("order", torch.tensor(order, dtype=torch.long))

        def encode(self, rgb):
            """rgb (H,W,3) -> per-token traversal summaries (T,W)."""
            H, W3, _ = rgb.shape
            W = self.W
            nrows, ncols = self.row.num_embeddings, self.col.num_embeddings
            cy = torch.arange(nrows, device=rgb.device) * (H // nrows) + (H // nrows) // 2
            cx = torch.arange(ncols, device=rgb.device) * (W3 // ncols) + (W3 // ncols) // 2
            gy, gx = torch.meshgrid(cy, cx, indexing="ij")
            toks = rgb[gy.reshape(-1), gx.reshape(-1)]
            pos = (self.row.weight.repeat_interleave(ncols, dim=0)
                   + self.col.weight.repeat(nrows, 1))
            seq = self.tok(toks) + pos
            for b in self.blocks:
                seq = b(seq)
            S = seq.shape[0]
            alpha = torch.sigmoid(self.alpha_logit)
            pw = torch.pow(alpha, torch.arange(S - 1, -1, -1,
                                               device=seq.device,
                                               dtype=seq.dtype))
            Xq = seq[self.order]
            summ = (pw[None, :, None] * Xq).sum(dim=1)
            return self.mix(summ)

        def head(self, tokouts, rgb, idx):
            """tokouts (T,W), rgb (H,W,3), idx (H,W) long -> (H,W) depth."""
            g = tokouts[idx]
            f = torch.cat([g, rgb], dim=-1)
            return self.h2(torch.nn.functional.silu(self.h1(f))).squeeze(-1)

    return JointAttnReadout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ordered-near")
    ap.add_argument("--traversal", choices=["raster", "near", "far"],
                    default="near")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--schedule", choices=["const", "cosine"], default="cosine")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", nargs="+",
                    default=["church", "dogs", "squirrel", "train"])
    ap.add_argument("--heldout", nargs="+",
                    default=["geo_webcam_0_combined", "geo_webcam_1_combined"])
    args = ap.parse_args()

    import numpy as np
    import torch
    JointAttnReadout = _define_model(torch)
    BASE.seed_all(args.seed)
    device = BASE.pick_device()

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
    sidx = BASE.stride_index(H0 * W0, 7, device)

    def enc(n):
        rgb, _ = frames[n]
        return net.encode(rgb), pidx

    net = JointAttnReadout(args.width, nrows, ncols, args.blocks,
                           args.heads, args.traversal).to(device).float()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} traversal={args.traversal} "
          f"seq={nrows * ncols}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    Tcat = torch.cat([frames[n][1].reshape(-1)[sidx] for n in args.train])
    timer = BASE.PhaseTimer()
    t0 = time.time()
    for it in range(args.steps):
        BASE.apply_schedule(opt, args.schedule, args.lr, it, args.steps)
        opt.zero_grad()
        ps = []
        for n in args.train:
            rgb, _ = frames[n]
            with timer.time("forward"):
                to, hidx = enc(n)
                p = net.head(to, rgb, hidx).reshape(-1)[sidx].double()
            ps.append(p)
        p = torch.cat(ps)
        with timer.time("backward"):
            with torch.no_grad():
                A = torch.stack([p.detach(), torch.ones_like(p)], 1)
                ab, _, _, _ = torch.linalg.lstsq(A, Tcat)
                a, b = ab[0], ab[1]
            loss = (a * p + b - Tcat).abs().mean()
            loss.backward()
        opt.step()
        if BASE.cadence(it, args.steps, 500):
            print(f"iter {it}: loss {loss.item():.4f} "
                  f"alpha {torch.sigmoid(net.alpha_logit).item():.4f}",
                  flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "traversal": args.traversal,
           "width": args.width, "blocks": args.blocks,
           "heads": args.heads, "nctx": 0,
           "stride": args.stride, "steps": args.steps,
           "lr": args.lr, "schedule": args.schedule, "seed": args.seed,
           "field": "full-frame",
           "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "nparams": nparams,
           "alpha": float(torch.sigmoid(net.alpha_logit).item()),
           "frames": {}}
    net.eval()
    with timer.time("eval"):
        with torch.no_grad():
            for split, names in (("train", args.train), ("heldout", args.heldout)):
                for n in names:
                    rgb, t = frames[n]
                    to, hidx = enc(n)
                    p = net.head(to, rgb, hidx).cpu().numpy()
                    t = t.cpu().numpy()
                    Hh, Ww = next(r["shape"] for r in man["fixtures"] if r["name"] == n)
                    masks = H.strata_masks(t.reshape(Hh, Ww), rgb.cpu().numpy().reshape(Hh, Ww, 3))
                    per = {k: H.align_mae(p.reshape(Hh, Ww), t.reshape(Hh, Ww), masks[k])
                           for k in H.STRATA}
                    rep["frames"][n] = {"split": split, "strata": per}
    timer.summary()
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: round(v, 4) for k, v in fr["strata"].items()},
              flush=True)
    json.dump(rep, open(os.path.join(outdir, "arm_eval.json"), "w"), indent=1)
    print(f"wrote {outdir}/arm_eval.json (out-of-tree; attn_run.json untouched)",
          flush=True)


if __name__ == "__main__":
    main()
