"""DiT-tiny faithful-machinery student (student Phase 19): torch, outside listings.

Teacher-faithful package at student scale (A1/A2/A4 cited, not
re-derived): stride-16 full-RGB-patch tokens (768->W) + learned
row/col pos (ADD) + AdaLN-Zero modulation from a learned const
vector (no timestep, no text stream) + B LayerNorm blocks (unfused
multi-head self-attention, GELU-form SILU MLP, gated residuals) +
nearest-cell broadcast head (family head as built). VAE out:
direct depth prediction. Training: log-normalized targets +
aligned-MAE + Sobel-gradient aux (training-time only, zero
inference cost). Weights out-of-tree; metrics -> attn_run_<tag>.
Usage: python3 scripts/train_dit_student.py --tag dit64b2 --blocks 2
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


def _define_model(torch):
    """Model classes (torch lives here and in main only, never module top)."""

    def geluform(x):
        return torch.nn.functional.silu(x * 1.702) / 1.702

    class Block(torch.nn.Module):
        def __init__(self, W, Hh=4):
            super().__init__()
            assert W % Hh == 0, (W, Hh)
            self.Hh = Hh
            self.mod = torch.nn.Linear(W, 6 * W)
            self.n1 = torch.nn.LayerNorm(W, elementwise_affine=False)
            self.q = torch.nn.Linear(W, W)
            self.k = torch.nn.Linear(W, W)
            self.v = torch.nn.Linear(W, W)
            self.o = torch.nn.Linear(W, W)
            self.n2 = torch.nn.LayerNorm(W, elementwise_affine=False)
            self.m1 = torch.nn.Linear(W, 4 * W)
            self.m2 = torch.nn.Linear(4 * W, W)

        def forward(self, x, temb):
            import math
            S, W = x.shape
            Hh, Dh = self.Hh, W // self.Hh
            sh1, sc1, g1, sh2, sc2, g2 = self.mod(
                torch.nn.functional.silu(temb)).chunk(6, dim=-1)
            h = self.n1(x) * (1 + sc1) + sh1
            q = self.q(h).view(S, Hh, Dh).transpose(0, 1)
            k = self.k(h).view(S, Hh, Dh).transpose(0, 1)
            v = self.v(h).view(S, Hh, Dh).transpose(0, 1)
            a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(Dh))
            a = torch.softmax(a, dim=-1)
            o = (a @ v).transpose(0, 1).reshape(S, W)
            x = x + g1 * self.o(o)
            h = self.n2(x) * (1 + sc2) + sh2
            x = x + g2 * self.m2(geluform(self.m1(h)))
            return x

    class DiTTiny(torch.nn.Module):
        def __init__(self, W, nrows, ncols, B, Hh=4, patch=16):
            super().__init__()
            self.W = W
            self.patch = patch
            self.pemb = torch.nn.Linear(3 * patch * patch, W)
            self.row = torch.nn.Embedding(nrows, W)
            self.col = torch.nn.Embedding(ncols, W)
            self.temb = torch.nn.Parameter(torch.zeros(W))
            self.blocks = torch.nn.ModuleList([Block(W, Hh) for _ in range(B)])
            self.h1 = torch.nn.Linear(W + 3, W)
            self.h2 = torch.nn.Linear(W, 1)

        def encode(self, rgb):
            """rgb (H,W,3) -> image-token outs (T,W)."""
            H, W3, _ = rgb.shape
            W, P = self.W, self.patch
            nrows, ncols = self.row.num_embeddings, self.col.num_embeddings
            toks = rgb.unfold(0, P, P).unfold(1, P, P)
            toks = toks.permute(0, 1, 3, 4, 2).reshape(-1, 3 * P * P)
            pos = (self.row.weight.repeat_interleave(ncols, dim=0)
                   + self.col.weight.repeat(nrows, 1))
            seq = self.pemb(toks) + pos
            for b in self.blocks:
                seq = b(seq, self.temb)
            return seq

        def head(self, tokouts, rgb, idx):
            """tokouts (T,W), rgb (H,W,3), idx (H,W) long -> (H,W)."""
            g = tokouts[idx]
            f = torch.cat([g, rgb], dim=-1)
            return self.h2(torch.nn.functional.silu(self.h1(f))).squeeze(-1)

    return DiTTiny


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


SOBEL_X = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
SOBEL_Y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dit64b2")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--schedule", choices=["const", "cosine"], default="cosine")
    ap.add_argument("--grad-w", type=float, default=0.1)
    ap.add_argument("--aux-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--train", nargs="+",
                    default=["church", "dogs", "squirrel", "train"])
    ap.add_argument("--heldout", nargs="+",
                    default=["geo_webcam_0_combined", "geo_webcam_1_combined"])
    args = ap.parse_args()

    import numpy as np
    import torch
    DiTTiny = _define_model(torch)
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

    # log-normalization constants from train split (fixed, banked).
    # shifted-log: banked teacher depths are signed-normalized
    # (min -1), so plain log(t) is unimplementable; log(t-lo+1)
    # keeps the compression literally (constitution correction).
    tall = np.concatenate([frames[n][1].cpu().numpy().reshape(-1)
                           for n in args.train])
    lo = float(tall.min())
    lt = np.log(np.maximum(tall - lo + 1.0, 1e-9))
    mu = float(lt.mean())
    sigma = float(lt.std())
    print(f"lognorm lo={lo:.4f} mu={mu:.4f} sigma={sigma:.4f}", flush=True)

    def to_z(t_np):
        return (np.log(np.maximum(t_np - lo + 1.0, 1e-9)) - mu) / sigma

    Z = {n: (torch.tensor(to_z(frames[n][1].cpu().numpy()),
                          dtype=torch.float64, device=device))
         for n in frames}

    net = DiTTiny(args.width, nrows, ncols, args.blocks,
                  args.heads).to(device).float()
    nparams = sum(p.numel() for p in net.parameters())
    print(f"params={nparams} grid={nrows}x{ncols} seq={nrows * ncols}",
          flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    kx = torch.tensor(SOBEL_X, dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = torch.tensor(SOBEL_Y, dtype=torch.float32, device=device).view(1, 1, 3, 3)

    def grad_mag(f2d):
        f = f2d.unsqueeze(0).unsqueeze(0).float()
        gx = torch.nn.functional.conv2d(f, kx, padding=1)
        gy = torch.nn.functional.conv2d(f, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze()

    Tcat = torch.cat([Z[n].reshape(-1)[sidx] for n in args.train])
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
        aux = torch.zeros((), device=device)
        if it % args.aux_every == 0 or it == args.steps - 1:
            terms = []
            for n in args.train:
                rgb, _ = frames[n]
                to = net.encode(rgb)
                pf = net.head(to, rgb, pidx).double()
                pa = a * pf + b
                zt = Z[n].reshape(rgb.shape[0], rgb.shape[1])
                terms.append((grad_mag(pa.reshape(zt.shape))
                              - grad_mag(zt.float())).abs().mean())
            aux = sum(terms) / len(terms)
        (loss + args.grad_w * aux).backward()
        opt.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {loss.item():.4f} aux {float(aux):.4f}",
                  flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)
    assert bool(torch.isfinite(loss).all()), f"final loss non-finite: {loss.item()}"

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "arch": "dit-tiny", "width": args.width,
           "blocks": args.blocks, "heads": args.heads, "patch": 16,
           "adaln": "const-vector", "geluform": "silu(x*1.702)/1.702",
           "gradloss": args.grad_w, "lognorm": {"lo": lo, "mu": mu, "sigma": sigma},
           "stride": args.stride, "nctx": 0, "steps": args.steps,
           "lr": args.lr, "schedule": args.schedule, "seed": args.seed,
           "field": "full-frame",
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
                pz = net.head(to, rgb, pidx).cpu().numpy()
                # range hygiene (teacher depths live ~[-1,1]; clip only
                # touches insane values, never sane models)
                zl = np.clip(np.asarray(rep["lognorm"]["sigma"]) * pz
                             + np.asarray(rep["lognorm"]["mu"]),
                             np.log(1e-9), np.log(1e4))
                p = np.exp(zl) + np.asarray(rep["lognorm"]["lo"]) - 1.0
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
