"""Depth student training (student Phase 2): torch, outside the listings.

Per-pixel MLP: RGB(3) -> W -> W -> 1, SiLU activations, no norms.
Every op in the deployed form has an emitted CUDA/F pattern (MATMUL,
ADD, SILU -- README table is the law; LAYERNORM/CONV gaps bite here
by design, so the architecture uses neither). Affine-invariant loss:
least-squares align pred->teacher per frame (same doctrine as the
strata harness), MAE. Deterministic: fixed seed, single process,
no cudnn autotune paths (linear-only ops are deterministic).
Weights live out-of-tree (~/student_runs/<tag>/); metrics + config
written to samples/marigold/student_run.json (in-repo, committed).
Usage: python3 scripts/train_student.py --width 16 --steps 2000 --tag w16
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


def load_frame(name, patch=1, multi=False):
    import numpy as np
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    row = next(r for r in man["fixtures"] if r["name"] == name)
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(__import__("glob").glob(
        os.path.join(FIXDIR, "inputs", name + ".*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    if patch == 1 and not multi:
        return im.reshape(-1, 3), im, t.reshape(-1)
    if multi:
        Hh, Ww, _ = im.shape
        p = np.pad(im, ((16, 16), (16, 16), (0, 0)), mode="edge")
        parts = []
        yy, xx = np.mgrid[0:Hh, 0:Ww]
        # dense 3x3 (field 3)
        d3 = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
        parts.append(np.concatenate(
            [d3[i:i + Hh, j:j + Ww, :] for i in range(3) for j in range(3)],
            axis=-1))
        # stride-4 over 17x17 (field 17), stride-8 over 33x33 (field 33)
        for stride, half in ((4, 8), (8, 16)):
            off = 16 + np.arange(-half, half + 1, stride)
            grid = np.concatenate(
                [p[yy + dy, xx + dx, :] for dy in off for dx in off], axis=-1)
            parts.append(grid)
        return np.concatenate(parts, axis=-1).reshape(-1, 177), im, \
            t.reshape(-1)
    # 3x3 neighborhood im2col (edge-replicated): deployed form is
    # GATHER-neighborhood + MATMUL, both in-table (GATHER F, MATMUL F).
    # Stays feedforward-dense; no new opcodes, no norms, no conv.
    p = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode="edge")
    Hh, Ww, _ = im.shape
    cols = [p[i:i + Hh, j:j + Ww, :] for i in range(3) for j in range(3)]
    return np.concatenate(cols, axis=-1).reshape(-1, 27), im, t.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="w16")
    ap.add_argument("--train", nargs="+",
                    default=["church", "dogs", "squirrel", "train"])
    ap.add_argument("--heldout", nargs="+",
                    default=["geo_webcam_0_combined", "geo_webcam_1_combined"])
    ap.add_argument("--patch", type=int, default=1,
                    help="1 = per-pixel RGB(3); 3 = 3x3 neighborhood (27)")
    ap.add_argument("--multi", action="store_true",
                    help="multi-scale context: dense 3x3 (27) + stride-4 "
                    "17x17 (75) + stride-8 33x33 (75) = 177 dims, 33px "
                    "field covering the 10-60px detail scale from the "
                    "Phase-2 diagnosis (texture/nonplanar worst). "
                    "Deployed form: GATHER neighborhoods + CONCAT + "
                    "MATMUL/ADD/SILU MLP -- all emitted CUDA/F patterns "
                    "(GATHER F, CONCAT F, MATMUL F, ADD F, SILU F). "
                    "No norms, no conv, no new opcodes.")
    ap.add_argument("--perc", type=float, default=0.0,
                    help="perceptual weight (iREPA pattern): MSE between "
                    "L2-normalized frozen-DINOv2 token embeddings of "
                    "student vs teacher depth maps (3ch-replicated, "
                    "ImageNet-normalized). Training-time only, zero "
                    "inference cost. Adapted: our MLP has no spatial "
                    "intermediates, so we align encoder embeddings of "
                    "final outputs, not mid-layers. 0 = off.")
    ap.add_argument("--sink", type=float, default=0.0,
                    help="Sinkhorn block weight: per 8x8-grid block, "
                    "entropic OT (tau=0.1, K=5 log-domain iterations, "
                    "paper values first) between 32-bin student/teacher "
                    "depth histograms; mean over blocks. Permutation-"
                    "invariant within blocks, aimed at texture/detail. "
                    "0 = off.")
    ap.add_argument("--dinov2", default="~/student_runs/dinov2-base",
                    help="frozen DINOv2 path (out-of-tree, ungated "
                    "facebook/dinov2-base stand-in for gated DINOv3)")
    ap.add_argument("--aux-every", type=int, default=50,
                    help="aux-signal cadence in steps (full frames)")
    args = ap.parse_args()

    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    Xtr, Ttr = [], []
    for n in args.train:
        x, _, t = load_frame(n, args.patch, args.multi)
        Xtr.append(x)
        Ttr.append(t)
    Xtr = torch.tensor(np.concatenate(Xtr), dtype=torch.float32, device=device)
    Ttr = torch.tensor(np.concatenate(Ttr), dtype=torch.float64, device=device)
    # deterministic stride sample (fixed order, no RNG in-loop)
    idx = torch.arange(0, Xtr.shape[0], 7, device=device)
    Xs, Ts = Xtr[idx], Ttr[idx]
    print(f"train px: {Xs.shape[0]} (stride-7 of {Xtr.shape[0]})", flush=True)

    W = args.width
    Din = 177 if args.multi else (27 if args.patch == 3 else 3)
    net = torch.nn.Sequential(
        torch.nn.Linear(Din, W), torch.nn.SiLU(),
        torch.nn.Linear(W, W), torch.nn.SiLU(),
        torch.nn.Linear(W, 1)).to(device).float()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # -- aux signals: frozen DINOv2 embeddings of teacher maps (once) ------
    use_perc = args.perc > 0
    use_sink = args.sink > 0
    temb, dino = {}, None
    if use_perc:
        from transformers import Dinov2Model
        dino = Dinov2Model.from_pretrained(
            os.path.expanduser(args.dinov2)).to(device).float().eval()
        for p in dino.parameters():
            p.requires_grad_(False)
        man0 = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
        imean = torch.tensor([0.485, 0.456, 0.406], device=device)
        istd = torch.tensor([0.229, 0.224, 0.225], device=device)
        for n in args.train:
            row = next(r for r in man0["fixtures"] if r["name"] == n)
            tm = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float32)
            img = torch.tensor(np.stack([tm, tm, tm], 0)[None],
                               dtype=torch.float32, device=device)
            img = (img - img.min()) / max(float((img.max() - img.min())), 1e-6)
            img = (img - imean[None, :, None, None]) / istd[None, :, None, None]
            with torch.no_grad():
                e = dino(img).last_hidden_state[0, 1:]
                temb[n] = (e / e.norm(dim=-1, keepdim=True)).detach()
        print(f"perceptual: teacher embeddings cached "
              f"({list(temb)[0]}: {tuple(temb[args.train[0]].shape)})",
              flush=True)

    def frame_maps():
        """Full-frame student maps (differentiable -- NO no_grad here;
        caller backward()s through them)."""
        out = []
        for n in args.train:
            x, _, _ = load_frame(n, args.patch, args.multi)
            out.append(net(torch.tensor(x, dtype=torch.float32,
                                        device=device)).reshape(-1))
        return out

    def dinov2_embed(att_maps):
        """Frozen DINOv2 token embeddings WITH live gradients into the
        student maps (teacher embeddings precomputed detached; the DINO
        weights are frozen but the path student->embeddings stays
        attached -- that is the training signal). Input: list of
        (flat_attached_float64, H, W); output: list of
        (tokens,768) L2-normalized attached tensors."""
        embs = []
        for f, Hh, Ww in att_maps:
            img = f.reshape(1, 1, Hh, Ww).repeat(1, 3, 1, 1).float()
            img = (img - img.min()) / max(float((img.max() - img.min())), 1e-6)
            img = (img - imean[None, :, None, None]) / istd[None, :, None, None]
            e = dino(img).last_hidden_state[0, 1:]
            embs.append(e / e.norm(dim=-1, keepdim=True))
        return embs

    def sinkhorn_loss(s_maps_att, t_maps, ab):
        """Mean entropic-OT distance over an 8x8 block grid, as a torch
        scalar WITH live gradients (plan detached, cost attached --
        standard Sinkhorn-divergence practice). Per block: deterministic
        stride subsample (256 px), squared cost on affinely-aligned
        student (attached) vs teacher (detached) values, K=5 log-domain
        iterations (paper values: K=5, tau=0.1). loss = mean over blocks
        of <P_detached, C_attached>. Permutation-invariant within blocks
        by construction (coupling, not pointwise match)."""
        from scipy.special import logsumexp
        import math
        tau, K, S = 0.1, 5, 256
        a, b = float(ab[0]), float(ab[1])
        tot = torch.zeros((), device=device, dtype=torch.float64)
        nb = 0
        for (pm_t, Hh, Ww), tm in zip(s_maps_att, t_maps):
            th, tw = tm.shape
            assert (Hh, Ww) == (th, tw), ((Hh, Ww), tm.shape)
            for by in range(8):
                for bx in range(8):
                    y0, y1 = by * Hh // 8, (by + 1) * Hh // 8
                    x0, x1 = bx * Ww // 8, (bx + 1) * Ww // 8
                    yy, xx = torch.meshgrid(
                        torch.arange(y0, y1, device=device),
                        torch.arange(x0, x1, device=device), indexing="ij")
                    sel = (yy * Ww + xx).reshape(-1)
                    sel = sel[::max(1, len(sel) // S)][:S]
                    s_att = a * pm_t.reshape(-1)[sel] + b
                    t_v = torch.tensor(tm.reshape(-1)[sel.cpu().numpy()],
                                       device=device, dtype=torch.float64)
                    n = min(len(sel), len(t_v))
                    s_att = s_att[:n]
                    t_v = t_v[:n]
                    C_att = (s_att.unsqueeze(1) - t_v.unsqueeze(0)) ** 2
                    mx = float(C_att.detach().max())
                    C_att = C_att / (mx if mx > 0 else 1)
                    Cn = C_att.detach().double().cpu().numpy()
                    la = np.full(n, math.log(1.0 / n))
                    f = np.zeros(n)
                    g = np.zeros(n)
                    for _ in range(K):
                        f = tau * (la - logsumexp((g[None, :] - Cn) / tau,
                                                  axis=1))
                        g = tau * (la - logsumexp((f[:, None] - Cn) / tau,
                                                  axis=0))
                    P = np.exp((f[:, None] + g[None, :] - Cn) / tau)
                    P = P / max(P.sum(), 1e-300)
                    Pt = torch.tensor(P, device=device, dtype=torch.float64)
                    tot = tot + (Pt * C_att).sum() / (n * n)
                    nb += 1
        return tot / max(nb, 1)

    manT = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))

    def tmap(name):
        row = next(r for r in manT["fixtures"] if r["name"] == name)
        return np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)

    t0 = time.time()
    for it in range(args.steps):
        opt.zero_grad()
        p = net(Xs).reshape(-1).double()
        # alignment does not take gradients (standard scale-shift-
        # invariant practice; lstsq-backward OOMs at 400k rows)
        with torch.no_grad():
            A = torch.stack([p.detach(), torch.ones_like(p)], 1)
            ab, _, _, _ = torch.linalg.lstsq(A, Ts)
            a, b = ab[0], ab[1]
        loss = (a * p + b - Ts).abs().mean()
        aux = ""
        objs = [loss]
        if (use_perc or use_sink) and (it % args.aux_every == 0
                                       or it == args.steps - 1):
            # full-frame aux eval (deterministic cadence, no RNG)
            full = frame_maps()
            tmaps = [tmap(n) for n in args.train]
            shapes = [tm.shape for tm in tmaps]
            if use_perc:
                se = dinov2_embed([(f, Hh, Ww) for f, (Hh, Ww) in
                                   zip(full, shapes)])
                lp = sum((((e - temb[n][:e.shape[0]]) ** 2).mean()
                          for n, e in zip(args.train, se))) / len(full)
                objs.append(args.perc * lp)
                aux += f" perc {lp.item():.4f}"
            if use_sink:
                with torch.no_grad():
                    A2 = torch.stack(
                        [torch.cat([f.detach() for f in full]),
                         torch.ones(sum(f.numel() for f in full),
                                    device=device, dtype=torch.float64)], 1)
                    Tall = torch.cat(
                        [torch.tensor(tmap(n).ravel(), device=device)
                         for n in args.train])
                    ab2, _, _, _ = torch.linalg.lstsq(A2, Tall)
                ls = sinkhorn_loss(
                    [(f, Hh, Ww) for f, (Hh, Ww) in zip(full, shapes)],
                    tmaps, (ab2[0], ab2[1]))
                objs.append(args.sink * ls)
                aux += f" sink {ls.item():.4f}"
        # single backward over base + aux (shared parameters -- one graph)
        sum(objs).backward()
        opt.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {loss.item():.4f}{aux}", flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    # -- evaluate per stratum (same rules/alignment as the harness) --------
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "width": W, "steps": args.steps, "lr": args.lr,
           "seed": args.seed, "patch": args.patch, "multi": args.multi,
           "perc": args.perc, "sink": args.sink, "aux_every": args.aux_every,
           "dinov2": os.path.expanduser(args.dinov2) if args.perc > 0 else None,
           "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, names in (("train", args.train), ("heldout", args.heldout)):
            for n in names:
                x, im3, t = load_frame(n, args.patch, args.multi)
                p = net(torch.tensor(x, dtype=torch.float32,
                                     device=device)).reshape(-1).double()
                p = p.cpu().numpy()
                Hh, Ww = next(r["shape"] for r in man["fixtures"]
                              if r["name"] == n)
                masks = H.strata_masks(t.reshape(Hh, Ww),
                                       im3.reshape(Hh, Ww, 3))
                per = {k: H.align_mae(p.reshape(Hh, Ww), t.reshape(Hh, Ww), masks[k])
                       for k in H.STRATA}
                rep["frames"][n] = {"split": split, "strata": per}
    for n, fr in rep["frames"].items():
        print(n, fr["split"], {k: round(v, 4) for k, v in fr["strata"].items()},
              flush=True)
    json.dump(rep, open(os.path.join(FIXDIR, "student_run.json"), "w"), indent=1)
    print("wrote samples/marigold/student_run.json", flush=True)


if __name__ == "__main__":
    main()
