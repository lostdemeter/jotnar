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


def load_frame(name, patch=1):
    import numpy as np
    from PIL import Image
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    row = next(r for r in man["fixtures"] if r["name"] == name)
    t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
    cand = sorted(__import__("glob").glob(
        os.path.join(FIXDIR, "inputs", name + ".*")))
    im = np.asarray(Image.open(cand[0]).convert("RGB")
                    .resize((t.shape[1], t.shape[0]))) / 255.0
    if patch == 1:
        return im.reshape(-1, 3), im, t.reshape(-1)
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
    args = ap.parse_args()

    import numpy as np
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    Xtr, Ttr = [], []
    for n in args.train:
        x, _, t = load_frame(n, args.patch)
        Xtr.append(x)
        Ttr.append(t)
    Xtr = torch.tensor(np.concatenate(Xtr), dtype=torch.float32, device=device)
    Ttr = torch.tensor(np.concatenate(Ttr), dtype=torch.float64, device=device)
    # deterministic stride sample (fixed order, no RNG in-loop)
    idx = torch.arange(0, Xtr.shape[0], 7, device=device)
    Xs, Ts = Xtr[idx], Ttr[idx]
    print(f"train px: {Xs.shape[0]} (stride-7 of {Xtr.shape[0]})", flush=True)

    W = args.width
    Din = 27 if args.patch == 3 else 3
    net = torch.nn.Sequential(
        torch.nn.Linear(Din, W), torch.nn.SiLU(),
        torch.nn.Linear(W, W), torch.nn.SiLU(),
        torch.nn.Linear(W, 1)).to(device).float()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
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
        loss.backward()
        opt.step()
        if it % 500 == 0 or it == args.steps - 1:
            print(f"iter {it}: loss {loss.item():.4f}", flush=True)
    print(f"train wall: {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(OUTBASE, args.tag)
    os.makedirs(outdir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(outdir, "weights.pt"))

    # -- evaluate per stratum (same rules/alignment as the harness) --------
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import test_strata_marigold as H
    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    rep = {"tag": args.tag, "width": W, "steps": args.steps, "lr": args.lr,
           "seed": args.seed, "patch": args.patch,
           "train": args.train, "heldout": args.heldout,
           "torch": torch.__version__, "device": device,
           "weights": os.path.join(outdir, "weights.pt"),
           "frames": {}}
    net.eval()
    with torch.no_grad():
        for split, names in (("train", args.train), ("heldout", args.heldout)):
            for n in names:
                x, im3, t = load_frame(n, args.patch)
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
