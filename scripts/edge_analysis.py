"""Edge-concentration analysis (student Phase 18): zero training.

Sobel-gradient edge seeds (>= frame 80th percentile) on banked
teacher depth; Chebyshev bands (edge<=2px, halo<=8px, interior) via
iterated numpy max-filter dilation; aligned-MAE per band with the
harness global fit, for 4 banked weight sets reloaded forward-only
on CPU (mscale64, jattn64b2, texmsc-2k, texmsc-8kcos) over the 6
real fixtures. Scalars banked to samples/marigold/edge_analysis.json.
Usage: python3 scripts/edge_analysis.py
"""
import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

FIXDIR = os.path.join(ROOT, "samples", "marigold")
SEED_PCTL = 80
EDGE_R, HALO_R = 2, 8


def sobel_mag(t):
    """Sobel gradient magnitude, numpy only (edge-replicated)."""
    import numpy as np
    p = np.pad(t, 1, mode="edge")
    gx = (-p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]
          + p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    return np.sqrt(gx * gx + gy * gy)


def dilate(b, r):
    """Chebyshev dilation radius r via iterated 8-neighbor OR (exact)."""
    import numpy as np
    m = b.copy()
    for _ in range(r):
        p = np.pad(m, 1, mode="edge")
        m = (p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
             | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
             | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:])
    return m


def edge_bands(t):
    """-> {edge, halo, interior} bool masks, disjoint, covering."""
    import numpy as np
    mag = sobel_mag(t)
    seeds = mag >= np.percentile(mag, SEED_PCTL)
    edge = dilate(seeds, EDGE_R)
    halo = dilate(seeds, HALO_R) & ~edge
    interior = ~(edge | halo)
    return {"edge": edge, "halo": halo, "interior": interior}


def main():
    import glob
    import numpy as np
    import torch
    import test_strata_marigold as H
    import train_student as T
    import train_attn_student as JA
    from PIL import Image

    man = json.load(open(os.path.join(FIXDIR, "teacher_manifest.json")))
    real = [r for r in man["fixtures"] if not r.get("synthetic")]
    assert len(real) == 6, len(real)

    JAMod = JA._define_model(torch)

    def load_mlp(tag):
        sd = torch.load(os.path.expanduser(f"~/student_runs/{tag}/weights.pt"),
                        map_location="cpu", weights_only=True)
        n = torch.nn.Sequential(
            torch.nn.Linear(177, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 1)).float().eval()
        n.load_state_dict(sd)
        return n

    mlps = {t: load_mlp(t) for t in ("mscale64", "texmsc-2k", "texmsc-8kcos")}
    sdj = torch.load(os.path.expanduser("~/student_runs/jattn64b2/weights.pt"),
                     map_location="cpu", weights_only=True)

    out = {"rule": {"seeds": "sobel-mag >= frame p80", "edge": "<=2px",
                    "halo": "<=8px non-edge", "dilation": "iterated 3x3-max, numpy"},
           "fit": "global affine per frame (align_mae), MAE per band",
           "systems": {}}
    res = {}
    with torch.no_grad():
        for row in real:
            n = row["name"]
            t = np.load(os.path.join(FIXDIR, row["npy"])).astype(np.float64)
            cand = sorted(glob.glob(os.path.join(FIXDIR, "inputs", n + ".*")))
            im = np.asarray(Image.open(cand[0]).convert("RGB")
                            .resize((t.shape[1], t.shape[0]))) / 255.0
            bands = edge_bands(t)
            cov = {k: float(v.mean()) for k, v in bands.items()}
            res.setdefault("_coverage", {})[n] = cov
            Hh, Ww = tuple(row["shape"])
            x = torch.tensor(T.multiscale_features(im, 33, 3, False),
                             dtype=torch.float32)
            for tag, net in mlps.items():
                p = net(x).reshape(-1).double().numpy().reshape(Hh, Ww)
                res.setdefault(tag, {})[n] = {
                    k: float(H.align_mae(p, t, bands[k])) for k in bands}
            jnet = JAMod(64, Hh // 32, Ww // 32, 4, 2, 1).float().eval()
            jnet.load_state_dict(sdj, strict=False)
            rgbt = torch.tensor(im.astype(np.float32))
            to = jnet.encode(rgbt)
            yy, xx = torch.meshgrid(torch.arange(Hh), torch.arange(Ww),
                                    indexing="ij")
            pj = jnet.head(to, rgbt, ((yy // 32) * (Ww // 32) + (xx // 32)).long()).numpy()
            res.setdefault("jattn64b2", {})[n] = {
                k: float(H.align_mae(pj, t, bands[k])) for k in bands}
    for tag in ("mscale64", "jattn64b2", "texmsc-2k", "texmsc-8kcos"):
        out["systems"][tag] = res[tag]
    out["coverage"] = res["_coverage"]
    json.dump(out, open(os.path.join(FIXDIR, "edge_analysis.json"), "w"), indent=1)
    print("wrote samples/marigold/edge_analysis.json", flush=True)
    for tag in ("mscale64", "jattn64b2", "texmsc-2k", "texmsc-8kcos"):
        m = {k: np.mean([out["systems"][tag][n][k] for n in out["systems"][tag]])
             for k in ("edge", "halo", "interior")}
        print(f"{tag}: edge={m['edge']:.4f} halo={m['halo']:.4f} "
              f"interior={m['interior']:.4f} ratio={m['edge'] / m['interior']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
