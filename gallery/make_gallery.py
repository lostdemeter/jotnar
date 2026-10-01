"""Gallery generator (v1.5 demos, committed tool): figures proving the work.

1. palette_wheel.png -- 100 DDColor queries: angle=hue, radius=causal dB
   (inverted: center=strong), size/mass, color=the hue itself (CSV data).
2. flagship/ -- showcase.py outputs (all modes + internals + sheets).
3. teal_before_after.png -- DDColor base vs coordinated 5-slot teal write
   (LIB-074 recipe), Lab-merged to RGB (exact D65 formulas, no new dep).
4. recall_curve.png -- assoc_mem recall vs noise flips (demo 2/3 machinery).
Usage: python3 gallery/make_gallery.py [--skip-showcase] (DDColor parts
need the HF blob + ddcolor checkout; flagship/recall always run)
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SIBLING = os.path.join(os.path.dirname(ROOT), "phi-core")
_phi_dir = os.environ.get("PHI_CORE_DIR") or SIBLING
sys.path.insert(0, _phi_dir)
sys.path.insert(0, ROOT)
GAL = os.path.dirname(os.path.abspath(__file__))


def lab2rgb(L, ab):
    """CIELAB (L 0..100, ab unbounded) -> sRGB 0..1, D65, exact formulas."""
    import numpy as np
    fy = (L + 16.0) / 116.0
    fx = fy + ab[0] / 500.0
    fz = fy - ab[1] / 200.0
    d = 6.0 / 29.0
    fx3 = np.where(fx > d, fx ** 3, 3 * d * d * (fx - 4.0 / 29.0))
    fy3 = np.where(fy > d, fy ** 3, 3 * d * d * (fy - 4.0 / 29.0))
    fz3 = np.where(fz > d, fz ** 3, 3 * d * d * (fz - 4.0 / 29.0))
    X, Y, Z = 0.95047 * fx3, 1.0 * fy3, 1.08883 * fz3
    R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    f = lambda t: np.where(t > 0.0031308, 1.055 * np.power(np.clip(t, 0, None), 1 / 2.4) - 0.055, 12.92 * t)
    return np.clip(np.stack([f(R), f(G), f(B)], -1), 0, 1)


def fig_wheel():
    import csv
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(ROOT, "docs", "PALETTE_CATALOG.csv"))))
    hue = np.array([float(r["hue_deg"]) for r in rows])
    c12 = np.array([float(r["causal_db_f012"]) for r in rows])
    c14 = np.array([float(r["causal_db_f014"]) for r in rows])
    caus = (c12 + c14) / 2
    th = np.radians(hue)
    r = 65 - caus  # center = strongest
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    cols = plt.cm.hsv(hue / 360.0)
    ax.scatter(th, r, s=np.clip(70 - caus, 8, 60), c=cols, alpha=0.8,
               edgecolors="k", linewidths=0.3)
    ax.set_title("DDColor palette: 100 queries by hue (angle) and causal "
                 "share (center=strongest)", pad=20)
    fig.savefig(os.path.join(GAL, "palette_wheel.png"), dpi=90)
    plt.close(fig)
    print("palette_wheel.png")


def fig_teal():
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, "/home/thorin/Documents/OpenCode/ddcolor_reverse")
    from ddcolor_ref.model import DDColor
    model = DDColor(encoder_name="convnext-t",
                    decoder_name="MultiScaleColorDecoder",
                    num_input_channels=3, input_size=(256, 256), nf=512,
                    num_output_channels=2, last_norm="Spectral",
                    do_normalize=False, num_queries=100, num_scales=3,
                    dec_layers=9)
    blob = torch.load(os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                                         "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"),
                      map_location="cpu")["params"]
    model.load_state_dict(blob, strict=False)
    model.eval()
    im = np.asarray(Image.open("/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png"
                               ).convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    x = torch.tensor(np.stack([im, im, im])[None])
    with torch.no_grad():
        base = model(x).numpy()[0]
    ro = model.refine_net[0][0]
    for Q in (87, 57, 4, 98, 28):
        w = ro.weight_orig.data[:, Q, 0, 0].clone()
        n = float(w.norm())
        ro.weight_orig.data[:, Q, 0, 0] = torch.tensor(
            [n * np.cos(np.radians(180.0)), n * np.sin(np.radians(180.0))],
            dtype=w.dtype)
    with torch.no_grad():
        got = model(x).numpy()[0]
    L = im * 100.0
    b_rgb = (lab2rgb(L, base) * 255).astype(np.uint8)
    g_rgb = (lab2rgb(L, got) * 255).astype(np.uint8)
    pair = np.concatenate([b_rgb, g_rgb], axis=1)
    Image.fromarray(pair).save(os.path.join(GAL, "teal_before_after.png"))
    print("teal_before_after.png")


def fig_recall():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    text = open(os.path.join(ROOT, "programs", "assoc_mem.asm")).read()
    sdir = os.path.join(ROOT, "programs")
    rng = np.random.default_rng(0)
    pats = rng.choice([-1.0, 1.0], size=(16, 64))

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def rate(nflip):
        keys, values = pats.T.copy(), pats.copy()
        ok = tot = 0
        r = np.random.default_rng(1000 + nflip)
        for i in range(16):
            for _ in range(4):
                cue = pats[i].copy()
                cue[r.choice(64, size=nflip, replace=False)] *= -1
                feeds = ASM.run_text(text, REGISTRY,
                                     {"cue": enc(cue.reshape(1, -1)),
                                      "keys": enc(keys),
                                      "values": enc(values)},
                                     sigs=SIGS, basedir=sdir)
                got = np.where(np.asarray(feeds["OUT"][0]).reshape(-1) > 0, 1.0, -1.0)
                ok += bool((got == pats[i]).all())
                tot += 1
        return ok / tot

    xs = [0, 4, 8, 12, 16, 20, 24]
    ys = [rate(f) for f in xs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("noise flips (of 64 bits)")
    ax.set_ylabel("recall rate")
    ax.set_title("assoc_mem recall vs noise (16 patterns, from scratch)")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(GAL, "recall_curve.png"), dpi=90)
    plt.close(fig)
    print("recall_curve.png", list(zip(xs, [round(y, 2) for y in ys])))


def main():
    import sys as _s
    os.makedirs(GAL, exist_ok=True)
    fig_wheel()
    fig_recall()
    if "--skip-showcase" not in _s.argv:
        import subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "showcase.py"),
             os.path.join(ROOT, "samples", "input_example.png"),
             os.path.join(GAL, "flagship")],
            capture_output=True, text=True, cwd=ROOT)
        print("showcase:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
    try:
        fig_teal()
    except Exception as e:
        print(f"teal figure SKIP ({e})")


if __name__ == "__main__":
    main()
