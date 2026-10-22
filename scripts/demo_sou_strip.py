"""State-of-the-union strip: teacher vs basin-C vs control, live.

Four panels per frame (frozen): (1) RGB input, (2) teacher depth via
Marigold CLI — THEIRS reference, (3) basin-C lesion-matched student
(DIRECTIONAL-MOVE, 0 bars), (4) clean-control student (BASELINE).
Every panel banners provenance + verdict status (a strip mistakable
for a quality claim fails review). Demo plumbing only — reuses
demo_dpt_webcam capture/viz patterns; no model changes (a tweak to
look better on camera is scope breach and fraud); weights read-only
out-of-tree; captured frames out-of-tree (/tmp).
Student mapping (frozen): stills resized to the 1024x672 train grid
(PIL Bilinear — the load_frame rule verbatim), then multiscale +
MLP + viz. Teacher via sibling CLI (sanitized env, marigold-env).
Usage: python3 scripts/demo_sou_strip.py [--source auto|camera|stills]
       [--frames N] [--out DIR]
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OUTDIR = "/tmp/opencode/sou_strip"
TW, TH = 1024, 672
TEACHER_SEED = "2025"

LABELS = {
    "rgb": "RGB input (not depth)",
    "teacher": "THEIRS Marigold-V2 depth = REFERENCE (not GT, not a grade)",
    "basinc": "basin-C lesion-matched = DIRECTIONAL-MOVE, 0 BARS (6 cells, 1 frame)",
    "control": "clean control = BASELINE (mscale64 recipe, 0 bars)",
}


def viz_panel(depth, lines):
    """Depth -> colorized PIL image with label lines (numpy + PIL only)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    d = np.asarray(depth, dtype=np.float64)
    lo, hi = float(np.min(d)), float(np.max(d))
    g = np.zeros_like(d) if hi <= lo else (d - lo) / (hi - lo)
    gray = Image.fromarray((g * 255 + 0.5).astype(np.uint8))
    viz = ImageOps.colorize(gray, black="navy", mid="teal", white="yellow")
    dr = ImageDraw.Draw(viz)
    y = 8
    for ln in lines:
        dr.text((8, y), ln, fill="white")
        y += 14
    return viz


def student_panel(rgb01, tag, arch_json):
    """mscale student depth on a frame (CPU, read-only weights)."""
    import numpy as np
    import torch
    from PIL import Image
    import train_student as T
    rep = json.load(open(arch_json))
    im = np.asarray(Image.fromarray(
        (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)).convert("RGB").resize(
        (TW, TH))).astype(np.float64) / 255.0
    x = T.multiscale_features(im, rep.get("field", 33),
                              rep.get("densewin", 3), False)
    W, D = rep.get("width", 64), rep.get("depth", 2)
    layers = [torch.nn.Linear(x.shape[1], W), torch.nn.SiLU()]
    for _ in range(D - 1):
        layers += [torch.nn.Linear(W, W), torch.nn.SiLU()]
    layers += [torch.nn.Linear(W, 1)]
    net = torch.nn.Sequential(*layers).float().eval()
    net.load_state_dict(torch.load(
        os.path.expanduser(f"~/student_runs/{tag}/weights.pt"),
        map_location="cpu", weights_only=True))
    with torch.no_grad():
        p = net(torch.tensor(x, dtype=torch.float32)).reshape(-1).double()
    Hh, Ww = im.shape[:2]
    return p.cpu().numpy().reshape(Hh, Ww)


def teacher_panel(paths, workdir):
    """Teacher depth via sibling CLI (1 call, all frames). Returns
    {path: depth array} at model res. Sanitized env (proven pattern)."""
    import numpy as np
    venv_py = os.path.expanduser("~/marigold-env/bin/python")
    sib = os.path.join(os.path.dirname(ROOT), "marigold-v2")
    if not (os.path.isfile(venv_py) and os.path.isdir(sib)):
        raise RuntimeError("teacher stack absent (venv/sibling)")
    import torch as _t
    if not (_t.cuda.is_available()):
        raise RuntimeError("teacher panel needs GPU")
    keep = {}
    for k in ("HOME", "USER", "LOGNAME", "TMPDIR", "CUDA_VISIBLE_DEVICES",
              "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
              "HF_HUB_OFFLINE", "http_proxy", "https_proxy", "HTTP_PROXY",
              "HTTPS_PROXY", "no_proxy"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep["PATH"] = os.path.dirname(venv_py) + ":/usr/bin:/bin"
    indir = os.path.join(workdir, "tin")
    outdir = os.path.join(workdir, "tout")
    os.makedirs(indir, exist_ok=True)
    for i, p in enumerate(paths):
        shutil.copy(p, os.path.join(indir, f"f{i:02}.png"))
    r = subprocess.run(
        [venv_py, os.path.join(sib, "scripts", "infer.py"),
         "--image_dir", indir, "--output_dir", outdir,
         "--seed", TEACHER_SEED, "--width", str(TW), "--height", str(TH)],
        capture_output=True, text=True, cwd=sib, timeout=1800, env=keep)
    assert r.returncode == 0, (r.stderr or r.stdout)[-500:]
    outs = sorted(glob.glob(os.path.join(outdir, "*", "predictions_npy",
                                         "*.npy")))
    assert len(outs) == len(paths), (len(outs), len(paths))
    return {p: np.load(o).astype(np.float64)
            for p, o in zip(paths, outs)}


def render_strip(rows):
    """rows: list of (PIL.Image, [label lines]) -> labeled strip image."""
    from PIL import Image, ImageDraw
    W, H = rows[0][0].size
    strip = Image.new("RGB", (W, len(rows) * (H + 22)), "black")
    for i, (img, lines) in enumerate(rows):
        assert img.size == (W, H), (img.size, (W, H))
        assert np.isfinite(np.asarray(img, dtype=np.float64)).all(), \
            "non-finite panel"
        strip.paste(img, (0, i * (H + 22)))
        dr = ImageDraw.Draw(strip)
        y = i * (H + 22)
        for ln in lines:
            dr.text((8, y), ln, fill="white")
            y += 11
    return strip


import numpy as np


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "camera", "stills"],
                    default="auto")
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--out", default=OUTDIR)
    args = ap.parse_args()
    import numpy as np
    from PIL import Image
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import demo_dpt_webcam as DEMO

    t0 = time.time()
    paths, origin = DEMO.capture(args.source, args.frames, args.out)
    print(f"source: {origin} ({len(paths)} frames)", flush=True)
    arch_c = os.path.join(ROOT, "samples", "marigold",
                          "student_run_lm-C.json")
    arch_b = os.path.join(ROOT, "samples", "marigold",
                          "student_run_lm-clean.json")
    walls = {}
    try:
        tdepths = teacher_panel(paths, args.out)
        print(f"teacher: {len(tdepths)} frames "
              f"({time.time() - t0:.0f}s incl. load)", flush=True)
    except RuntimeError as e:
        print(f"teacher: SKIP ({e})", flush=True)
        tdepths = {}
    rows = []
    for i, p in enumerate(paths):
        rgb = np.asarray(Image.open(p).convert("RGB"))
        rgb01 = rgb.astype(np.float64) / 255.0
        t1 = time.time()
        dc = student_panel(rgb01, "lm-C", arch_c)
        tb = student_panel(rgb01, "lm-clean", arch_b)
        walls[i] = (time.time() - t1) * 1000 / 2
        left = Image.fromarray(rgb).resize((TW, TH))
        rh, rw = rgb.shape[:2]
        pro = f"{origin} {rw}x{rh}"
        panels = [(left, [pro, LABELS["rgb"]])]
        if p in tdepths:
            panels.append((viz_panel(tdepths[p], ["THEIRS",
                                                  LABELS["teacher"]]),
                           ["THEIRS", LABELS["teacher"]]))
        panels += [(viz_panel(dc, ["basin-C", LABELS["basinc"]]),
                    ["basin-C", LABELS["basinc"]]),
                   (viz_panel(tb, ["control", LABELS["control"]]),
                    ["control", LABELS["control"]])]
        W0 = (TW * len(panels), TH + 28)
        comb = Image.new("RGB", W0, "black")
        for j, (img, lines) in enumerate(panels):
            r = img.resize((TW, TH)) if img.size != (TW, TH) else img
            comb.paste(r, (j * TW, 28))
            from PIL import ImageDraw as _ID
            _ID.Draw(comb).text((j * TW + 8, 8), lines[0], fill="white")
        rows.append((comb, [f"frame {i} {origin}"] + [ln for _, ls in panels
                                                      for ln in ls]))
    strip = render_strip(rows)
    outp = os.path.join(args.out, "sou_strip.png")
    strip.save(outp)
    ws = sorted(walls.values())
    print(f"student ms/frame: {[f'{w:.0f}' for w in walls.values()]}",
          flush=True)
    print(f"panels: {LABELS}", flush=True)
    print(f"wrote {outp} ({strip.size[0]}x{strip.size[1]})", flush=True)


if __name__ == "__main__":
    main()
