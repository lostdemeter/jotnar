"""Live DPT-tiny webcam demo (pipeline, UNTRAINED weights, measured cadence).

Served CUDA step wall on 672x1024 DPT-tiny: ~1192 ms/frame end-to-end
(measured 5x, median) -> frame-strip timelapse (not a live loop;
0.8 fps is a slideshow, stated not sold). ffmpeg snapshots
(cv2 absent, no new deps), served DPT-tiny step per frame,
side-by-side RGB + depth-viz strip. Every artifact banners
UNTRAINED weights / pipeline demo (honesty is load-bearing).
Camera busy (desktop holds /dev/video0) -> dav2_reverse/captures
stills fallback, stated in output; never signals user processes.
Captured frames stay out-of-tree (/tmp/opencode).
Usage: python3 scripts/demo_dpt_webcam.py [--source auto|camera|stills]
       [--frames N] [--interval S] [--out DIR] [--exe PATH]
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
sys.path.insert(0, os.path.join(ROOT, "programs"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

OUTDIR = "/tmp/opencode/dptstrip"
STILLS = "/home/balrog/Documents/OpenCode/dav2_revisited/dav2_reverse/captures"
BANNER = "UNTRAINED weights (random kaiming) — PIPELINE DEMO, not depth"
TW, TH = 1024, 672


def grab_camera(dst):
    """One ffmpeg snapshot; returns path or None (busy/absent -> None)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-f", "v4l2", "-video_size", "1280x720", "-i", "/dev/video0",
             "-frames:v", "1", "-y", dst],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return dst if (r.returncode == 0 and os.path.isfile(dst)) else None


def fallback_stills(n):
    """dav2_reverse captures (read-only use); repeats deterministically."""
    pool = sorted(glob.glob(os.path.join(STILLS, "*.png")))
    assert pool, "no fallback stills"
    return [pool[i % len(pool)] for i in range(n)]


def capture(source, nframes, outdir):
    """Returns (paths, origin_tag). Camera tried once in auto mode."""
    os.makedirs(outdir, exist_ok=True)
    if source in ("auto", "camera"):
        ok, paths = True, []
        for i in range(nframes):
            p = os.path.join(outdir, f"cam{i:02}.png")
            if grab_camera(p) is None:
                ok = False
                break
            paths.append(p)
        if ok:
            return paths, "camera"
        if source == "camera":
            raise RuntimeError("camera held/absent and --source camera forced")
    paths = []
    for i, s in enumerate(fallback_stills(nframes)):
        d = os.path.join(outdir, f"still{i:02}.png")
        shutil.copy(s, d)
        paths.append(d)
    return paths, ("stills-forced" if source == "stills"
                   else "stills-fallback(camera-held)")


def prepare_inputs(prog_text, Wd, ids, rgb01):
    """Sample dict for compile + feed dict for runs (float32 / int64)."""
    import numpy as np
    import dpt_tiny as DT
    Hh, Ww, _ = rgb01.shape
    sample = {"RGB": np.ascontiguousarray(
        rgb01.reshape(-1, 3).astype(np.float32))}
    for ti in range(256):
        sample[f"ptap{ti}"] = np.ascontiguousarray(ids["patch"][ti])
    for nm in ("pos", "lnw", "lnb", "pemb"):
        sample[nm] = np.ascontiguousarray(Wd[nm].astype(np.float32))
    for bi in range(6):
        for nm in ("wq", "wk", "wv", "wo", "w1", "w2"):
            sample[f"b{bi}{nm}"] = np.ascontiguousarray(
                Wd[f"b{bi}{nm}"].astype(np.float32))
    for nm in ("r32", "r16", "r8", "r4", "hp", "hl"):
        sample[nm] = np.ascontiguousarray(Wd[nm].astype(np.float32))
    for li in range(4):
        for ri in range(2):
            for ci in range(2):
                for ti in range(9):
                    sample[f"f{li}r{ri}c{ci}t{ti}"] = np.ascontiguousarray(
                        Wd[f"f{li}r{ri}c{ci}t{ti}"].astype(np.float32))
    for nm in ("up16_8", "up16_4", "dn16_32", "up8_4", "up4_full", "up32_16"):
        sample[nm] = np.ascontiguousarray(ids[nm])
    for nm, (gh, gw) in (("conv32", (21, 32)), ("conv16", (42, 64)),
                         ("conv8", (84, 128)), ("conv4", (168, 256))):
        for ti in range(9):
            sample[f"{nm}t{ti}"] = np.ascontiguousarray(
                DT._ids_conv_taps(gh, gw)[ti])
    return sample


def compile_served(workdir="/tmp/dpttiny_cu"):
    import dpt_tiny as DT
    import test_dpttiny as T
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    prog, ids = DT.build_dpttiny()
    text = prog.text()
    Wd = T.gen_weights()
    sample = prepare_inputs(text, Wd, ids, __import__("numpy").zeros((TH, TW, 3)))
    art = compile_program(text, "cuda", sample=sample, outputs=["OUT"],
                          basedir=os.path.join(ROOT, "programs"))
    exe = build_cu(art["source"], workdir, name="dpttiny",
                   units=art.get("units"))
    in_order = [ln.split()[1] for ln in text.splitlines()
                if ln.startswith("IN ")]
    return exe, in_order, Wd, ids


def write_static_inputs(workdir, in_order, sample):
    """Weights/ids written once (per-frame only RGB moves)."""
    import numpy as np
    for nm in in_order:
        if nm == "RGB":
            continue
        arr = sample[nm]
        (arr.astype(np.float32) if arr.dtype != np.int64
         else arr).tofile(os.path.join(workdir, f"f_{nm}.bin"))


def served_step(exe, in_order, workdir, rgb01, Wd, ids):
    """One served step -> (672,1024) float32 depth. No training anywhere."""
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.fromarray(
        (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)).convert("RGB").resize(
        (TW, TH))).astype(np.float32) / 255.0
    im.reshape(-1, 3).astype(np.float32).tofile(
        os.path.join(workdir, "f_RGB.bin"))
    argv = [exe] + [os.path.join(workdir, f"f_{nm}.bin")
                    for nm in in_order]
    argv.append(os.path.join(workdir, "o_OUT.bin"))
    r = subprocess.run(argv, capture_output=True, text=True, cwd=workdir)
    assert r.returncode == 0, r.stderr[:300]
    return np.fromfile(os.path.join(workdir, "o_OUT.bin"),
                       dtype=np.float32).reshape(TH, TW)


def visualize(depth):
    """Depth -> colorized PIL image (numpy + PIL only) + banner."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps
    d = np.asarray(depth, dtype=np.float64)
    lo, hi = float(np.min(d)), float(np.max(d))
    g = np.zeros_like(d) if hi <= lo else (d - lo) / (hi - lo)
    gray = Image.fromarray((g * 255 + 0.5).astype(np.uint8))
    viz = ImageOps.colorize(gray, black="navy", mid="teal", white="yellow")
    dr = ImageDraw.Draw(viz)
    dr.text((8, 8), BANNER, fill="white")
    return viz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "camera", "stills"],
                    default="auto")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--out", default=OUTDIR)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--workdir", default="/tmp/dpttiny_cu")
    args = ap.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw

    t0 = time.time()
    paths, origin = capture(args.source, args.frames, args.out)
    print(f"source: {origin} ({len(paths)} frames)", flush=True)
    if args.exe:
        exe = args.exe
        import dpt_tiny as DT
        import test_dpttiny as T
        prog, ids = DT.build_dpttiny()
        in_order = [ln.split()[1] for ln in prog.text().splitlines()
                    if ln.startswith("IN ")]
        Wd = T.gen_weights()
    else:
        exe, in_order, Wd, ids = compile_served(args.workdir)
    print(f"served binary ready ({time.time() - t0:.0f}s incl. build)",
          flush=True)
    write_static_inputs(
        args.workdir, in_order,
        prepare_inputs(None, Wd, ids, np.zeros((TH, TW, 3))))
    strips, walls = [], []
    for i, p in enumerate(paths):
        if i and args.source != "stills":
            time.sleep(max(0.0, args.interval))
        rgb = np.asarray(Image.open(p).convert("RGB"))
        t1 = time.time()
        depth = served_step(exe, in_order, args.workdir,
                            rgb.astype(np.float64) / 255.0, Wd, ids)
        walls.append((time.time() - t1) * 1000)
        viz = visualize(depth)
        left = Image.fromarray(rgb).resize((TW, TH))
        comb = Image.new("RGB", (2 * TW, TH + 28), "black")
        comb.paste(left, (0, 28))
        comb.paste(viz.resize((TW, TH)), (TW, 28))
        ImageDraw.Draw(comb).text(
            (8, 8), f"{BANNER} | {origin} | frame {i}", fill="white")
        strips.append(comb)
    strip = Image.new("RGB", (2 * TW, len(strips) * (TH + 28)))
    for i, s in enumerate(strips):
        strip.paste(s, (0, i * (TH + 28)))
    outp = os.path.join(args.out, "dptstrip.png")
    strip.save(outp)
    walls_sorted = sorted(walls)
    print(f"step wall ms: {[f'{w:.0f}' for w in walls]} "
          f"median {walls_sorted[len(walls_sorted) // 2]:.0f}", flush=True)
    print(f"wrote {outp}", flush=True)
    print(f"{BANNER} | origin={origin}", flush=True)


if __name__ == "__main__":
    main()
