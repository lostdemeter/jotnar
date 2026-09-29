"""Showcase: run every enhancement state on one image, visualize internals.

Runs iso / splat / splat+ctrl (+strong beta), saves each output plus the
internal states of the flagship config (amplitude A, blurred As, detail D,
bucket map, coherence, gain), and assembles a labeled contact sheet.
Usage:
  python3 showcase.py [input.png] [outdir] [--beta 0.5]
Parity bar applies per config (proves every shown state is wired).
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
import phi_core.lattice as S
from chain.holo_phi import (enhance_image_int, enhance_luminance_int,
                            sqrt_trip, apply_gain_int)
from chain.oracle import (enhance_image_float, enhance_image_float_splat)
from chain.control import load_ctrl

BAR_DB = 40.0
FAIL = []

BUCKET_COLORS = [(220, 60, 60), (60, 120, 220), (60, 180, 80),
                 (220, 200, 60), (150, 150, 150)]  # V H D1 D2 iso


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def srgb_to_linear(im01):
    return np.power(np.clip(im01, 0, 1), 2.2).astype(np.float32)


def linear_to_srgb(lin01):
    return np.power(np.clip(lin01, 0, 1), 1.0 / 2.2)


def u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def norm01(x):
    x = x.astype(np.float64)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def lapvar(im):
    g = np.asarray(im, dtype=np.float64)
    y = 0.2126 * g[:, :, 0] + 0.7152 * g[:, :, 1] + 0.0722 * g[:, :, 2]
    return float(np.var(y[2:, 1:-1] + y[:-2, 1:-1] + y[1:-1, 2:] + y[1:-1, :-2]
                        - 4 * y[1:-1, 1:-1]))


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def states_of(rgb_lin, beta):
    """Internal states of the flagship config, via the same chain ops."""
    from chain.splat import splat_blur
    from chain.holo_phi import _load_scales
    m_acc, m_cov = _load_scales()
    y = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
         + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)
    y_t = S.encode(np.ascontiguousarray(y))
    a_t = sqrt_trip(y_t)
    A = S.decode(a_t[0], a_t[1]) * (1 - a_t[2].astype(np.float64))
    as_t, diag = splat_blur(a_t, m_acc, m_cov)
    As = S.decode(as_t[0], as_t[1]) * (1 - as_t[2].astype(np.float64))
    return {"A": A, "As": As, "D": A - As, "bucket": diag["bucket"],
            "coh": diag["coh"]}


def sheet(panels, cols, path):
    """Labeled contact sheet from [(title, PIL image)] (RGB, same height)."""
    W = max(im.width for _, im in panels)
    H = panels[0][1].height
    rows = (len(panels) + cols - 1) // cols
    canvas = Image.new("RGB", (W * cols, (H + 22) * rows), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    for i, (title, im) in enumerate(panels):
        x, y = (i % cols) * W, (i // cols) * (H + 22)
        if im.width != W or im.height != H:
            im = im.resize((W, H))
        canvas.paste(im, (x, y + 22))
        dr.text((x + 6, y + 5), title, fill=(240, 240, 240))
    canvas.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?",
                    default="/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png")
    ap.add_argument("outdir", nargs="?", default="/tmp/showcase")
    ap.add_argument("--beta", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    P = load_ctrl()

    rgb01 = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float64) / 255.0
    rgb_lin = srgb_to_linear(rgb01).astype(np.float32)
    Hh, Ww, _ = rgb_lin.shape
    print(f"input: {a.input} {Ww}x{Hh} ctrl={P}")

    configs = [("iso", {"blur": "iso", "ctrl": False}),
               ("splat", {"blur": "splat", "ctrl": False}),
               ("splat+ctrl", {"blur": "splat", "ctrl": True}),
               ("v4-soft", {"blur": "splat_soft", "ctrl": "soft"}),
               ("splat+ctrl-strong", {"blur": "splat", "ctrl": True})]
    betas = {"iso": a.beta, "splat": a.beta, "splat+ctrl": a.beta,
             "v4-soft": a.beta, "splat+ctrl-strong": 1.0}
    outs, panels, metrics = {}, [], []
    t_all = time.time()
    for name, kw in configs:
        beta = betas[name]
        t = time.time()
        out_lin, _, info = enhance_image_int(rgb_lin, beta=beta, **kw)
        dt = time.time() - t
        if kw["blur"] in ("splat", "splat_soft"):
            att = P["iso_atten"] if kw["ctrl"] else 1.0
            mid = P.get("mid_atten", 0.6) if kw["ctrl"] else 1.0
            stg = P.get("strong_atten", 1.0) if kw["ctrl"] else 1.0
            is_soft = kw["ctrl"] == "soft"
            is_ssoft = kw["blur"] == "splat_soft"
            ora, _ = enhance_image_float_splat(rgb_lin, beta=beta,
                                               iso_atten=att,
                                               coh_thr=P["coh_thr"],
                                               mid_atten=mid,
                                               coh_hi=P.get("coh_hi", 0.5),
                                               strong_atten=stg,
                                               soft=is_soft,
                                               soft_blur=is_ssoft)
        else:
            ora, _ = enhance_image_float(rgb_lin, beta=beta)
        d = psnr(out_lin, ora)
        out8 = u8(linear_to_srgb(out_lin))
        outs[name] = out8
        Image.fromarray(out8).save(os.path.join(a.outdir, f"out_{name}.png"))
        sharp = lapvar(out8)
        md = float(np.abs(out8.astype(float)
                          - u8(rgb01).astype(float)).mean())
        metrics.append((name, beta, d, sharp, md, dt, info.get("gate_frac")))
        # the 40dB bar covers the operating point (beta<=0.5); strong beta is
        # reported, not barred: bigger beta magnifies bucket-flip cost
        # linearly (measured doctrine, cf test_ctrl noise row).
        if beta <= 0.5:
            check(f"parity-{name}", d >= BAR_DB, f"{d:.1f}dB {dt:.1f}s")
        else:
            print(f"parity-{name}: measured (no bar) {d:.1f}dB {dt:.1f}s")
        panels.append((f"{name} b={beta} {d:.0f}dB", Image.fromarray(out8)))
    # modes must actually differ (not aliased flags): splat re-routes blur,
    # ctrl re-weights boost. Basis: mean abs diff well above 1 LSB.
    d_iso_splat = float(np.abs(outs["iso"].astype(float)
                               - outs["splat"].astype(float)).mean())
    d_splat_ctrl = float(np.abs(outs["splat"].astype(float)
                                - outs["splat+ctrl"].astype(float)).mean())
    check("modes-differ", d_iso_splat > 0.3 and d_splat_ctrl > 0.1,
          f"iso/splat={d_iso_splat:.2f} splat/ctrl={d_splat_ctrl:.2f} LSB")
    base_sharp = lapvar(u8(rgb01))
    print(f"\n{'mode':18s} {'beta':>5s} {'parity':>7s} {'sharp':>7s} "
          f"{'meandiff':>8s} {'sec':>5s} {'gatefrac':>8s}")
    for name, beta, d, sharp, md, dt, gf in metrics:
        gfs = f"{gf:.3f}" if gf is not None else "-"
        print(f"{name:18s} {beta:5.2f} {d:7.1f} {sharp:7.0f} "
              f"{md:8.2f} {dt:5.1f} {gfs:>8s}")
    print(f"input sharp: {base_sharp:.0f}  total {time.time() - t_all:.1f}s")

    # internals of the flagship config
    st = states_of(rgb_lin, a.beta)
    Image.fromarray(u8(norm01(st["A"]))).save(os.path.join(a.outdir, "st_amplitude.png"))
    Image.fromarray(u8(norm01(st["As"]))).save(os.path.join(a.outdir, "st_structure.png"))
    Image.fromarray(u8(norm01(np.abs(st["D"])))).save(os.path.join(a.outdir, "st_detail.png"))
    Image.fromarray(u8(np.clip(st["coh"], 0, 1))).save(os.path.join(a.outdir, "st_coherence.png"))
    bcol = np.zeros((Hh, Ww, 3), np.uint8)
    for i, c in enumerate(BUCKET_COLORS):
        bcol[st["bucket"] == i] = c
    Image.fromarray(bcol).save(os.path.join(a.outdir, "st_buckets.png"))

    states = [("input", Image.fromarray(u8(rgb01))),
              ("amplitude A=sqrt(Y)", Image.fromarray(u8(norm01(st["A"]))).convert("RGB")),
              ("structure As=splat(A)", Image.fromarray(u8(norm01(st["As"]))).convert("RGB")),
              ("detail |A-As|", Image.fromarray(u8(norm01(np.abs(st["D"])))).convert("RGB")),
              ("buckets V/H/D1/D2/iso", Image.fromarray(bcol)),
              ("coherence", Image.fromarray(u8(np.clip(st["coh"], 0, 1))).convert("RGB"))]
    sheet([panels[0]] + [(t, im) for t, im in states[1:]] + panels[1:],
          3, os.path.join(a.outdir, "sheet_states.png"))
    sheet(panels, 2, os.path.join(a.outdir, "sheet_modes.png"))
    print(f"\nwrote {a.outdir}/: out_*.png st_*.png sheet_states.png sheet_modes.png")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
