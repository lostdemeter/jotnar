"""demo: image -> enhanced image via TRUE-AMPLITUDE integer chain.

Backend: numpy integer emu (no torch needed). Float only at sensor/display
boundaries (sRGB<->linear) and offline (kernel/LUT builds).
Usage:
  python3 demo.py input.png output.png [--beta 0.5] [--blur iso|splat] [--ctrl on|off]
  python3 demo.py --selftest   # synthetic gradient, prints parity dB
Fidelity gate: int-vs-oracle >=40dB on the processed frame (proves wiring).
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chain.holo_phi import enhance_image_int, AUDIT
from chain.oracle import enhance_image_float, enhance_image_float_splat

BAR_DB = 40.0


def srgb_to_linear(im01):
    return np.power(np.clip(im01, 0, 1), 2.2).astype(np.float32)


def linear_to_srgb(lin01):
    return np.power(np.clip(lin01, 0, 1), 1.0 / 2.2)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def run(rgb01, beta, sigma, use_alpha, blur="iso", ctrl=False, depth=None,
          motion=None):
    from chain.control import load_ctrl
    rgb_lin = srgb_to_linear(rgb01).astype(np.float32)
    out_lin, yenh, info = enhance_image_int(rgb_lin, beta=beta, sigma=sigma,
                                            use_alpha=use_alpha, blur=blur,
                                            ctrl=ctrl, depth=depth,
                                            motion=motion)
    if blur in ("splat", "splat_soft"):
        P = load_ctrl()
        att = P["iso_atten"] if ctrl else 1.0
        mid = P.get("mid_atten", 0.6) if ctrl else 1.0
        stg = P.get("strong_atten", 1.0) if ctrl else 1.0
        oracle_lin, _ = enhance_image_float_splat(rgb_lin, beta=beta,
                                                  iso_atten=att,
                                                  coh_thr=P["coh_thr"],
                                                  mid_atten=mid,
                                                  coh_hi=P.get("coh_hi", 0.5),
                                                  strong_atten=stg,
                                                  soft=(ctrl in ("soft", "v5")),
                                                  soft_blur=(blur == "splat_soft"),
                                                  v5=(ctrl == "v5"),
                                                  v5_w0=P.get("v5_w0", 0.0),
                                                  v5_w1=P.get("v5_w1", 0.0),
                                                  v5_w2=P.get("v5_w2", 0.0),
                                                  depth=depth, motion=motion)
    else:
        oracle_lin, _ = enhance_image_float(rgb_lin, beta=beta, sigma=sigma,
                                            use_alpha=use_alpha)
    d = psnr(out_lin, oracle_lin)
    out_srgb = np.clip(linear_to_srgb(out_lin) * 255.0, 0, 255).astype(np.uint8)
    return out_srgb, d, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("output", nargs="?")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--alpha", action="store_true",
                    help="enable deprecated parabola ablation (default off, Debt 2)")
    ap.add_argument("--blur", default="iso", choices=["iso", "splat", "splat_soft"],
                    help="splat_soft = relu-blended bank (v4, continuous)")
    ap.add_argument("--ctrl", default="off", choices=["on", "soft", "v5", "off"],
                    help="on = v3 hard field, soft = v4 sigmoid field, "
                         "v5 = learned detail gate (needs splat_soft blur)")
    ap.add_argument("--depth", default=None, metavar="TAG",
                    help="depth prior via DAV2 (cached samples/depth/TAG.npy)")
    ap.add_argument("--motion", default=None, metavar="NPY",
                    help="forward flow (H,W,2) float pixels .npy side-channel; "
                         "live RIFE producer lands in Phase 2")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    use_alpha = bool(a.alpha)
    ctrl = {"on": True, "soft": "soft", "v5": "v5", "off": False}[a.ctrl]
    if ctrl and a.blur not in ("splat", "splat_soft"):
        ap.error("--ctrl needs --blur splat or splat_soft")
    if ctrl == "v5" and a.blur != "splat_soft":
        ap.error("--ctrl v5 needs --blur splat_soft (coh+D source)")

    if a.selftest or (not a.input):
        Hh, Ww = 128, 128
        gx, gy = np.meshgrid(np.linspace(0, 1, Ww), np.linspace(0, 1, Hh))
        rgb01 = np.stack([(gx + gy) / 2, gx, np.full_like(gx, 0.5)], -1)
        rgb01[60:68, :, :] = 0.9
        out, d, info = run(rgb01.astype(np.float32), a.beta, a.sigma, use_alpha,
                           blur=a.blur, ctrl=ctrl)
        Image.fromarray(out).save("/tmp/holo_selftest.png")
        print(f"selftest: parity {d:.2f}dB (bar {BAR_DB}) -> {'GO' if d >= BAR_DB else 'NO-GO'}")
        print(f"info={info} audit={AUDIT}")
        Image.fromarray((rgb01 * 255).astype(np.uint8)).save("/tmp/holo_selftest_in.png")
        print("wrote /tmp/holo_selftest.png + /tmp/holo_selftest_in.png")
        sys.exit(0 if d >= BAR_DB else 1)

    rgb01 = np.asarray(Image.open(a.input).convert("RGB"), dtype=np.float64) / 255.0
    depth = None
    if a.depth is not None:
        from chain.depthprior import get_depth
        depth = get_depth(rgb01, a.depth)
    motion = None
    if a.motion is not None:
        import numpy as _np
        motion = _np.load(a.motion)
    out, d, info = run(rgb01.astype(np.float32), a.beta, a.sigma, use_alpha,
                       blur=a.blur, ctrl=ctrl, depth=depth, motion=motion)
    Image.fromarray(out).save(a.output)
    print(f"enhanced {a.input} -> {a.output} beta={a.beta} sigma={a.sigma} alpha={use_alpha} blur={a.blur} ctrl={ctrl} depth={a.depth} motion={a.motion}")
    print(f"parity int-vs-oracle: {d:.2f}dB (bar {BAR_DB}) -> {'GO' if d >= BAR_DB else 'NO-GO'}")
    print(f"audit={AUDIT}")
    sys.exit(0 if d >= BAR_DB else 1)


if __name__ == "__main__":
    main()
