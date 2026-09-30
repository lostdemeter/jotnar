"""Generative gates (stage 2): a micro-AI written blind in listings, then run.

programs/stabilize_mgd.asm was designed on paper BEFORE running. The paper
claim said "denoiser"; MEASUREMENT falsified it (flat-region out 25.1dB vs
in 26.3dB): A + beta*D keeps the noisy base, so NO boost architecture can
denoise -- output-domain averaging or As-output would be needed (backlog
structures, explicitly not claimed). What survived, gated here: a temporal
STABILIZER -- memory steadies the boost on static content while moving
pixels trust the current frame bit-clean. Renamed, not re-tuned:
falsification at full resolution is the paper-first discipline working.
  1. flicker-wins: static noisy run, listing frame-diff < 0.85x still-iso
  2. no-smear: moving sharpness within 10% of the still path
  3. listing-vs-oracle parity >=40dB (MATCHED quantized inputs both sides)
  4. STATIC mask exact on fixtures; frame 0 == still-iso bit-exactly
If any prediction fails, the DESIGN is wrong (not the gates) -- report, don't
tune the fixture. Usage: python3 test_stabilize.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS, SIGS
from chain import oracle as O
from chain.holo_phi import enhance_image_int

FAIL = []
N = 48


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=255.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def lapvar(im):
    return float(np.var(im[2:, 1:-1] + im[:-2, 1:-1] + im[1:-1, 2:] + im[1:-1, :-2] - 4 * im[1:-1, 1:-1]))


def load_prog():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "programs", "stabilize_mgd.asm")) as fh:
        return fh.read()


def run_seq(text, frames8, flows):
    """frames8: uint8 sRGB frames (quantized ONCE upstream -- listing and
    oracle both consume the same bytes; input mismatch is a test bug, not
    a result). Threads feeds['D'] as next dprev."""
    outs, dprev = [], None
    for rgb, fl in zip(frames8, flows):
        feeds = ASM.run_text(text, REGISTRY,
                             {"rgb": rgb, "dprev": dprev, "flow": fl},
                             sigs=SIGS)
        outs.append(feeds["OUT"])
        dprev = feeds["D"]
    return outs


def main():
    text = load_prog()
    rng = np.random.default_rng(23)
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                         indexing="ij")
    bar = np.where(xx > 0.5, 0.85, 0.15)
    # fixtures quantized ONCE (like real image files); everything downstream
    # consumes the same bytes.
    frames = [(np.clip(bar + rng.normal(0, 0.05, bar.shape), 0, 1) * 255
               ).astype(np.uint8) for _ in range(4)]
    frames = [np.stack([f] * 3, -1) for f in frames]
    flows = [np.zeros((N, N, 2))] * 4

    # 1. flicker-wins on the static noisy run (the stabilizer's true claim).
    #    FALSIFICATION RECORD: the paper design claimed denoising (output
    #    closer to clean than input); measured flat-region out 25.1dB vs in
    #    26.3dB -- A + beta*D keeps the noisy base, so NO boost architecture
    #    can denoise (output-domain averaging or As-output would be needed;
    #    both are backlog structures, explicitly not claimed). Renamed
    #    denoise->stabilize rather than tuning the fixture.
    outs = run_seq(text, frames, flows)
    fl = np.mean([np.abs(outs[i].astype(float) - outs[i - 1].astype(float)).mean()
                  for i in range(1, 4)])
    souts = []
    for rgb in frames:
        lin = np.power(rgb.astype(np.float64) / 255.0, 2.2).astype(np.float32)
        r, _, _ = enhance_image_int(lin, beta=0.5, blur="iso")
        souts.append(np.clip(np.power(np.clip(r, 0, 1), 1.0 / 2.2) * 255.0,
                             0, 255).astype(np.uint8))
    fls = np.mean([np.abs(souts[i].astype(float) - souts[i - 1].astype(float)).mean()
                   for i in range(1, 4)])
    check("flicker-wins", fl < 0.85 * fls, f"listing={fl:.2f} still={fls:.2f}")

    # 2. moving noisy run: sharpness within 10% of the still path.
    mframes, mflows = [], []
    for t in range(4):
        b = np.where(xx > 0.5 - 3 * t / N, 0.85, 0.15)
        mframes.append((np.stack([(np.clip(
            b + rng.normal(0, 0.05, b.shape), 0, 1)) * 255] * 3, -1)
            ).astype(np.uint8))
        mflows.append(np.full((N, N, 2), [0.0, 3.0]))
    mouts = run_seq(text, mframes, mflows)
    lin0 = np.power(mframes[-1].astype(np.float64) / 255.0, 2.2).astype(np.float32)
    ref, _, _ = enhance_image_int(lin0, beta=0.5, blur="iso")
    ref8 = np.clip(np.power(np.clip(ref, 0, 1), 1.0 / 2.2) * 255.0,
                   0, 255).astype(np.uint8)
    check("no-smear", lapvar(mouts[-1][:, :, 0].astype(float)) >= 0.9 * lapvar(ref8[:, :, 0].astype(float)),
          "moving sharpness within 10% of still path")

    # 3. listing-vs-oracle parity, LINEAR basis (peak 1). Doctrine: parity
    #    measures the MACHINE at its boundary (linear light, where the chain
    #    ends -- like every other parity gate in the repo); sRGB is for
    #    PERCEPTUAL claims (flicker rows above). Comparing sRGB bytes holds
    #    the chain accountable for gamma's shadow expansion (a 0.002 linear
    #    error reads 25 LSB in deep shadow) -- caught during development,
    #    basis corrected, not the bar lowered.
    import phi_core.lattice as S
    ys = [np.power(f[:, :, 0].astype(float) / 255.0, 2.2) for f in frames]
    ds = []
    dprev_o = None
    dprev = None
    for i, (rgb, y, fli) in enumerate(zip(frames, ys, flows)):
        feeds = ASM.run_text(text, REGISTRY,
                             {"rgb": rgb, "dprev": dprev, "flow": fli},
                             sigs=SIGS)
        dprev = feeds["D"]
        Y = feeds["YENH"]
        yv = np.clip(S.decode(Y[0], Y[1]) * (1 - Y[2].astype(np.float64)), 0, 1)
        yo, dprev_o = O.denoise_frame_float(y, dprev_o, fli, beta=0.5)
        ds.append(psnr(yv, np.clip(yo, 0, 1), peak=1.0))
    check("parity-oracle", all(v >= 40.0 for v in ds),
          f"{[f'{v:.1f}' for v in ds]}dB (linear, peak 1)")

    # 4. STATIC mask exact; frame 0 == still-iso bit-exactly.
    zf = np.zeros((N, N, 2))
    uf = np.full((N, N, 2), [0.0, 3.0])
    feeds_z = ASM.run_text(text, REGISTRY,
                           {"rgb": frames[0], "dprev": None, "flow": zf},
                           sigs=SIGS)
    feeds_u = ASM.run_text(text, REGISTRY,
                           {"rgb": frames[0], "dprev": None, "flow": uf},
                           sigs=SIGS)
    check("static-exact", bool(feeds_z["S"].all()) and not bool(feeds_u["S"].any()),
          "zero flow -> all-static; uniform flow -> none-static")
    lin0s = np.power(frames[0].astype(np.float64) / 255.0, 2.2).astype(np.float32)
    ref0, _, _ = enhance_image_int(lin0s, beta=0.5, blur="iso")
    ref08 = np.clip(np.power(np.clip(ref0, 0, 1), 1.0 / 2.2) * 255.0,
                    0, 255).astype(np.uint8)
    check("frame0-still", bool((feeds_z["OUT"] == ref08).all()),
          "frame 0 == still-iso path (dprev=None -> direct)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
