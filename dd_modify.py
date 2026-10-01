"""Palette ADD demo (one-shot): write a missing hue at a dead slot.

New hue 135deg (sparse bin: 120/150 hold 4 each) written into q56's slot:
refine row -> 0.2*(cos135, sin135) (weight_orig; hooks recompute), query
rows -> copy of q41's (strong voter, borrows its spatial pattern; the new
CONTENT is the hue). Predict (paper-first, honest +/-10dB yardstick --
norm-fit is triage-grade, NOT preview-grade like the SVD law): (a) output
hue mass at 135+-15deg increases >3x; (b) global disturbance < 35dB.
f_014 only. Numbers toward reliable add/remove (measured costs + margins,
not predicted exactness).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))
IMG = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png"
HUE, NORM = 135.0, 0.2


def main():
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, DD)
    from ddcolor_ref.model import DDColor

    model = DDColor(encoder_name="convnext-t",
                    decoder_name="MultiScaleColorDecoder",
                    num_input_channels=3, input_size=(256, 256), nf=512,
                    num_output_channels=2, last_norm="Spectral",
                    do_normalize=False, num_queries=100, num_scales=3,
                    dec_layers=9)
    blob = torch.load(PTH, map_location="cpu")["params"]
    model.load_state_dict(blob, strict=False)
    model.eval()
    votes = {}
    model.decoder.color_decoder.register_forward_hook(
        lambda mod, inp, out: votes.__setitem__("out", out.detach()))
    im = np.asarray(Image.open(IMG).convert("L").resize((256, 256)),
                    dtype=np.float32) / 255.0
    x = torch.tensor(np.stack([im, im, im])[None])

    def hue_mass(ab):
        ang = np.degrees(np.arctan2(ab[1], ab[0])) % 360
        band = np.abs((ang - HUE + 180) % 360 - 180) <= 15
        return float(band.mean()), float(np.abs(ab).mean())

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    with torch.no_grad():
        base = model(x).numpy()[0]
    m0, _ = hue_mass(base)
    v0 = float(votes["out"].numpy()[0, 56].sum())
    print(f"base: hue135 mass {m0:.4f}; q56 vote mass {v0:.1f}")
    # write: refine row -> new hue; query rows -> strong voter copy
    import math
    ro = model.refine_net[0][0]
    ro.weight_orig.data[:, 56, 0, 0] = torch.tensor(
        [NORM * math.cos(math.radians(HUE)),
         NORM * math.sin(math.radians(HUE))])
    qe = model.decoder.color_decoder.query_embed.weight.data
    qf = model.decoder.color_decoder.query_feat.weight.data
    qe[56] = qe[41].clone()
    qf[56] = qf[41].clone()
    with torch.no_grad():
        got = model(x).numpy()[0]
    m1, _ = hue_mass(got)
    v1 = float(votes["out"].numpy()[0, 56].sum())
    d = psnr(got, base)
    print(f"modified: hue135 mass {m1:.4f} (x{m1/max(m0,1e-9):.1f}); "
          f"q56 vote mass {v0:.1f}->{v1:.1f}; global {d:.1f}dB")
    # landing check: silence the WRITTEN slot -- if the write lives
    # there, removal moves output substantially (predicted < 40dB given
    # the strong-voter query rows)
    qe[56].zero_()
    qf[56].zero_()
    with torch.no_grad():
        noslot = model(x).numpy()[0]
    d2 = psnr(noslot, got)
    print(f"post-write silence q56: {d2:.1f}dB "
          f"({'CONFIRM slot live' if d2 < 40.0 else 'FALSIFY slot dead'})")
    print(f"ADD hue-appears: {'CONFIRM' if m1 > 3 * m0 else 'FALSIFY'} "
          f"(band: >3x)")
    print(f"ADD bounded: {'CONFIRM' if d < 35.0 else 'FALSIFY'} "
          f"(band: <35dB)")


if __name__ == "__main__":
    main()
