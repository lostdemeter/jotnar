"""Demo 1 (v1.5): expand DDColor's teal range by DIRECTION retune (one-shot).

Round 1 (gain x2) taught the mechanism: spectral norm RENORMALIZES
magnitude edits away (22.1dB of normalizer-lag artifact, teal DOWN) --
under spectral norm, DIRECTION is editable, magnitude is not.
Round 3 (both lessons): q27's own votes are weak on f_014, so borrow
strong-voter query rows (q41 copy) AND point refine at teal (0.2 norm).
Slot = place+color pair: loud place (borrowed), new color, preserved norm
(sigma-stable). Bands: energy >1.2x, global < 40dB. f_014.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))
IMG = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png"
QS, HUE = [87, 57, 4, 98, 28], 180.0  # top-5 teal-aligned slots (see text)


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
    im = np.asarray(Image.open(IMG).convert("L").resize((256, 256)),
                    dtype=np.float32) / 255.0
    x = torch.tensor(np.stack([im, im, im])[None])

    def teal_energy(ab):
        ux, uy = np.cos(np.radians(HUE)), np.sin(np.radians(HUE))
        return float(np.maximum(ab[0] * ux + ab[1] * uy, 0).mean())

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    with torch.no_grad():
        base = model(x).numpy()[0]
    e0 = teal_energy(base)
    ro = model.refine_net[0][0]
    for _Q in QS:
        _wrow = ro.weight_orig.data[:, _Q, 0, 0].clone()
        _n = float(_wrow.norm())
        ro.weight_orig.data[:, _Q, 0, 0] = torch.tensor(
            [_n * np.cos(np.radians(HUE)), _n * np.sin(np.radians(HUE))],
            dtype=ro.weight_orig.dtype)
    # query rows UNTOUCHED (round 5 stole the show last time): direction-only
    # edit on a strong slot -- votes stay, hue rotates, norm preserved.
    with torch.no_grad():
        got = model(x).numpy()[0]
    e1 = teal_energy(got)
    d = psnr(got, base)
    ux, uy = np.cos(np.radians(HUE)), np.sin(np.radians(HUE))
    te = lambda ab: np.maximum(ab[0] * ux + ab[1] * uy, 0)
    tb, tg = te(base), te(got)
    cut = np.percentile(tb, 90)
    new = ((tg >= cut) & (tb < cut)).mean()
    abs_churn = float(np.abs(tg - tb).mean())
    mean_shift = abs(float(tg.mean() - tb.mean()))
    print(f"redistribution: |dTeal| {abs_churn:.4f} vs |mean shift| {mean_shift:.4f} "
          f"(ratio {abs_churn/max(mean_shift,1e-9):.0f}x); top-decile churn {new:.3f}")
    np.save('/tmp/dd_demo1_base.npy', base); np.save('/tmp/dd_demo1_got.npy', got)
    print(f"teal energy {e0:.4f}->{e1:.4f} (x{e1/max(e0,1e-9):.2f}); "
          f"global {d:.1f}dB")
    print(f"DEMO1 teal-up: {'CONFIRM' if e1 > 1.2*e0 else 'FALSIFY'} "
          f"(band: >1.2x energy)")
    print(f"DEMO1 bounded: {'CONFIRM' if d < 40.0 else 'FALSIFY'} "
          f"(band: <40dB)")


if __name__ == "__main__":
    main()
