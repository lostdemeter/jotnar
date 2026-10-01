"""DDColor footprint probe (one-shot): silence a query, read where it bites.

Votes are diffuse (entropy ~0.95), so affinity maps don't segment -- but
CAUSAL footprints might: zero query q's slot (embed+feat rows), rerun,
measure global dB + per-pixel delta map. Correlate the footprint with L
(brightness) and output ab (chrominance): WHAT does the query do?
q39 (top mass) vs q0 (control). Numbers toward the ENGRAM test project.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))
IMG = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png"


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
    with torch.no_grad():
        base = model(x).numpy()[0]  # (2,256,256) ab
    print(f"base ab range [{base.min():.2f},{base.max():.2f}]")

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    for q in (39, 0):
        model.decoder.color_decoder.query_embed.weight.data[q].zero_()
        model.decoder.color_decoder.query_feat.weight.data[q].zero_()
        with torch.no_grad():
            got = model(x).numpy()[0]
        # restore (in-place zeroing would corrupt later probes)
        model.decoder.color_decoder.query_embed.weight.data[q].copy_(
            blob["decoder.color_decoder.query_embed.weight"][q])
        model.decoder.color_decoder.query_feat.weight.data[q].copy_(
            blob["decoder.color_decoder.query_feat.weight"][q])
        d = psnr(got, base)
        foot = np.abs(got - base).mean(0)
        foot /= foot.max() + 1e-30
        print(f"silence-q{q}: {d:.1f}dB; footprint-vs-L corr "
              f"{np.corrcoef(foot.ravel(), im.ravel())[0,1]:+.2f}; "
              f"footprint mass top-half-L pixels: "
              f"{foot[im > np.median(im)].mean():.3f} vs bottom "
              f"{foot[im <= np.median(im)].mean():.3f}")
        np.save(f"/tmp/dd_foot_q{q}.npy", foot)


if __name__ == "__main__":
    main()
