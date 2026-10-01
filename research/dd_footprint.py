"""DDColor footprint probe (one-shot): silence a query, read where it bites.

Votes are diffuse (entropy ~0.95), so affinity maps don't segment -- but
CAUSAL footprints might: zero query q's slot (embed+feat rows), rerun,
measure global dB + per-pixel delta map. Correlate the footprint with L
(brightness) and output ab (chrominance): WHAT does the query do?
q39 (top mass) vs q0 (control). Numbers toward the ENGRAM test project.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))
IMG = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png"
IMGS = [
    "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png",
    "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png",
]
# Bright-owner hunt (round 3): top-mass queries on the bright deciles.
# Predict (paper-first): >=1 of the four shows mirror-image footprint
# (bright-half mass > 2x dark-half). q0/q39 rows above are the record.
QUERIES = (41, 94, 24, 64)
HUNT_IMGS = [
    "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png",
]


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

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    # q39 loop (label candidate: dark-region colorizer from f_014): on a
    # FRESH image predict dark-bite (footprint-vs-L corr <= -0.5) with
    # substantial move (global < 40dB). q0 measured both rounds (no band).
    import sys as _sys
    _hunt = _sys.argv[1] == "hunt" if len(_sys.argv) > 1 else False
    _paths = HUNT_IMGS if _hunt else IMGS
    _queries = QUERIES if _hunt else (39, 0)
    for path in _paths:
        tag = os.path.basename(path)
        im = np.asarray(Image.open(path).convert("L").resize((256, 256)),
                        dtype=np.float32) / 255.0
        x = torch.tensor(np.stack([im, im, im])[None])
        with torch.no_grad():
            base = model(x).numpy()[0]  # (2,256,256) ab
        print(f"== {tag} base ab [{base.min():.2f},{base.max():.2f}]")
        for q in _queries:
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
            cc = float(np.corrcoef(foot.ravel(), im.ravel())[0, 1])
            print(f"silence-q{q}: {d:.1f}dB; footprint-vs-L corr {cc:+.2f}; "
                  f"darkmass {foot[im <= np.median(im)].mean():.3f} vs "
                  f"bright {foot[im > np.median(im)].mean():.3f}")
            np.save(f"/tmp/dd_foot_{tag}_q{q}.npy", foot)


if __name__ == "__main__":
    main()
