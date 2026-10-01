"""Palette catalog (one-shot): all 100 queries, both images.

Per query: hue angle + refine norm (zero runs, refine rows), vote mass
(saved maps), causal dB (silence slot, 100 forwards x 2 images @0.2s).
Writes docs/PALETTE_CATALOG.csv. Numbers toward reliable ENGRAM
modification (add/remove with predicted cost needs the full table).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))
IMGS = {
    "f_012": "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png",
    "f_014": "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png",
}


def main():
    import csv
    import time
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, DD)
    from ddcolor_ref.model import DDColor

    t0 = time.time()
    model = DDColor(encoder_name="convnext-t",
                    decoder_name="MultiScaleColorDecoder",
                    num_input_channels=3, input_size=(256, 256), nf=512,
                    num_output_channels=2, last_norm="Spectral",
                    do_normalize=False, num_queries=100, num_scales=3,
                    dec_layers=9)
    blob = torch.load(PTH, map_location="cpu")["params"]
    model.load_state_dict(blob, strict=False)
    model.eval()
    W = blob["refine_net.0.0.weight_orig"].float().numpy()[:, :, 0, 0].T
    Q = W[:100]
    hue = np.degrees(np.arctan2(Q[:, 1], Q[:, 0])) % 360
    rnorm = np.linalg.norm(Q, axis=1)
    xs = {}
    for tag, p in IMGS.items():
        im = np.asarray(Image.open(p).convert("L").resize((256, 256)),
                        dtype=np.float32) / 255.0
        xs[tag] = torch.tensor(np.stack([im, im, im])[None])

    def psnr(a, b):
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(4.0 / mse)

    mass = {t: np.load(f"/tmp/dd_votes_{t}.png.npy").sum((1, 2))
            for t in IMGS}
    rows = []
    with torch.no_grad():
        bases = {t: model(x).numpy()[0] for t, x in xs.items()}
        for q in range(100):
            model.decoder.color_decoder.query_embed.weight.data[q].zero_()
            model.decoder.color_decoder.query_feat.weight.data[q].zero_()
            ds = {}
            for t, x in xs.items():
                ds[t] = psnr(model(x).numpy()[0], bases[t])
            model.decoder.color_decoder.query_embed.weight.data[q].copy_(
                blob["decoder.color_decoder.query_embed.weight"][q])
            model.decoder.color_decoder.query_feat.weight.data[q].copy_(
                blob["decoder.color_decoder.query_feat.weight"][q])
            rows.append((q, hue[q], rnorm[q], mass["f_012"][q],
                         mass["f_014"][q], ds["f_012"], ds["f_014"]))
            if (q + 1) % 25 == 0:
                print(f"  {q+1}/100 ({time.time()-t0:.0f}s)", flush=True)
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "docs", "PALETTE_CATALOG.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["q", "hue_deg", "refine_norm", "mass_f012", "mass_f014",
                    "causal_db_f012", "causal_db_f014"])
        for r in rows:
            w.writerow([r[0]] + [round(float(v), 2) for v in r[1:]])
    print(f"done in {time.time()-t0:.0f}s -> docs/PALETTE_CATALOG.csv")


if __name__ == "__main__":
    main()
