"""DDColor query miner (one-shot, not a gate): spatial fingerprints.

Loads the ORIGINAL paper_tiny (untouched 211MB .pth -- never our V20
port, which deleted the transformer): convnext-t encoder + 9-layer
color decoder with 100 queries. Captures the decoder's per-query vote
maps (1,100,H,W) on grayscale images: each query's spatial fingerprint.
Questions: do queries specialize (focused maps)? Which dominate (mass)?
Same query across images -- stability? Numbers feed the ENGRAM test
project (object slots as label candidates).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DD = "/home/thorin/Documents/OpenCode/ddcolor_reverse"
PTH = (os.path.expanduser("~/.cache/huggingface/hub/models--piddnad--"
                          "DDColor-models/blobs/81dd643904f4664c3718513e3320ae3db0c567f0d3e18398606659adee7bfc17"))


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
    missing, unexpected = model.load_state_dict(blob, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
    for k in list(missing)[:5]:
        print("  MISSING:", k)
    model.eval()
    votes = {}

    def hook(mod, inp, out):
        votes["out"] = out.detach()

    model.decoder.color_decoder.register_forward_hook(hook)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    paths = [
        os.path.join(root, "samples", "input_example.png"),
        "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png",
        "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_014.png",
    ]
    with torch.no_grad():
        for p in paths:
            im = np.asarray(Image.open(p).convert("L").resize((256, 256)),
                            dtype=np.float32) / 255.0
            x = torch.tensor(np.stack([im, im, im])[None])
            model(x)
            v = votes["out"].numpy()[0]  # (100,H,W)
            print(f"{os.path.basename(p)}: votes {v.shape} "
                  f"mass top-5 queries {np.argsort(v.sum((1, 2)))[-5:][::-1]}")
            np.save(f"/tmp/dd_votes_{os.path.basename(p)}.npy", v)
    print("saved /tmp/dd_votes_*.npy")


if __name__ == "__main__":
    main()
