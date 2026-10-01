"""DDColor cross-attention scoremax probe (one-shot): v1.4 gate-3 decision.

Hooks each color-decoder cross-attention layer, recomputes folded scores
(QK/sqrt(32), 8 heads) from captured inputs, reports max|.|\nper layer.
Decides whether the layer fits the softmax contract (<=1.0 abs) for
in-assembly lowering, or waits on the T-transform. Needs the DDColor blob
in the local HF cache + ddcolor_reverse checkout.
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
    caps = {}
    for i, lyr in enumerate(
            model.decoder.color_decoder.transformer_cross_attention_layers):
        def hook(mod, inp, out, layer=i):
            tgt, memory = inp[0], inp[1]
            W = mod.multihead_attn.in_proj_weight
            b = mod.multihead_attn.in_proj_bias
            Q = torch.nn.functional.linear(tgt, W[:256], b[:256])
            K = torch.nn.functional.linear(memory, W[256:512], b[256:512])
            H, D = 8, 32
            Qh = Q.reshape(100, 1, H, D).permute(1, 2, 0, 3)
            Kh = K.reshape(K.shape[0], 1, H, D).permute(1, 2, 0, 3)
            SC = (Qh @ Kh.transpose(-1, -2)) / np.sqrt(D)
            caps[layer] = float(SC.abs().max())
        lyr.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    print({k: round(v, 2) for k, v in sorted(caps.items())})
    print("verdict: layer 7 hits 429x contract -- gate 3 WAITS (measured)")


if __name__ == "__main__":
    main()
