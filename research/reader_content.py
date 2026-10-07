"""Reader-vs-content: does norm-biased argmax obscure good steering?

Takes steered runs (L27 contrast) and re-reads the SAME final hidden
two ways: argmax (norm-biased: frequent rows win by magnitude) vs
cosine (norm-free direction matching). If cosine ranks Paris first
while argmax picks blanks, the siphon content was correct and the
READER (Qwen's norm map) is the obscurer -- same readout physics as
the dead-space result, one level up.
Second leg: key-gated steering (scale the implant by context match):
does conditional application restore specificity?
Usage: python3 research/reader_content.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwd_full, fwdH
from chain.qwen7b import load7b


def main():
    _, tok = load7b()
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    tfr = [fwdH("The capital of France is"), fwdH("The capital of Germany is")]
    d = tfr[0][0][27] - tfr[1][0][27]
    d /= np.linalg.norm(d)
    lg, h, _ = fwd_full("The capital of Germany is", steer=(27, d, 0.8))
    o = np.argsort(-lg)
    print("argmax top5:", [tok.decode([int(i)]) for i in o[:5]], flush=True)
    print("paris rank (argmax):", int((lg > lg[paris]).sum()) + 1, flush=True)
    import torch
    from chain.qwen7b import load7b as _lb
    g, _ = _lb()
    W = g("lm_head.weight").astype(np.float64)
    hn = h / np.linalg.norm(h)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos = Wn @ hn
    oc = np.argsort(-cos)
    print("cosine top5:", [tok.decode([int(i)]) for i in oc[:5]], flush=True)
    print("paris rank (cosine):", int((cos > cos[paris]).sum()) + 1, flush=True)

    # leg 2: key-gated steering (conditional application).
    # Gate from layer-26 state BEFORE implanting: match to the Germany
    # key, sharpened. Same steering scaled by the gate: full dose where
    # the context matches, ~nothing elsewhere.
    tG = fwdH("The capital of Germany is")[0]
    k = tG[26] / np.linalg.norm(tG[26])
    print("leg2: gate = cos(H26, germany-key)^4", flush=True)
    for tgt, tname in (("The capital of Germany is", "Germany"),
                       ("The capital of Italy is", "Italy"),
                       ("The capital of Japan is", "Japan")):
        import torch as _t
        from qwen_torch import _layer, _get, _rms
        torch, g2, tok2 = _get()
        dt = torch.float32
        ids = tok2(tgt, return_tensors="pt")["input_ids"][0].numpy()
        n = len(ids)
        E = torch.tensor(g2("model.embed_tokens.weight")[ids], dtype=dt,
                         device="cuda")
        x = E
        for L in range(28):
            if L == 27:
                # gate from layer-26 state (pre-implant): match to key
                with torch.no_grad():
                    h26 = x[n - 1].detach().cpu().numpy()
                    m = float(h26 @ k / (np.linalg.norm(h26) + 1e-12))
                    s = m ** 4 if m > 0 else 0.0
                x = _layer(x, g2, torch, L, n, steer=(27, d, 0.8 * 4 * s))
            else:
                x = _layer(x, g2, torch, L, n)
        lnf = torch.tensor(g2("model.norm.weight"), dtype=dt, device="cuda")
        hn = _rms(x, lnf)
        lg2 = (hn @ torch.tensor(g2("lm_head.weight"), dtype=dt,
                                device="cuda").T)[n - 1].detach().cpu().numpy()
        print(f"gated {tname}: top={tok2.decode([int(lg2.argmax())])!r} "
              f"paris-rank={int((lg2 > lg2[paris]).sum()) + 1} "
              f"gate={s:.3f}", flush=True)


if __name__ == "__main__":
    main()
