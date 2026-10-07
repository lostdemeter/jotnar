"""Memory vs highway: rank-1 MLP write vs residual add, same content.

Same France-direction, same lex gating, same prompts: one path adds
d to the residual (highway, broadcast); the other writes d (x) k^
into L27's down-projection (memory, retrieved by key match). If the
matrix write holds controls the residual add doesn't (or at better
margins), the missing component is confirmed: knowledge lives in
MLP weights, addressed by keys -- ROME-shaped, and our own
implant/bankhn history rhyming.
Usage: python3 research/mlp_vs_highway.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH, fwd_mid
from chain.qwen7b import load7b


def main():
    import torch
    _, tok = load7b()
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    tfr = [fwdH("The capital of France is"), fwdH("The capital of Germany is")]
    d = tfr[0][0][27] - tfr[1][0][27]
    d /= np.linalg.norm(d)
    tG, gids = fwdH("The capital of Germany is", keep="all")
    gtoks = tok.convert_ids_to_tokens(gids)
    gpos = next(i for i, t in enumerate(gtoks) if "ermany" in t)
    key = tG[2][gpos]
    key /= np.linalg.norm(key)
    kmid, Wd = fwd_mid("The capital of Germany is", 27)
    kh = kmid / (np.linalg.norm(kmid) ** 2)  # pseudoinverse: dW.k = d
    base = {}
    for name, pr in (("Germany", "The capital of Germany is"),
                     ("Italy", "The capital of Italy is"),
                     ("Japan", "The capital of Japan is")):
        lg, _ = fwd(pr)
        base[name] = (int(lg.argmax()), tok.decode([int(lg.argmax())]))
    print("unsteered:", {k: v[1] for k, v in base.items()}, flush=True)
    Wdt = torch.tensor(Wd, device="cuda")
    mag = float(np.linalg.norm(tfr[0][0][27]))
    for a in (0.25, 0.5, 1.0):
        import torch as _t
        # magnitude-scaled like residual steering (states ~mag units)
        dW = torch.tensor(a * mag * np.outer(d, kh), dtype=_t.float32,
                          device="cuda")
        ed = {27: {"wdown": Wdt + dW}}
        row = []
        for name, pr in (("Germany", "The capital of Germany is"),
                         ("Italy", "The capital of Italy is"),
                         ("Japan", "The capital of Japan is")):
            lg, _ = fwd(pr, edits=ed)
            top = int(lg.argmax())
            rk = int((lg > lg[paris]).sum()) + 1
            mg = float(np.sort(lg)[-1] - np.sort(lg)[-2])
            mark = "hold" if top == base[name][0] else "MOVED"
            if name == "Germany":
                mark = f"rank{rk}"
            row.append(f"{name}={tok.decode([top])[:8]}:{mark}")
        print(f"mlp-write a={a}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
