"""Order commutativity: sequential persistent installs, both orders + joint.

Two rank-1 MLP writes (L27 down-proj, ROME-form) installing Paris-on-
Germany and Berlin-on-France. Orders: A-then-B (B's key re-measured in
the A-written model), B-then-A (mirror), joint (both keys from base).
Bare matrix addition commutes trivially -- so any behavioral difference
isolates the true order channel: key drift under prior writes (plus
saturation/gating shifts). Null (identical) says install order is free
and catalog turns need no scheduler.
Usage: python3 research/order_commute.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH, fwd_mid
from chain.qwen7b import load7b

ALPHA = 0.5  # installing dose: single writes move ranks here
LAYER = 27


def main():
    import torch
    _, tok = load7b()
    paris = tok(" Paris", return_tensors="pt")["input_ids"][0].tolist()[0]
    berlin = tok(" Berlin", return_tensors="pt")["input_ids"][0].tolist()[0]
    tF, _ = fwdH("The capital of France is")
    tG, _ = fwdH("The capital of Germany is")
    d_FG = tF[0][LAYER] - tG[0][LAYER]
    d_FG /= np.linalg.norm(d_FG)
    d_GF = -d_FG
    kG0, Wd = fwd_mid("The capital of Germany is", LAYER)
    kF0, _ = fwd_mid("The capital of France is", LAYER)
    kh_G = kG0 / (np.linalg.norm(kG0) ** 2)
    kh_F = kF0 / (np.linalg.norm(kF0) ** 2)
    mag = float(np.linalg.norm(tG[0][LAYER]))
    Wdt = torch.tensor(Wd, device="cuda")

    def write(W, d, kh, a):
        import torch as _t
        return W + torch.tensor(a * mag * np.outer(d, kh),
                               dtype=_t.float32, device="cuda")

    def mid_in_model(prompt, W):
        import torch as _t  # noqa
        from qwen_torch import _layer, _get, _rms  # noqa
        torch, g, tok = _get()
        dt = torch.float32
        ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
        n = len(ids)
        E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                         device="cuda")
        x = E
        cap = {}
        for L in range(LAYER + 1):
            x = _layer(x, g, torch, L, n,
                       edits={LAYER: {"wdown": W}} if L == LAYER else None,
                       capture=cap if L == LAYER else None)
        return cap["mid"][n - 1].detach().cpu().numpy()

    W_A = write(Wdt, d_FG, kh_G, ALPHA)
    W_B = write(Wdt, d_GF, kh_F, ALPHA)
    # sequential: re-measure the second key inside the written model
    kF_inA = mid_in_model("The capital of France is", W_A)
    drift_A = float(kF_inA @ (kF0 / np.linalg.norm(kF0))
                    / (np.linalg.norm(kF_inA) + 1e-12))
    W_AB1 = write(W_A, d_GF, kF_inA / (np.linalg.norm(kF_inA) ** 2), ALPHA)
    kG_inB = mid_in_model("The capital of Germany is", W_B)
    drift_B = float(kG_inB @ (kG0 / np.linalg.norm(kG0))
                    / (np.linalg.norm(kG_inB) + 1e-12))
    W_AB2 = write(W_B, d_FG, kG_inB / (np.linalg.norm(kG_inB) ** 2), ALPHA)
    W_ABj = write(write(Wdt, d_FG, kh_G, ALPHA), d_GF, kh_F, ALPHA)
    print(f"key drift: F-key-in-A-written-model match={drift_A:.4f}; "
          f"G-key-in-B-written-model match={drift_B:.4f} "
          f"(1.0 = no drift = commutative)", flush=True)
    for name, W in (("A-then-B", W_AB1), ("B-then-A", W_AB2),
                    ("joint", W_ABj)):
        ed = {LAYER: {"wdown": W}}
        row = []
        for pr, tgt, tname in (
                ("The capital of Germany is", paris, "Paris"),
                ("The capital of France is", berlin, "Berlin"),
                ("The capital of Italy is", None, "ctl")):
            lg, _ = fwd(pr, edits=ed)
            top = int(lg.argmax())
            if tgt is None:
                row.append(f"Italy={tok.decode([top])[:8]}")
            else:
                rk = int((lg > lg[tgt]).sum()) + 1
                row.append(f"{tname}-rank{rk}")
        print(f"{name:10}: " + " ".join(row), flush=True)


if __name__ == "__main__":
    main()
