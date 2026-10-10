"""RoPE-as-phase: attention mass vs relative rotary phase, content fixed.

Same token X at positions 0 and m (filler between, ids specified
directly -- no tokenizer merges): content identical, distance varies.
Measure L0 P[m,0] (query at second X onto key at first X) for m=1..7
(S<=8 window). Content/position-decay predicts MONOTONE falloff;
rotary phase predicts OSCILLATION (theta_i periods across Dh=128
dims: non-monotonic mass vs distance). Oscillation = phase drives
mass beyond content (spacetime inherent in geometry, priced);
monotone = RoPE contributes nothing separable here (hypothesis dies
at L0, deeper layers confounded by contextualization -- stated).
X tokens x4 (token-specifics ruled out). L0 only, by design.
Usage: python3 research/rope_phase.py (needs 7B snapshot + GPU)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

XS = [1, 2, 3, 261]  # the, and, of, Italy (common singles)
FILL = [11, 7, 5, 28, 42, 6, 0]


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    import torch
    from qwen_torch import _get, _rope
    torch, g, tok = _get()
    inv = {i: tok.decode([i]) for i in XS + FILL}
    dt = torch.float32
    for xid in XS:
        curve = []
        for m in range(1, 8):
            ids = np.array([xid] + FILL[:m - 1] + [xid], dtype=np.int64)
            n = len(ids)
            assert n == m + 1 <= 8, (n, m)
            E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                             device="cuda")
            eps = 1e-6

            def rms(x, w):
                return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

            ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"),
                               dtype=dt)
            xn = rms(E, ln1)
            Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt)
            Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt)
            bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt)
            bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt)
            Q = (xn @ Wq.T + bq).reshape(n, 28, 128)
            K = (xn @ Wk.T + bk).reshape(n, 4, 128)
            K = K.repeat_interleave(7, dim=1)
            _QR, _KR, _ = (t.permute(1, 0, 2)
                           for t in (_rope(Q), _rope(K),
                                     torch.zeros(n, 28, 128)))
            P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                              + torch.triu(torch.full((n, n), float("-inf"),
                                                      device="cuda"), 1), dim=-1)
            P = P.detach().cpu().numpy()
            curve.append(float(P[:, m, 0].mean()))
        ds = np.diff(curve)
        signs = np.sign(ds)
        flips = int(((signs[1:] * signs[:-1]) < 0).sum())
        print(f"X={inv.get(xid, xid)!r}: " + " ".join(f"{v:.3f}" for v in curve)
              + f" sign-flips={flips}/5 "
              f"({'OSCILLATES' if flips >= 2 else 'monotone-ish'})", flush=True)


if __name__ == "__main__":
    main()
