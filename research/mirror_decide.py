"""Mirror-vs-mirror decider: research fwdH vs ladder-style loop, ONE process.

Background: cross-process comparisons showed research-vs-ladder
3.6-8.5 abs, but _layer-vs-manual matched at 0.0 in-process. This
runs both trajectories with the SAME loader/objects in one process:
match -> mirrors agree, discrepancy was environmental (document,
move on); mismatch -> real code difference, bisect continues.
Usage: python3 research/mirror_decide.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

PROMPT = "The capital of Germany is"


def main():
    import torch
    import qwen_torch as QT
    from chain.qwen7b import load7b
    g, tok = load7b()
    dt = torch.float32
    print("mirror file:", QT.__file__, flush=True)
    t, ids = QT.fwdH(PROMPT)
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device="cuda")
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1000000.0):
        Sq, Hh, D = x.shape
        i = torch.arange(D // 2, dtype=dt, device="cuda")
        th = base ** (-2.0 * i / D)
        ang = torch.arange(Sq, dtype=dt, device="cuda")[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        c, s = c[:, None, :], s[:, None, :]
        y = torch.empty_like(x)
        y[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
        y[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
        return y

    x = E
    ok = True
    for L in range(28):
        p = f"model.layers.{L}."
        w = {k: torch.tensor(g(p + k), dtype=dt, device="cuda")
             for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.up_proj.weight", "mlp.gate_proj.weight",
                       "mlp.down_proj.weight", "input_layernorm.weight",
                       "post_attention_layernorm.weight")}
        bl = f"model.layers.{L}.self_attn."
        bq = torch.tensor(g(bl + "q_proj.bias"), dtype=dt, device="cuda")
        bk = torch.tensor(g(bl + "k_proj.bias"), dtype=dt, device="cuda")
        bv = torch.tensor(g(bl + "v_proj.bias"), dtype=dt, device="cuda")
        xn = rms(x, w["input_layernorm.weight"])
        Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(n, 28, 128)
        K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(n, 4, 128)
        V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(n, 4, 128)
        K = K.repeat_interleave(7, dim=1)
        V = V.repeat_interleave(7, dim=1)
        _QR, _KR, _V = (z.permute(1, 0, 2) for z in (rope(Q), rope(K), V))
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((n, n), float("-inf"),
                                                  device="cuda"), 1), dim=-1)
        x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
        hn = rms(x, w["post_attention_layernorm.weight"])
        x = x + (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                 * (hn @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
        if L in (0, 2, 8, 16, 26, 27):
            d = float(np.abs(t[L] - x.detach().cpu().numpy()[n - 1]).max())
            flag = "OK" if d < 1e-4 else "DIFF"
            if d >= 1e-4:
                ok = False
            print(f"L{L}: {flag} maxabs={d:.6f}", flush=True)
    print("VERDICT:", "mirrors agree" if ok else "real code difference",
          flush=True)


if __name__ == "__main__":
    main()
