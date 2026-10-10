"""Zeta-phase rotary swap: do the zeros carry load, or any rotation?

RoPE: angle(m,j) = m * base^(-2j/D). Replacement: angle(m,j) =
gamma_j * ln(m+1), gamma_j = imaginary parts of the first 64 zeta
zeros (frozen to data/zeta_gamma64.json on first run -- doctrine:
formula tables, computed once, committed). Controls: permuted gammas
(same multiset, assignment destroyed) + uniform random in range
(any fast rotation). Local full-forward mirror (shared qwen_torch
untouched); battery reports top-1/agreement-with-stock/margin per
mode. Zeta ~= random controls -> frequencies don't matter (rotation
alone); zeta >> random -> zeros carry something (fundamental);
both collapse vs stock -> RoPE's profile is load-bearing after all.
Usage: python3 research/zeta_rope.py (needs 7B snapshot + GPU)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

GAMMA_FILE = os.path.join(ROOT, "data", "zeta_gamma64.json")
PROMPTS = ["The capital of Germany is", "The capital of Italy is",
           "The capital of Japan is", "Shakespeare wrote the play",
           "The tallest mountain on Earth is", "George Orwell wrote the novel"]


def gammas():
    if os.path.isfile(GAMMA_FILE):
        return json.load(open(GAMMA_FILE))
    import mpmath
    mpmath.mp.dps = 15
    g = [float(mpmath.im(mpmath.zetazero(k))) for k in range(1, 65)]
    json.dump(g, open(GAMMA_FILE, "w"))
    return g


def fwd_custom(prompt, cos_fn):
    """Full 28-layer forward with caller-supplied rotary tables.

    cos_fn(Sq, Dh) -> (cos, sin) as (Sq,1,Dh/2) broadcastables.
    Mirrors qwen_torch._layer math otherwise (float32, CUDA).
    """
    from qwen_torch import _get, _rms
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt, device="cuda")
    x = E
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
        xn = _rms(x, w["input_layernorm.weight"])
        Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(n, 28, 128)
        K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(n, 4, 128)
        V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(n, 4, 128)
        K = K.repeat_interleave(7, dim=1)
        V = V.repeat_interleave(7, dim=1)
        c, s = cos_fn(n, 128)
        y = torch.empty_like(Q)
        y[..., 0::2] = Q[..., 0::2] * c - Q[..., 1::2] * s
        y[..., 1::2] = Q[..., 0::2] * s + Q[..., 1::2] * c
        _QR = y.permute(1, 0, 2)
        y = torch.empty_like(K)
        y[..., 0::2] = K[..., 0::2] * c - K[..., 1::2] * s
        y[..., 1::2] = K[..., 0::2] * s + K[..., 1::2] * c
        _KR = y.permute(1, 0, 2)
        _V = V.permute(1, 0, 2)
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((n, n), float("-inf"),
                                                  device="cuda"), 1), dim=-1)
        x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
        hn = _rms(x, w["post_attention_layernorm.weight"])
        down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                * (hn @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
        x = x + down
    lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
    hn = _rms(x, lnf)
    lg = (hn @ torch.tensor(g("lm_head.weight"), dtype=dt,
                            device="cuda").T)[n - 1].detach().cpu().numpy()
    return lg


def main():
    from chain.qwen7b import load7b, snapshot_ok, hf_no_triton
    hf_no_triton()
    if not snapshot_ok():
        print("SKIP (needs Qwen2-7B-Instruct snapshot)")
        return
    import torch
    _, tok = load7b()
    gm = gammas()
    print(f"gammas frozen: {len(gm)} [{gm[0]:.2f}..{gm[-1]:.2f}]", flush=True)
    rng = np.random.default_rng(0)
    perm = np.array(gm)[rng.permutation(len(gm))].tolist()
    uni = rng.uniform(min(gm), max(gm), len(gm)).tolist()

    def tables(freqs):
        def fn(Sq, Dh):
            ln = np.log(np.arange(1, Sq + 1, dtype=np.float64))[:, None]
            ang = ln * np.asarray(freqs, dtype=np.float64)[None, :Dh // 2]
            c = torch.tensor(np.cos(ang), dtype=torch.float32,
                             device="cuda")[:, None, :]
            s = torch.tensor(np.sin(ang), dtype=torch.float32,
                             device="cuda")[:, None, :]
            return c, s
        return fn

    def rope_tables(Sq, Dh, base=1000000.0):
        i = torch.arange(Dh // 2, dtype=torch.float32, device="cuda")
        th = base ** (-2.0 * i / Dh)
        ang = torch.arange(Sq, dtype=torch.float32, device="cuda")[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        return c[:, None, :], s[:, None, :]

    # scaled: gammas x0.0198 so max angle == RoPE max (~7 rad at m=7):
    # separates "too fast" from "wrong structure". If scaled agrees
    # with stock, frequencies carry load (up to global scale); if it
    # collapses like the rest, structure is wrong, not just hot.
    SC = 7.0 / (max(gm) * np.log(8))
    zsc = (np.array(gm) * SC).tolist()
    print(f"scaled range: [{min(zsc):.3f}..{max(zsc):.3f}]", flush=True)
    modes = {"stock": rope_tables, "zeta": tables(gm), "perm": tables(perm),
             "uniform": tables(uni), "zetascaled": tables(zsc)}
    base = {}
    for pr in PROMPTS:
        from qwen_torch import fwd as _fwd
        lg, _ = _fwd(pr)
        base[pr] = int(lg.argmax())
    print("stock tops:", {p[:22]: tok.decode([v]) for p, v in base.items()},
          flush=True)
    for name, fn in modes.items():
        agree, gaps = 0, []
        for pr in PROMPTS:
            lg = fwd_custom(pr, fn)
            top = int(lg.argmax())
            agree += top == base[pr]
            o = np.argsort(-lg)
            gaps.append(float(lg[o[0]] - lg[o[1]]))
            print(f"  {name:7} {pr[:24]!r}: top={tok.decode([top])[:14]!r} "
                  f"{'AGREE' if top == base[pr] else '--'}", flush=True)
        print(f"{name:7}: agreement {agree}/{len(PROMPTS)} "
              f"mean-margin {float(np.mean(gaps)):.2f}", flush=True)


if __name__ == "__main__":
    main()
