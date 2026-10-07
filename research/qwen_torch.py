"""Canonical torch mirror for Qwen2-7B siphon probes (host-side).

Single full-precision reference both probes share: fwd() (logits with
optional residual steering) and fwdH() (per-layer residual trajectory).
Promoted here on second use (clean_screen duplicated it inline; far
probes import this). Any mirror fix lands here once, not per script.
"""
import numpy as np


def _get():
    import torch
    from chain.qwen7b import load7b
    g, tok = load7b()
    return torch, g, tok


def _rope(x, base=1000000.0):
    import torch
    dt = torch.float32
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


def _rms(x, w, eps=1e-6):
    import torch
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w


def _layer(x, g, torch, L, n, steer=None, edits=None, capture=None):
    import numpy as _np
    dt = torch.float32
    p = f"model.layers.{L}."
    w = {k: torch.tensor(g(p + k), dtype=dt, device="cuda")
         for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                   "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                   "mlp.up_proj.weight", "mlp.gate_proj.weight",
                   "mlp.down_proj.weight", "input_layernorm.weight",
                   "post_attention_layernorm.weight")}
    if edits is not None and L in edits:
        # Weight surgery (rank-1 writes etc.): {short-name: tensor}.
        # Shorts: wq/wk/wv/wo/wup/wgate/wdown/ln1/ln2 (mirrors builder).
        _short = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
                  "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
                  "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
                  "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
                  "ln2": "post_attention_layernorm.weight"}
        for k, v in edits[L].items():
            w[_short.get(k, k)] = v
    bl = f"model.layers.{L}.self_attn."
    bq = torch.tensor(g(bl + "q_proj.bias"), dtype=dt, device="cuda")
    bk = torch.tensor(g(bl + "k_proj.bias"), dtype=dt, device="cuda")
    bv = torch.tensor(g(bl + "v_proj.bias"), dtype=dt, device="cuda")
    if steer is not None and L == steer[0]:
        d, alpha = steer[1], steer[2]
        mag = float(x[n - 1].norm())
        x = x + torch.tensor(d * alpha * mag, dtype=dt, device="cuda")
    xn = _rms(x, w["input_layernorm.weight"])
    Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(n, 28, 128)
    K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(n, 4, 128)
    V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(n, 4, 128)
    K = K.repeat_interleave(7, dim=1)
    V = V.repeat_interleave(7, dim=1)
    _QR, _KR, _V = (t.permute(1, 0, 2) for t in (_rope(Q), _rope(K), V))
    P = torch.softmax(_QR @ _KR.transpose(-1, -2) / _np.sqrt(128)
                      + torch.triu(torch.full((n, n), float("-inf"),
                                              device="cuda"), 1), dim=-1)
    x = x + (P @ _V).permute(1, 0, 2).reshape(n, 3584) @ w["self_attn.o_proj.weight"].T
    hn = _rms(x, w["post_attention_layernorm.weight"])
    gate = torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
    up = hn @ w["mlp.up_proj.weight"].T
    if capture is not None:
        capture["mid"] = (gate * up).detach()
    down = (gate * up) @ w["mlp.down_proj.weight"].T
    return x + down


def fwd(prompt, steer=None, edits=None):
    """Prompt-end (logits, ids) with optional (layer, dir, gain)."""
    lg, _, ids = fwd_full(prompt, steer, edits)
    return lg, ids


def fwd_full(prompt, steer=None, edits=None):
    """Prompt-end (logits, residual, ids): steered full forward."""
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device="cuda")
    x = E
    for L in range(28):
        x = _layer(x, g, torch, L, n, steer, edits)
    lnf = torch.tensor(g("model.norm.weight"), dtype=dt, device="cuda")
    hn = _rms(x, lnf)
    h = x[n - 1].detach().cpu().numpy()
    lg = (hn @ torch.tensor(g("lm_head.weight"), dtype=dt,
                            device="cuda").T)[n - 1].detach().cpu().numpy()
    return lg, h, ids


def fwdH(prompt, layers=range(28), keep="last"):
    """Per-layer residuals: prompt-end row (keep="last") or full rows
    (keep="all", for position-specific keys)."""
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device="cuda")
    x = E
    traj = {}
    for L in range(28):
        x = _layer(x, g, torch, L, n)
        if L in layers:
            v = x.detach().cpu().numpy()
            traj[L] = v if keep == "all" else v[n - 1]
    return traj, ids


def fwd_mid(prompt, layer):
    """MID (post-SwiGLU hidden) prompt-end row at layer + weight shape."""
    import torch
    from chain.qwen7b import load7b
    torch, g, tok = _get()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].numpy()
    n = len(ids)
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device="cuda")
    x = E
    mid = {}
    for L in range(layer + 1):
        x = _layer(x, g, torch, L, n, capture=mid if L == layer else None)
    Wd = torch.tensor(g(f"model.layers.{layer}.mlp.down_proj.weight"),
                      dtype=dt, device="cuda").cpu().numpy()
    return mid["mid"][n - 1].detach().cpu().numpy(), Wd
