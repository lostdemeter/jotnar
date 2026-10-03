"""Shared Qwen2-7B-Instruct layer-0 helpers (siphon track).

Loader + boundary H builder + MLP reference for the 7B scale
(H=3584, I=18944, 28Q/4KV heads, Dh=128). Used by freeze_qwen7b_l0.py
and test_qwen7b_teach.py (two users; promotes on third). The 0.5B
mirror (chain/qwen_mirror.py) is untouched -- green gates don't churn.
"""
import os

SNAP = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-7B-Instruct", "snapshots",
                    "f2826a00ceef68f0f2b946d945ecc0477ce4450c")
N_HEADS, N_KV, DH, HIDDEN = 28, 4, 128, 3584


def load7b():
    """(get_weight fn, tokenizer). Absent-bias reads return zeros(1,)
    (broadcast-neutral). Raises ImportError without deps."""
    import json
    import numpy as np
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    cache = {}

    def f(path):
        if path not in cache:
            cache[path] = load_file(os.path.join(SNAP, path), device="cpu")
        return cache[path]

    def g(n):
        if n in wm:
            sd = f(wm[n])
            if n in sd:
                return sd[n].float().double().numpy()
        return np.zeros((1,), dtype=np.float64)

    tok = AutoTokenizer.from_pretrained(SNAP)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return g, tok


def build_H7(g, tok, prompt, n=8):
    """Real L0 residual stream (float64 boundary attention). Positions run
    on seq, shared across heads (correct RoPE broadcasting)."""
    import math
    import numpy as np
    import torch
    dt = torch.float64
    ids = tok(prompt, return_tensors="pt")["input_ids"][0][:n].numpy()
    assert len(ids) == n, f"prompt gives {len(ids)} tokens, need {n}"
    toks = tok.convert_ids_to_tokens(ids)
    E = g("model.embed_tokens.weight")[np.array(ids)]
    xt = torch.tensor(E, dtype=dt)
    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt)
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1000000.0):
        Sq, D = x.shape[-2], x.shape[-1]
        i = torch.arange(D // 2, dtype=dt)
        th = base ** (-2.0 * i / D)
        ang = torch.arange(Sq, dtype=dt)[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        while c.dim() < x.dim():
            c, s = c.unsqueeze(0), s.unsqueeze(0)
        y = torch.empty_like(x)
        y[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
        y[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
        return y

    Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt)
    Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt)
    Wv = torch.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=dt)
    Wo = torch.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=dt)
    bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt)
    bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt)
    bv = torch.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=dt)
    xn = rms(xt, ln1)
    Q = (xn @ Wq.T + bq).reshape(n, N_HEADS, DH)
    K = (xn @ Wk.T + bk).reshape(n, N_KV, DH)
    V = (xn @ Wv.T + bv).reshape(n, N_KV, DH)
    rep = N_HEADS // N_KV
    K = K.repeat_interleave(rep, dim=1)
    V = V.repeat_interleave(rep, dim=1)
    _QR, _KR, _V = (rope(t) for t in (Q, K, V))
    _QR, _KR, _V = (t.permute(1, 0, 2) for t in (_QR, _KR, _V))
    SC = (_QR @ _KR.transpose(-1, -2)) / math.sqrt(DH)
    P = torch.softmax(SC + torch.triu(
        torch.full((n, n), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = (P @ _V).permute(1, 0, 2).reshape(n, HIDDEN)
    return (xt + CTX @ Wo.T).numpy(), ids, toks


def mlp7_forward(H, g):
    """L0 MLP reference (float64). Returns (REF, MID, Wup, Wg, Wd, ln2)."""
    import numpy as np
    import torch
    dt = torch.float64
    eps = 1e-6
    ln2 = torch.tensor(g("model.layers.0.post_attention_layernorm.weight"),
                       dtype=dt)
    Wup = torch.tensor(g("model.layers.0.mlp.up_proj.weight"), dtype=dt)
    Wg = torch.tensor(g("model.layers.0.mlp.gate_proj.weight"), dtype=dt)
    Wd = torch.tensor(g("model.layers.0.mlp.down_proj.weight"), dtype=dt)

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    HN = rms(torch.tensor(H, dtype=dt), ln2)
    UP, GATE = HN @ Wup.T, HN @ Wg.T
    MID = torch.nn.functional.silu(GATE) * UP
    DOWN = MID @ Wd.T
    REF = (torch.tensor(H, dtype=dt) + DOWN).numpy()
    return (REF, MID.numpy(), Wup.numpy(), Wg.numpy(), Wd.numpy(),
            ln2.numpy())
