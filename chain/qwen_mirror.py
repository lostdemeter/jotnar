"""Shared Qwen2-0.5B layer-0 boundary helpers (promoted, third copy).

test_realw.py, test_implant.py, and read_model.py each carried this mirror
inline (the promotion rule fires at three uses: select_mux precedent).
Attention lives here as boundary float (folded scores hit ~964, 1000x past
the softmax contract -- T-transform backlog); MLP weights + H come out as
triples-ready float64. New code uses this; the three inline copies migrate
on touch (backlog, stated -- do not churn green gates for aesthetics).
"""
import os

QWEN = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-0.5B", "snapshots",
                    "91d2aff3f957f99e4c74c962f2f408dcc88a18d8")


def load_layer0():
    """(get_weight fn, embed fn, tokenizer). Raises ImportError without deps."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    sd = load_file(os.path.join(QWEN, "model.safetensors"), device="cpu")
    g = lambda n: sd[n].float().double().numpy()
    tok = AutoTokenizer.from_pretrained(QWEN)
    os.environ["HF_HUB_OFFLINE"] = "1"

    def embed(ids):
        import torch as _t
        return sd["model.embed_tokens.weight"][_t.tensor(ids)].float().double().numpy()

    return g, embed, tok


def build_H(g, embed, tok, prompt, n=8):
    """Real residual stream: embeddings + full float64 attention (biases,
    GQA-7x, RoPE-1e6, /8 folding, causal mask). Returns (H, ids, toks).
    Needs exactly n tokens (asserts -- shape bugs fail loud)."""
    import torch
    import numpy as np
    dt = torch.float64
    ids = tok(prompt, return_tensors="pt")["input_ids"][0][:n].numpy()
    assert len(ids) == n, f"prompt gives {len(ids)} tokens, need {n}"
    toks = tok.convert_ids_to_tokens(ids)
    E = embed(ids)
    xt = torch.tensor(E, dtype=dt)
    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt)
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1e6):
        Sq, Dh = x.shape[0], x.shape[-1]
        i = torch.arange(Dh // 2, dtype=dt)
        th = base ** (-2 * i / Dh)
        ang = torch.arange(Sq, dtype=dt)[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        y = torch.empty_like(x)
        y[..., 0::2] = x[..., 0::2] * c.unsqueeze(-2) - x[..., 1::2] * s.unsqueeze(-2)
        y[..., 1::2] = x[..., 0::2] * s.unsqueeze(-2) + x[..., 1::2] * c.unsqueeze(-2)
        return y

    Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt)
    Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt)
    Wv = torch.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=dt)
    Wo = torch.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=dt)
    bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt)
    bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt)
    bv = torch.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=dt)
    xn = rms(xt, ln1)
    Q = (xn @ Wq.T + bq).reshape(n, 14, 64)
    K = (xn @ Wk.T + bk).reshape(n, 2, 64)
    V = (xn @ Wv.T + bv).reshape(n, 2, 64)
    K = K.repeat_interleave(7, dim=1)
    V = V.repeat_interleave(7, dim=1)
    QR, KR = rope(Q.permute(1, 0, 2)), rope(K.permute(1, 0, 2))
    SC = (QR @ KR.transpose(-1, -2)) / 8.0
    P = torch.softmax(SC + torch.triu(
        torch.full((n, n), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = (P @ V.permute(1, 0, 2)).permute(1, 0, 2).reshape(n, 896)
    H = (xt + CTX @ Wo.T).numpy()
    return H, ids, toks
