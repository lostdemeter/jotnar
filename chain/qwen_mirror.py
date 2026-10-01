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


def block_inputs(prompt, n=8):
    """Full boundary bundle for projection-level tests (5th inline mirror
    consolidated HERE going forward; older copies migrate on touch): E,
    XN (rmsnorm'd embeddings), H (post-attention residual), CTX, HN,
    MID, and weight dict W (q/k/v/o/up/gate/down/ln1/ln2 as float64).
    All torch.float64, no lattice (boundary by doctrine)."""
    import torch
    import numpy as np
    g, embed, tok = load_layer0()
    H, ids, toks = build_H(g, embed, tok, prompt, n)
    dt = torch.float64
    eps = 1e-6
    E = embed(ids)
    xt = torch.tensor(E, dtype=dt)
    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt)

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    XN = rms(xt, ln1).numpy()
    REF, MID, Wupf, Wgf, Wdf, ln2f = mlp_forward(H, g)
    # CTX + HN recomputed (duplicates build_H internals ~15 lines; this fn
    # is the consolidation point going forward, build_H delegates later).
    Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt)
    Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt)
    Wv = torch.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=dt)
    Wo = torch.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=dt)
    bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt)
    bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt)
    bv = torch.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=dt)

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

    _Q = (rms(xt, ln1) @ Wq.T + bq).reshape(n, 14, 64)
    _K = (rms(xt, ln1) @ Wk.T + bk).reshape(n, 2, 64)
    _V = (rms(xt, ln1) @ Wv.T + bv).reshape(n, 2, 64)
    _K = _K.repeat_interleave(7, dim=1)
    _V = _V.repeat_interleave(7, dim=1)
    _QR, _KR = rope(_Q.permute(1, 0, 2)), rope(_K.permute(1, 0, 2))
    _SC = (_QR @ _KR.transpose(-1, -2)) / 8.0
    _P = torch.softmax(_SC + torch.triu(
        torch.full((n, n), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = ((_P @ _V.permute(1, 0, 2)).permute(1, 0, 2).reshape(n, 896)).numpy()
    ln2 = torch.tensor(g("model.layers.0.post_attention_layernorm.weight"),
                       dtype=dt)
    HN = rms(torch.tensor(H, dtype=dt), ln2).numpy()
    W = {"q": g("model.layers.0.self_attn.q_proj.weight"),
         "k": g("model.layers.0.self_attn.k_proj.weight"),
         "v": g("model.layers.0.self_attn.v_proj.weight"),
         "o": g("model.layers.0.self_attn.o_proj.weight"),
         "up": Wupf, "gate": Wgf, "down": Wdf,
         "ln1": ln1.numpy(), "ln2": ln2f}
    return {"E": E, "XN": XN, "H": H, "CTX": CTX, "HN": HN, "MID": MID,
            "REF": REF, "W": W, "ids": ids, "toks": toks}


def mlp_forward(H, g):
    """MLP block in torch.float64 (the test_realw mirror, promoted): returns
    (REF output, MID intermediates, Wupf, Wgf, Wdf, ln2f as float64 numpy).
    Boundary math for listing runs; parity claim lives in test_realw."""
    import torch
    import numpy as np
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
