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

    # Memory-safe path hook: streaming callers release the whole-file
    # cache when done (peak would otherwise pin ~15GB bf16 for the
    # process lifetime). Arity unchanged (g, tok) -- attribute only.
    g.clear_cache = cache.clear  # noqa: attribute on closure
    g._cache = cache

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
        # x: (seq, heads, dim). Positions arange(seq), shared across
        # heads (explicit layout -- no dim guessing).
        Sq, Hh, D = x.shape
        i = torch.arange(D // 2, dtype=dt)
        th = base ** (-2.0 * i / D)
        ang = torch.arange(Sq, dtype=dt)[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        c, s = c[:, None, :], s[:, None, :]
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
    _QR, _KR = rope(Q), rope(K)
    _QR, _KR, _V = (t.permute(1, 0, 2) for t in (_QR, _KR, V))
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


# -- 28-layer memory-safe path (post-SSD guard; L0 helpers above untouched)
QWEN7B_INTER = 18944
QWEN7B_VOCAB = 152064
QScale = 128.0 ** -0.5  # temperature fold 1/sqrt(dh), exact by linearity


def snapshot_ok():
    """True when the 7B-Instruct snapshot is present (weights, not stub)."""
    return os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json"))


def shapes_7b(smax):
    """Zero-RAM compile payload for the 28-layer listing (chain.mem buddy).

    Returns {stream: FRef/IRef} with listing-convention shapes (K-major:
    matrices transposed, biases tiled to (S, dim)). compile_program only
    reads shapes/kinds through stubs -- values stream to disk one weight
    at a time via prep_7b_weight(). Small streams (pos/cmask/x0) are NOT
    stubbed: rope tables need real pos values (emit_c refuses stubs).
    """
    from chain.emit_c import FRef, IRef
    import numpy as np
    H, I_, S = HIDDEN, QWEN7B_INTER, smax
    KV = N_KV * DH
    sample = {"pos": np.arange(S, dtype=np.int64),
              "cmask": np.tril(np.ones((S, S), dtype=np.int64)),
              "x0": FRef((S, H), "float16")}
    per = N_HEADS // N_KV
    sample["cmask7"] = IRef((per * S, S), "int64")
    per = {"wq": (H, H), "wk": (H, KV), "wv": (H, KV), "wo": (H, H),
           "wup": (H, I_), "wgate": (H, I_), "wdown": (I_, H),
           "ln1": (H,), "ln2": (H,),
           "bq": (S, H), "bk": (S, KV), "bv": (S, KV)}
    for L in range(28):
        for k, sh in per.items():
            sample[f"{k}{L}"] = FRef(sh, "float16")
    sample["lnf"] = FRef((H,), "float16")
    sample["wlog"] = FRef((H, QWEN7B_VOCAB), "float16")
    _ = IRef  # integer stubs unused here (pos/cmask stay real); kept explicit
    return sample


def prep_7b_weight(g, dotted, short, smax):
    """One 7B weight as fp16 in listing convention (caller dels + writes).

    Mirrors the retired bulk path exactly (transpose K-major; wq AND bq
    scaled by 1/sqrt(128) -- forgetting bq cost 0.5 rel, measured;
    biases tiled host-side to (S, dim)). Scale math runs in float32
    then casts to fp16: rounding differs from the old float64-then-cast
    by < fp16 quantum (inside the parity bar, priced in test_qwen7b_gen).
    Peak per call: one fp64 source transient + one fp16 out.
    """
    import numpy as np
    w = g(dotted)
    if short in ("ln1", "ln2", "lnf"):
        return np.ascontiguousarray(w, dtype=np.float16)
    if short in ("bq", "bk", "bv"):
        t = np.ascontiguousarray(np.tile(w, (smax, 1)), dtype=np.float32)
        if short == "bq":
            t = t * np.float32(QScale)
        out = t.astype(np.float16)
        del t
        return np.ascontiguousarray(out)
    t = np.ascontiguousarray(w.T, dtype=np.float32)
    if short == "wq":
        t = t * np.float32(QScale)
    out = t.astype(np.float16)
    del t
    return np.ascontiguousarray(out)


def clear_7b_cache(g):
    """Release the safetensor file cache (attached by load7b, if present)."""
    fn = getattr(g, "clear_cache", None)
    if callable(fn):
        fn()


_QWEN7B_KEYMAP = {"wq": "self_attn.q_proj.weight",
                  "wk": "self_attn.k_proj.weight",
                  "wv": "self_attn.v_proj.weight",
                  "wo": "self_attn.o_proj.weight",
                  "wup": "mlp.up_proj.weight",
                  "wgate": "mlp.gate_proj.weight",
                  "wdown": "mlp.down_proj.weight",
                  "ln1": "input_layernorm.weight",
                  "ln2": "post_attention_layernorm.weight",
                  "bq": "self_attn.q_proj.bias",
                  "bk": "self_attn.k_proj.bias",
                  "bv": "self_attn.v_proj.bias"}


def input_sources():
    """Listing input name -> (short, dotted weight name), all 28 layers.

    Single source of truth for the streaming dump loops (exact-match
    lookup -- never parse trailing digits: ln1@L0 is "ln10" but ln1@L10
    is "ln110"). Plus ("lnf", None) and ("wlog", None) globals.
    """
    out = {f"{s}{L}": (s, f"model.layers.{L}.{src}")
           for s, src in _QWEN7B_KEYMAP.items() for L in range(28)}
    out["lnf"] = ("lnf", "model.norm.weight")
    out["wlog"] = ("wlog", "lm_head.weight")
    return out


def hf_no_triton():
    """Disable torch's triton eager overrides for HF reference runs.

    torch >= 2.14 routes some aten ops (e.g. the bmm inside RoPE) through
    triton JIT kernels, which need a full dev toolchain (Python.h, libcuda
    link) just to compile their driver shim. The reference only needs
    correct aten numerics (5-token comparisons), so drop back to aten via
    the supported registry filter. No-op on torch without the API (where
    no such override exists). Must run before the first CUDA op.
    """
    try:
        from torch._native import registry as _reg
        _reg.deregister_op_overrides(disable_dsl_names="triton")
    except ImportError:
        pass
