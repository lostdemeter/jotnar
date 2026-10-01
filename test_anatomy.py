"""Anatomy instance 1 (v1.6 gate 1): SmolLM2-135M layer-0 MLP, two-stage.

Same SwiGLU shape as Qwen (RMSNorm + gate/up/silu/mul/down + residual)
with different numbers (hidden 576, eps 1e-5, outputs ~23, NO biases):
two-scale execution (tight stage A, wide stage B) + TRUE-eps (eps_rms).
Proves the comparative method transfers: second family, same gates.
SKIPs without the local HF cache.
Usage: python3 test_anatomy.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"SKIP ({e})")
        sys.exit(0)
    snap = None
    base = os.path.join(os.path.expanduser("~"), ".cache", "huggingface",
                        "hub", "models--HuggingFaceTB--SmolLM2-135M",
                        "snapshots")
    if os.path.isdir(base):
        subs = os.listdir(base)
        if subs:
            snap = os.path.join(base, subs[0])
    if snap is None or not os.path.isfile(os.path.join(snap, "model.safetensors")):
        print("SKIP (needs SmolLM2-135M in local HF cache)")
        sys.exit(0)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    sd = load_file(os.path.join(snap, "model.safetensors"), device="cpu")
    g = lambda n: sd[n].float().double().numpy()
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt")["input_ids"][0][:8].numpy()
    E = sd["model.embed_tokens.weight"][torch.tensor(ids)].float().double().numpy()
    dt = torch.float64
    xt = torch.tensor(E, dtype=dt)
    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt)
    ln2 = torch.tensor(g("model.layers.0.post_attention_layernorm.weight"), dtype=dt)
    eps = 1e-5

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1e5):
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
    xn = rms(xt, ln1)
    Q = (xn @ Wq.T).reshape(8, 9, 64)
    K = (xn @ Wk.T).reshape(8, 3, 64)
    V = (xn @ Wv.T).reshape(8, 3, 64)
    K = K.repeat_interleave(3, dim=1)
    V = V.repeat_interleave(3, dim=1)
    QR, KR = rope(Q.permute(1, 0, 2)), rope(K.permute(1, 0, 2))
    SC = (QR @ KR.transpose(-1, -2)) / 8.0
    P = torch.softmax(SC + torch.triu(
        torch.full((8, 8), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = (P @ V.permute(1, 0, 2)).permute(1, 0, 2).reshape(8, 576)
    H = (xt + CTX @ Wo.T).numpy()
    Wup = torch.tensor(g("model.layers.0.mlp.up_proj.weight"), dtype=dt)
    Wg = torch.tensor(g("model.layers.0.mlp.gate_proj.weight"), dtype=dt)
    Wd = torch.tensor(g("model.layers.0.mlp.down_proj.weight"), dtype=dt)
    HN = rms(torch.tensor(H, dtype=dt), ln2)
    UP, GATE = HN @ Wup.T, HN @ Wg.T
    DOWN = (torch.nn.functional.silu(GATE) * UP) @ Wd.T
    REF = (torch.tensor(H, dtype=dt) + DOWN).numpy()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(root, "programs")
    ta = open(os.path.join(sdir, "mlp_smolm2_a.asm")).read()
    tb = open(os.path.join(sdir, "mlp_smolm2_b.asm")).read()
    fa = ASM.run_text(ta, REGISTRY,
                      {"H": enc(H), "wup": enc(Wup.numpy().T),
                       "wgate": enc(Wg.numpy().T), "ln": enc(ln2.numpy())},
                      sigs=SIGS, basedir=sdir)
    fb = ASM.run_text(tb, REGISTRY,
                      {"MID": fa["MID"], "H": enc(H),
                       "wdown": enc(Wd.numpy().T)},
                      sigs=SIGS, basedir=sdir)
    got = dec(fb["OUT"])
    mse = float(np.mean((got - REF) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("anatomy-smolm2", d >= 40.0,
          f"{d:.1f}dB two-stage + true-eps (peak=1.0)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
