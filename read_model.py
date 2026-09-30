"""Model read (one-shot, not a gate): full directional readout of real down_proj.

Per sampled direction: ablate, record the per-TOKEN delta vector (the
functional fingerprint) + global dB + sval. Output feeds docs/MODEL_READ.md
(the demonstration) and validates chain/read.py queries against brute force.
Slow (~112 listing runs); run once, commit the table as evidence.
"""
import os
import sys
import time

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

QWEN = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-0.5B", "snapshots",
                    "91d2aff3f957f99e4c74c962f2f408dcc88a18d8")


def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    t0 = time.time()
    sd = load_file(os.path.join(QWEN, "model.safetensors"), device="cpu")
    g = lambda n: sd[n].float().double().numpy()
    tok = AutoTokenizer.from_pretrained(QWEN)
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt")["input_ids"][0][:8].numpy()
    toks = tok.convert_ids_to_tokens(ids)
    print("tokens:", toks)
    E = sd["model.embed_tokens.weight"][torch.tensor(ids)].float().double().numpy()
    dt = torch.float64
    xt = torch.tensor(E, dtype=dt)
    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt)
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt)
    Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt)
    Wv = torch.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=dt)
    Wo = torch.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=dt)
    bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt)
    bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt)
    bv = torch.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=dt)
    xn = rms(xt, ln1)

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

    Q = (xn @ Wq.T + bq).reshape(8, 14, 64)
    K = (xn @ Wk.T + bk).reshape(8, 2, 64)
    V = (xn @ Wv.T + bv).reshape(8, 2, 64)
    K = K.repeat_interleave(7, dim=1)
    V = V.repeat_interleave(7, dim=1)
    QR, KR = rope(Q.permute(1, 0, 2)), rope(K.permute(1, 0, 2))
    SC = (QR @ KR.transpose(-1, -2)) / 8.0
    P = torch.softmax(SC + torch.triu(
        torch.full((8, 8), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = (P @ V.permute(1, 0, 2)).permute(1, 0, 2).reshape(8, 896)
    H = (xt + CTX @ Wo.T).numpy()
    Wupf = g("model.layers.0.mlp.up_proj.weight")
    Wgf = g("model.layers.0.mlp.gate_proj.weight")
    Wdf = g("model.layers.0.mlp.down_proj.weight")
    ln2f = g("model.layers.0.post_attention_layernorm.weight")

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    pay0 = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}

    def run_down(WdT):
        pay = dict(pay0)
        pay["wdown"] = enc(WdT)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    base = run_down(Wdf.T)
    WdT = Wdf.T
    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)
    idx = list(range(0, 896, 8))
    rows = []
    for i in idx:
        Wi = WdT - np.outer(U[:, i] * s[i], Vt[i])
        got = run_down(Wi)
        mse_tok = ((got - base) ** 2).mean(-1)
        tokdb = np.array([float("inf") if v == 0 else 10 * np.log10(1.0 / v)
                          for v in mse_tok])
        mse = float(((got - base) ** 2).mean())
        rows.append((i, s[i], float("inf") if mse == 0 else 10 * np.log10(1.0 / mse),
                     tokdb))
        if len(rows) % 28 == 0:
            print(f"  {len(rows)}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez("/tmp/qwen0_readout.npz",
             idx=np.array([r[0] for r in rows]),
             sval=np.array([r[1] for r in rows]),
             gdb=np.array([r[2] for r in rows]),
             tokdb=np.stack([r[3] for r in rows]))
    print(f"done {len(rows)} dirs in {time.time()-t0:.0f}s -> /tmp/qwen0_readout.npz")


if __name__ == "__main__":
    main()
