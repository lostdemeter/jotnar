"""Attn-program isolation: hand-fed caches + q vs torch layer-0 row.

Feeds the 1-layer attn graph torch-computed q_rot/K/V (no prog1, no
relay) and compares y against torch's full layer-0 output row.
MATCH -> bug is in relay/files; MISMATCH -> bug in the attn graph.
Usage: python3 scripts/probe_attn.py
"""
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

S, HID, KVD, NH, NKV, DH, PER = 16, 3584, 512, 28, 4, 128, 7
PROMPT = "The capital of France is"
N = 5  # use last prompt row


def main():
    import torch
    from chain.qwen7b import load7b
    g, tok = load7b()
    dt = torch.float32
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][:N].numpy()
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

    ln1 = torch.tensor(g("model.layers.0.input_layernorm.weight"), dtype=dt,
                       device="cuda")
    Wq = torch.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=dt,
                      device="cuda")
    Wk = torch.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=dt,
                      device="cuda")
    Wv = torch.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=dt,
                      device="cuda")
    Wo = torch.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=dt,
                      device="cuda")
    bq = torch.tensor(g("model.layers.0.self_attn.q_proj.bias"), dtype=dt,
                      device="cuda")
    bk = torch.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=dt,
                      device="cuda")
    bv = torch.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=dt,
                      device="cuda")
    xn = rms(E, ln1)
    Q = (xn @ Wq.T + bq).reshape(N, NH, DH)
    K = (xn @ Wk.T + bk).reshape(N, NKV, DH)
    V = (xn @ Wv.T + bv).reshape(N, NKV, DH)
    rep = NH // NKV
    K = K.repeat_interleave(rep, dim=1)
    V = V.repeat_interleave(rep, dim=1)
    QR, KR = rope(Q), rope(K)
    _QR, _KR, _V = (t.permute(1, 0, 2) for t in (QR, KR, V))
    SC = (_QR @ _KR.transpose(-1, -2)) / np.sqrt(DH)
    P = torch.softmax(SC + torch.triu(
        torch.full((N, N), float("-inf"), dtype=dt, device="cuda"), 1), dim=-1)
    CTX = (P @ _V).permute(1, 0, 2).reshape(N, HID)
    H = E + CTX @ Wo.T
    ln2 = torch.tensor(g("model.layers.0.post_attention_layernorm.weight"),
                       dtype=dt, device="cuda")
    Wup = torch.tensor(g("model.layers.0.mlp.up_proj.weight"), dtype=dt,
                       device="cuda")
    Wg = torch.tensor(g("model.layers.0.mlp.gate_proj.weight"), dtype=dt,
                      device="cuda")
    Wd = torch.tensor(g("model.layers.0.mlp.down_proj.weight"), dtype=dt,
                      device="cuda")
    hn = rms(H, ln2)
    Y = H + (torch.nn.functional.silu(hn @ Wg.T) * (hn @ Wup.T)) @ Wd.T
    ref = Y[N - 1].detach().cpu().numpy()
    # caches: rotated-K rows + V rows for prefix, fp16 files
    Kc = np.zeros((S, KVD), dtype=np.float16)
    Vc = np.zeros((S, KVD), dtype=np.float16)
    # caches: rotated-K group rows + V group rows for the prefix.
    # (KR/V repeat_interleave(7): every 7th head is a group head.)
    Kc[:N] = KR[:, ::7, :].reshape(N, KVD).detach().cpu().numpy().astype(
        np.float16)
    Vc[:N] = V[:, ::7, :].reshape(N, KVD).detach().cpu().numpy().astype(
        np.float16)
    qrow = QR[N - 1].reshape(1, HID).detach().cpu().numpy().astype(np.float16)

    from chain.builder import Prog, qwen_attn_layer
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    p = Prog("attn1")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    p.inp("x0", "qr", "K", "V", "wo", "wup", "wgate", "wdown", "ln2",
          "attmask7", "allrows")
    y = qwen_attn_layer(p, "x0", "qr", "K", "V", "wo", "wup", "wgate",
                        "wdown", "ln2", "attmask7", S, pre="0")
    from chain.emit_c import FRef, IRef
    sample = {"x0": FRef((1, HID), "float16"),
              "qr": FRef((1, HID), "float16"),
              "K": FRef((S, KVD), "float16"),
              "V": FRef((S, KVD), "float16"),
              "wo": FRef((HID, HID), "float16"),
              "wup": FRef((HID, 18944), "float16"),
              "wgate": FRef((HID, 18944), "float16"),
              "wdown": FRef((18944, HID), "float16"),
              "ln2": FRef((HID,), "float16"),
              "attmask7": IRef((PER, S), "int64"),
              "allrows": np.arange(S, dtype=np.int64)}
    art = compile_program(p.text(), "cuda", sample=sample, outputs=[y],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=True)
    work = "/tmp/probe_attn"
    os.makedirs(work, exist_ok=True)
    exe = build_cu(art["source"], work, name="a1")
    pay = {"x0": E[N - 1:N].detach().cpu().numpy().astype(np.float16),
           "qr": qrow,
           "K": Kc, "V": Vc,
           "wo": g("model.layers.0.self_attn.o_proj.weight").T,
           "wup": g("model.layers.0.mlp.up_proj.weight").T,
           "wgate": g("model.layers.0.mlp.gate_proj.weight").T,
           "wdown": g("model.layers.0.mlp.down_proj.weight").T,
           "ln2": g("model.layers.0.post_attention_layernorm.weight")}
    am = np.zeros((PER, S), dtype=np.int64)
    am[:, :N] = 1
    pay["attmask7"] = am
    pay["allrows"] = np.arange(S, dtype=np.int64)
    argv = [exe]
    for name in art["inputs"]:
        k, _, _ = art["streams"][name]
        fn = os.path.join(work, f"in_{name}.bin")
        pay[name].astype(np.float16 if k == "F" else np.int64).tofile(fn)
        argv.append(fn)
    fo = os.path.join(work, "out.bin")
    argv.append(fo)
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:300]
    got = np.fromfile(fo, dtype=np.float32).reshape(1, HID)[0]
    print(f"attn-vs-torch maxabs: {np.abs(got - ref).max():.3e}", flush=True)


if __name__ == "__main__":
    main()
