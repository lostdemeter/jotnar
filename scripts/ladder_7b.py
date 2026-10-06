"""Parity ladder: geometric-vs-torch relative error at L = 1,2,4,8,16 layers.

Attributes the 28-layer parity number across depth: uniform quantum per
layer (linear) or a compounding hotspot (knee) decides whether fidelity
work is needed at all, and where. Full-7B width (H=3584), fp16 geo vs
torch-fp32 reference, S=8 prompt. SKIPs without snapshot/GPU/nvcc.
Usage: python3 scripts/ladder_7b.py [--layers 1,2,4,8,16]
"""
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

PROMPT = "The capital of France is Paris, and the capital of Germany is"
NTOK = 8


def torch_ref_nlayer(L):
    import torch
    from chain.qwen7b import SNAP, load7b
    g, tok = load7b()
    dt = torch.float32
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][:NTOK].numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device=device)
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1000000.0):
        Sq, Hh, D = x.shape
        i = torch.arange(D // 2, dtype=dt, device=device)
        th = base ** (-2.0 * i / D)
        ang = torch.arange(Sq, dtype=dt, device=device)[:, None] * th[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        c, s = c[:, None, :], s[:, None, :]
        y = torch.empty_like(x)
        y[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
        y[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
        return y

    W = {}
    x = E
    for L_ in range(L):
        p = f"model.layers.{L_}."
        w = {k: torch.tensor(g(p + k), dtype=dt, device=device)
             for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.up_proj.weight", "mlp.gate_proj.weight",
                       "mlp.down_proj.weight", "input_layernorm.weight",
                       "post_attention_layernorm.weight")}
        W[L_] = {k: v.cpu().numpy() for k, v in w.items()}
        bl = f"model.layers.{L_}.self_attn."
        bq = torch.tensor(g(bl + "q_proj.bias"), dtype=dt, device=device)
        bk = torch.tensor(g(bl + "k_proj.bias"), dtype=dt, device=device)
        bv = torch.tensor(g(bl + "v_proj.bias"), dtype=dt, device=device)
        W[L_].update({
            "self_attn.q_proj.bias": bq.cpu().numpy(),
            "self_attn.k_proj.bias": bk.cpu().numpy(),
            "self_attn.v_proj.bias": bv.cpu().numpy()})
        xn = rms(x, w["input_layernorm.weight"])
        Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(NTOK, 28, 128)
        K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(NTOK, 4, 128)
        V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(NTOK, 4, 128)
        K = K.repeat_interleave(7, dim=1)
        V = V.repeat_interleave(7, dim=1)
        _QR, _KR, _V = (t.permute(1, 0, 2) for t in (rope(Q), rope(K), V))
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((NTOK, NTOK), float("-inf"),
                                                  device=device), 1), dim=-1)
        x = x + (P @ _V).permute(1, 0, 2).reshape(NTOK, 3584) @ w["self_attn.o_proj.weight"].T
        hn = rms(x, w["post_attention_layernorm.weight"])
        down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                * (hn @ w["mlp.up_proj.weight"].T)
                ) @ w["mlp.down_proj.weight"].T
        x = x + down
    return x.detach().cpu().numpy(), ids, W


def geo_nlayer(L, ref, ids, W):
    from chain.builder import Prog, qwen_layer
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    from chain.qwen7b import load7b as _lb
    g, _tok = _lb()
    E = g("model.embed_tokens.weight")[np.array(ids)]
    p = Prog(f"qwen{L}L7b")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    in_names = ["x0", "pos", "cmask"]
    for L_ in range(L):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            in_names.append(f"{k}{L_}")
    p.inp(*in_names)
    y = "x0"
    for L_ in range(L):
        y = qwen_layer(p, "x0" if L_ == 0 else y,
                       f"wq{L_}", f"wk{L_}", f"wv{L_}", f"wo{L_}",
                       f"wup{L_}", f"wgate{L_}", f"wdown{L_}", f"ln1{L_}",
                       f"ln2{L_}", "pos", "cmask", pre=str(L_),
                       bq=f"bq{L_}", bk=f"bk{L_}", bv=f"bv{L_}")
    n = len(ids)
    keymap = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
              "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
              "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
              "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
              "ln2": "post_attention_layernorm.weight",
              "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
              "bv": "self_attn.v_proj.bias"}
    sample = {"x0": np.ascontiguousarray(E),
              "pos": np.arange(n, dtype=np.int64),
              "cmask": np.tril(np.ones((n, n), dtype=np.int64))}
    for L_ in range(L):
        for k, src in keymap.items():
            w = W[L_][src]
            wt = np.ascontiguousarray(w.T)
            if k in ("wq", "bq"):
                wt = wt / np.sqrt(128)
            if k in ("bq", "bk", "bv"):
                wt = np.ascontiguousarray(np.tile(wt, (n, 1)))
            sample[f"{k}{L_}"] = wt
    art = compile_program(p.text(), "cuda", sample=sample, outputs=[y],
                          basedir=os.path.join(ROOT, "programs"), use_fp16=True)
    work = f"/tmp/ladder{L}"
    os.makedirs(work, exist_ok=True)
    exe = build_cu(art["source"], work, name="ql")
    argv = [exe]
    for name in art["inputs"]:
        kk, _, _ = art["streams"][name]
        fn = os.path.join(work, f"in_{name}.bin")
        if kk == "F":
            sample[name].astype(np.float16).tofile(fn)
        else:
            sample[name].astype(np.int64).tofile(fn)
        argv.append(fn)
    fo = os.path.join(work, "out.bin")
    argv.append(fo)
    r = subprocess.run(argv, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:300]
    got = np.fromfile(fo, dtype=np.float32).reshape(ref.shape)
    dmax = float(np.abs(got - ref).max())
    return dmax / max(float(np.abs(ref).max()), 1e-12)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,2,4,8,16")
    args = ap.parse_args()
    from chain.qwen7b import snapshot_ok
    if not snapshot_ok():
        print("SKIP (needs snapshot)")
        sys.exit(0)
    for Ls in [int(x) for x in args.layers.split(",")]:
        t0 = time.perf_counter()
        ref, ids, W = torch_ref_nlayer(Ls)
        rel = geo_nlayer(Ls, ref, ids, W)
        print(f"L={Ls:3d} rel={rel:.3e} ({time.perf_counter() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
