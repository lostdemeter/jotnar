"""Depth driver gate: builder-generated 2-layer Qwen2-7B block on CUDA.

qwen_layer x2 (L0+L1, GQA-28/4 + SwiGLU + residuals) at full 7B width
(H=3584, I=18944, Dh=128) in FP16 storage, vs a torch-float32 reference.
Proves the depth driver (builder unrolling) and the width story at the
scale that matters; 28 layers is more of the same (stated, unblocked).
SKIPs without the 7B-Instruct snapshot.
Usage: python3 tests/test_qwen7b_depth.py (slow: ~1GB weights, ~10min)
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def torch_ref_2layer(prompt="The capital of France is Paris, and the capital of Germany is"):
    import torch
    from chain.qwen7b import SNAP, load7b
    g, tok = load7b()
    dt = torch.float32
    ids = tok(prompt, return_tensors="pt")["input_ids"][0][:8].numpy()
    assert len(ids) == 8, f"need 8 toks, got {len(ids)}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    E = torch.tensor(g("model.embed_tokens.weight")[ids], dtype=dt,
                     device=device)
    eps = 1e-6

    def rms(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, base=1000000.0):
        # x: (seq, heads, dim); positions shared across heads.
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

    x = E
    W = {}
    for L in (0, 1):
        p = f"model.layers.{L}."
        w = {k: torch.tensor(g(p + k), dtype=dt, device=device)
             for k in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.up_proj.weight", "mlp.gate_proj.weight",
                       "mlp.down_proj.weight", "input_layernorm.weight",
                       "post_attention_layernorm.weight")}
        W[L] = {k: v.cpu().numpy() for k, v in w.items()}
        bl = f"model.layers.{L}.self_attn."
        bq = torch.tensor(g(bl + "q_proj.bias"), dtype=dt, device=device)
        bk = torch.tensor(g(bl + "k_proj.bias"), dtype=dt, device=device)
        bv = torch.tensor(g(bl + "v_proj.bias"), dtype=dt, device=device)
        W[L].update({
            "self_attn.q_proj.bias": bq.cpu().numpy(),
            "self_attn.k_proj.bias": bk.cpu().numpy(),
            "self_attn.v_proj.bias": bv.cpu().numpy()})
        xn = rms(x, w["input_layernorm.weight"])
        Q = (xn @ w["self_attn.q_proj.weight"].T + bq).reshape(8, 28, 128)
        K = (xn @ w["self_attn.k_proj.weight"].T + bk).reshape(8, 4, 128)
        V = (xn @ w["self_attn.v_proj.weight"].T + bv).reshape(8, 4, 128)
        K = K.repeat_interleave(7, dim=1)
        V = V.repeat_interleave(7, dim=1)
        _QR, _KR, _V = (t.permute(1, 0, 2) for t in (rope(Q), rope(K), V))
        P = torch.softmax(_QR @ _KR.transpose(-1, -2) / np.sqrt(128)
                          + torch.triu(torch.full((8, 8), float("-inf"),
                                                  device=device), 1), dim=-1)
        x = x + (P @ _V).permute(1, 0, 2).reshape(8, 3584) @ w["self_attn.o_proj.weight"].T
        hn = rms(x, w["post_attention_layernorm.weight"])
        down = (torch.nn.functional.silu(hn @ w["mlp.gate_proj.weight"].T)
                * (hn @ w["mlp.up_proj.weight"].T)
                ) @ w["mlp.down_proj.weight"].T
        x = x + down
    return x.detach().cpu().numpy(), ids, W


def main():
    from chain.qwen7b import SNAP
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct in local HF cache)")
        sys.exit(0)
    try:
        ref, ids, W = torch_ref_2layer()
    except ImportError as e:
        print(f"SKIP (needs torch stack: {e})")
        sys.exit(0)
    print(f"torch ref range [{ref.min():.2f},{ref.max():.2f}]", flush=True)
    from chain.builder import Prog, qwen_layer
    from chain.emit_c import compile_program
    from chain.emit_cuda import build_cu
    import subprocess
    p = Prog("qwen2L7b")
    p.config("eps_rms", "1e-6").config("beta", -10000.0)
    p.config("rope_base", 1000000.0)
    in_names = ["x0", "pos", "cmask"]
    for L in (0, 1):
        for k in ("wq", "wk", "wv", "wo", "wup", "wgate", "wdown",
                  "ln1", "ln2", "bq", "bk", "bv"):
            in_names.append(f"{k}{L}")
    p.inp(*in_names)
    y0 = qwen_layer(p, "x0", "wq0", "wk0", "wv0", "wo0", "wup0", "wgate0",
                    "wdown0", "ln10", "ln20", "pos", "cmask", pre="0",
                    bq="bq0", bk="bk0", bv="bv0")
    y1 = qwen_layer(p, y0, "wq1", "wk1", "wv1", "wo1", "wup1", "wgate1",
                    "wdown1", "ln11", "ln21", "pos", "cmask", pre="1",
                    bq="bq1", bk="bk1", bv="bv1")
    n = len(ids)
    sample, dump = {}, {}
    keymap = {"wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
              "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
              "wup": "mlp.up_proj.weight", "wgate": "mlp.gate_proj.weight",
              "wdown": "mlp.down_proj.weight", "ln1": "input_layernorm.weight",
              "ln2": "post_attention_layernorm.weight",
              "bq": "self_attn.q_proj.bias", "bk": "self_attn.k_proj.bias",
              "bv": "self_attn.v_proj.bias"}
    # x0 = embeddings
    import torch as _t
    from chain.qwen7b import load7b as _lb
    g, _tok = _lb()
    E = g("model.embed_tokens.weight")[np.array(ids)]
    sample["x0"] = np.ascontiguousarray(E)
    dump["x0"] = sample["x0"]
    sample["pos"] = np.arange(n, dtype=np.int64)
    sample["cmask"] = np.tril(np.ones((n, n), dtype=np.int64))
    dump["pos"] = sample["pos"]
    dump["cmask"] = sample["cmask"]
    for L in (0, 1):
        for k, src in keymap.items():
            w = W[L][src]
            wt = np.ascontiguousarray(w.T)  # listing convention (K-major)
            if k in ("wq", "bq"):
                # temperature folded into Q offline (exact by linearity of
                # RoPE+matmul in Q): weight AND bias scale together --
                # (X@W + b)/s == X@(W/s) + b/s. Forgetting bq cost 0.5
                # relative (measured); the fold is exact.
                wt = wt / np.sqrt(128)
            if k in ("bq", "bk", "bv"):
                # bias planes: tiled host-side (broadcast-as-input)
                wt = np.ascontiguousarray(np.tile(wt, (n, 1)))
            sample[f"{k}{L}"] = wt
            dump[f"{k}{L}"] = wt
    use_fp16 = os.environ.get("QWEN_FP16", "1") == "1"
    art = compile_program(p.text(), "cuda", sample=sample, outputs=[y1],
                          basedir=os.path.join(ROOT, "programs"),
                          use_fp16=use_fp16)
    work = "/tmp/qwen7b_depth"
    os.makedirs(work, exist_ok=True)
    exe = build_cu(art["source"], work, name="q2l")
    argv = [exe]
    vram = 0
    for name in art["inputs"]:
        kk, _, _ = art["streams"][name]
        fn = os.path.join(work, f"in_{name}.bin")
        if kk == "F":
            dump[name].astype(np.float16 if use_fp16 else np.float32).tofile(fn)
            vram += dump[name].size * (2 if use_fp16 else 4)
        else:
            dump[name].astype(np.int64).tofile(fn)
            vram += dump[name].size * 8
        argv.append(fn)
    fo = os.path.join(work, "out.bin")
    argv.append(fo)
    import time
    t0 = time.perf_counter()
    r = subprocess.run(argv, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    check("q7b-depth-run", r.returncode == 0,
          f"rc={r.returncode} {r.stderr[:300]} wall={dt:.0f}s")
    if r.returncode != 0:
        print("FAILURES:", FAIL)
        sys.exit(1)
    got = np.fromfile(fo, dtype=np.float32).reshape(ref.shape)
    dmax = float(np.abs(got - ref).max())
    rel = dmax / max(float(np.abs(ref).max()), 1e-12)
    print(f"q7b-depth: maxabs={dmax:.3e} rel={rel:.3e} VRAM-inputs={vram / 1e9:.2f}GB",
          flush=True)
    bar = 1e-3 if use_fp16 else 5e-3
    check("q7b-depth-eps", np.isfinite(dmax) and rel < bar,
          f"rel={rel:.3e} ({'fp16 weight-quantum' if use_fp16 else 'fp32 structural'} class)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
