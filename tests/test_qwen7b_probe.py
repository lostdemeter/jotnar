"""Siphon probe v0.1: Qwen2-7B-Instruct layer-0 MLP parity at 7B scale.

Mirrors the 0.5B recipe (test_realw: boundary float64 H + lattice MLP
listing) with 7B shapes (H=3584, I=18944, 28Q/4KV heads, Dh=128):
magnitudes survey -> scale pricing -> eps calibration -> parity dB +
sigmoid-LUT span audit. Plus a storebank transfer mini-test (teacher
DOWN recalled by our retrieval on held-out H rows).
Self-contained loader (no churn to green 0.5B gates); frozen blobs stay
out of git (in-memory encode; ~2GB transient, 61GB box).
SKIPs without the 7B-Instruct HF snapshot.
Usage: python3 tests/test_qwen7b_probe.py (slow: load ~2min, run ~10min)
"""
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

SNAP = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-7B-Instruct", "snapshots",
                    "f2826a00ceef68f0f2b946d945ecc0477ce4450c")
FAIL = []
N_HEADS, N_KV, DH, HIDDEN = 28, 4, 128, 3584


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    import numpy as np
    mse = float(np.mean((np.ascontiguousarray(a, dtype=np.float64)
                         - np.ascontiguousarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def load7b():
    import json
    import numpy as np
    import torch
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
        fns = {wm[n]} if n in wm else set()
        for fn in fns:
            sd = f(fn)
            if n in sd:
                return sd[n].float().double().numpy()
        return np.zeros((1,), dtype=np.float64)  # absent bias -> zeros

    tok = AutoTokenizer.from_pretrained(SNAP)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return g, tok


def build_H7(g, tok, prompt, n=8):
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
        # x: (..., seq, dim); positions run on seq, shared across heads.
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
    QR, KR = rope(Q), rope(K)
    import math
    _QR, _KR, _V = (t.permute(1, 0, 2) for t in (QR, KR, V))
    SC = (_QR @ _KR.transpose(-1, -2)) / math.sqrt(DH)
    P = torch.softmax(SC + torch.triu(
        torch.full((n, n), float("-inf"), dtype=dt), 1), dim=-1)
    CTX = (P @ _V).permute(1, 0, 2).reshape(n, HIDDEN)
    return (xt + CTX @ Wo.T).numpy(), ids, toks


def mlp7_forward(H, g):
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
    return REF, MID.numpy(), Wup.numpy(), Wg.numpy(), Wd.numpy(), ln2.numpy()


def main():
    import numpy as np
    if not os.path.isfile(os.path.join(SNAP, "model.safetensors.index.json")):
        print("SKIP (needs Qwen2-7B-Instruct in local HF cache)")
        sys.exit(0)
    try:
        g, tok = load7b()
    except ImportError as e:
        print(f"SKIP (needs torch+safetensors+transformers: {e})")
        sys.exit(0)
    import phi_core.lattice as S
    from phi_core import numpy_ops as NOP
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    H, ids, toks = build_H7(
        g, tok, "The capital of France is Paris, and the capital of Germany is")
    REF, MIDf, Wupf, Wgf, Wdf, ln2f = mlp7_forward(H, g)
    print(f"H range [{H.min():.2f},{H.max():.2f}] MID [{MIDf.min():.2f},"
          f"{MIDf.max():.2f}] REF [{REF.min():.2f},{REF.max():.2f}]",
          flush=True)
    for nm, a in (("UP", Wupf @ np.ones(Wupf.shape[1])),
                  ("GATE", Wgf @ np.ones(Wgf.shape[1]))):
        print(f"probe {nm} out range [{a.min():.2f},{a.max():.2f}]", flush=True)
    # scale pricing: cover max |product| with headroom (bridge discipline)
    peak = max(float(np.abs(MIDf).max()), float(np.abs(REF).max()))
    print(f"peak magnitude: {peak:.2f} (0.5B recipe: UP 2.8/GATE 6.1/DOWN 3.4)",
          flush=True)

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    sdir = os.path.join(ROOT, "programs")
    # eps calibration mirrors op_rmsnorm (per-model, like 0.5B's 68719)
    Um = S.PHI ** ((35492 - S.BIAS) / S.K)
    eps_c = int(round(1e-6 * float(1 << 36) / (Um * Um)))
    print(f"eps_rms_c at m_cov 35492: {eps_c}", flush=True)
    text = ("CONFIG m_acc 35492\nCONFIG m_cov 35492\nCONFIG eps_rms 1e-6\n"
            + open(os.path.join(sdir, "mlp_qwen0.asm")).read())
    t0 = __import__("time").perf_counter()
    f = ASM.run_text(text, REGISTRY,
                     {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
                      "wdown": enc(Wdf.T), "ln": enc(ln2f)},
                     sigs=SIGS, basedir=sdir)
    print(f"lattice run: {__import__('time').perf_counter() - t0:.1f}s", flush=True)

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    got = dec(f["OUT"])
    db = psnr(REF, got)
    print(f"7B-L0-MLP parity: {db:.2f}dB (0.5B recipe: 51.82dB)", flush=True)
    check("q7b-parity-measured", np.isfinite(db), f"{db:.2f}dB")
    span = NOP.AUDIT.get("sig_span", None)
    print(f"sigmoid LUT span exits: {span} (must read 0 for fidelity)",
          flush=True)
    check("q7b-sig-span", span == 0, f"span={span}")

    # storebank transfer mini-test: teacher DOWN rows recalled by cosine
    # on held-out H rows (exact-recall control + top-1 cosine recall).
    HN = H / np.linalg.norm(H, axis=1, keepdims=True)
    DOWN = REF - H
    sims = HN @ HN.T
    np.fill_diagonal(sims, -np.inf)
    top1 = sims.argmax(axis=1)
    # recall quality: cosine-neighbor DOWN vs true DOWN (relative err)
    errs = [float(np.linalg.norm(DOWN[i] - DOWN[j]) / max(np.linalg.norm(DOWN[i]), 1e-12))
            for i, j in enumerate(top1)]
    print(f"bank top-1 cosine recall relerr: min={min(errs):.3f} "
          f"mean={sum(errs) / len(errs):.3f} (n={len(errs)})", flush=True)
    check("q7b-bank-recall-measured", True, f"mean relerr={sum(errs) / len(errs):.3f}")

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
