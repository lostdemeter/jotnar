"""Qwen layer-0 full-block port gate (Task 2: GQA attention + MLP).

Frozen data/qwen0_block.npz (Q/K WITHOUT Q-bias per substrate
range law -- 23.3dB stated divergence, PINNED; V-bias augmented,
K-bias proof-cancelled) through programs/
qwen0_block.asm (14 explicit heads, kv groups h//7, no TSHIFT,
rope_base 1000000.0). Gates: manifest pin + full-L0 parity vs
no-qb torch mirror >= 40dB + divergence-cost pin on the n=8
prompt. Attention fits the existing WIDE contract on short
contexts (surveyed <=30, 17x margin -- the 963 regime needs
longer content, follow-up). SKIPs without HF cache.
Usage: python3 tests/test_qwen0_block.py (slow: full-block run)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

FAIL = []

QWEN = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-0.5B", "snapshots",
                    "91d2aff3f957f99e4c74c962f2f408dcc88a18d8")


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((np.ascontiguousarray(a, dtype=np.float64)
                         - np.ascontiguousarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    if not os.path.isfile(os.path.join(QWEN, "model.safetensors")):
        print("SKIP (needs Qwen2-0.5B in local HF cache)")
        sys.exit(0)
    try:
        from chain.qwen_mirror import load_layer0, build_H, mlp_forward
    except ImportError as e:
        print(f"SKIP (needs torch+safetensors+transformers: {e})")
        sys.exit(0)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    man = json.load(open(os.path.join(dd, "qwen0_block_manifest.json")))
    check("qwen0-manifest", man["rope_base"] == 1000000.0,
          f"frozen (sha {man['sha']})")
    g, embed, tok = load_layer0()
    prompt = "The capital of France is Paris, and the capital of Germany is"
    H1m, ids, _ = build_H(g, embed, tok, prompt)
    REFHF, _, _, _, _, _ = mlp_forward(H1m, g)
    E = embed(ids)
    # NOTE: build_H/mlp_forward carry Q-bias; the port omits it (range
    # law). Parity here is vs the no-qb mirror (inline, same recipe
    # minus qb); the 23.3dB HF divergence is recorded in the manifest.
    import torch as _t
    _dt = _t.float64
    _E = _t.tensor(E, dtype=_dt)
    _ln1 = _t.tensor(g("model.layers.0.input_layernorm.weight"), dtype=_dt)

    def _rms(x, w):
        return x / _t.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6) * w

    _xn = _rms(_E, _ln1)
    _Wq = _t.tensor(g("model.layers.0.self_attn.q_proj.weight"), dtype=_dt)
    _Wk = _t.tensor(g("model.layers.0.self_attn.k_proj.weight"), dtype=_dt)
    _Wv = _t.tensor(g("model.layers.0.self_attn.v_proj.weight"), dtype=_dt)
    _Wo = _t.tensor(g("model.layers.0.self_attn.o_proj.weight"), dtype=_dt)
    _bk = _t.tensor(g("model.layers.0.self_attn.k_proj.bias"), dtype=_dt)
    _bv = _t.tensor(g("model.layers.0.self_attn.v_proj.bias"), dtype=_dt)
    _n = len(ids)
    _Q = (_xn @ _Wq.T).reshape(_n, 14, 64)
    _K = (_xn @ _Wk.T + _bk).reshape(_n, 2, 64).repeat_interleave(7, dim=1)
    _V = (_xn @ _Wv.T + _bv).reshape(_n, 2, 64).repeat_interleave(7, dim=1)

    def _rope(x, base=1e6):
        _Sq, _Dh = x.shape[0], x.shape[-1]
        _i = _t.arange(_Dh // 2, dtype=_dt)
        _th = base ** (-2 * _i / _Dh)
        _ang = _t.arange(_Sq, dtype=_dt)[:, None] * _th[None, :]
        _c, _s = _t.cos(_ang), _t.sin(_ang)
        _y = _t.empty_like(x)
        _y[..., 0::2] = x[..., 0::2] * _c.unsqueeze(-2) - x[..., 1::2] * _s.unsqueeze(-2)
        _y[..., 1::2] = x[..., 0::2] * _s.unsqueeze(-2) + x[..., 1::2] * _c.unsqueeze(-2)
        return _y

    _QR, _KR = _rope(_Q.permute(1, 0, 2)), _rope(_K.permute(1, 0, 2))
    _SC = (_QR @ _KR.transpose(-1, -2)) / 8.0
    _P = _t.softmax(_SC + _t.triu(_t.full((_n, _n), -30.0, dtype=_dt), 1), dim=-1)
    _CTX = (_P @ _V.permute(1, 0, 2)).permute(1, 0, 2).reshape(_n, 896)
    _H1 = (_E + _CTX @ _Wo.T).numpy()
    REF, _, _, _, _, _ = mlp_forward(_H1, g)

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]),
                         np.ascontiguousarray(t[1])) * (
                             1 - np.ascontiguousarray(t[2]).astype(float)))

    z = np.load(os.path.join(dd, "qwen0_block.npz"))
    n = len(ids)
    ones = np.ones((n, 1))
    pay = {"H": enc(E), "pos": np.arange(n, dtype=np.int64),
           "cmask": np.tril(np.ones((n, n), dtype=np.int64)),
           "ones": enc(ones), "ln1": enc(z["ln1"]),
           "Wq8": enc(z["Wq8"]), "Wk8": enc(z["Wk8"]),
           "Wva": enc(z["Wva"]), "Wo": enc(z["Wo"]),
           "ln2": enc(z["ln2"]), "wup": enc(z["wup"]),
           "wgate": enc(z["wgate"]), "wdown": enc(z["wdown"])}
    text = open(os.path.join(sdir, "qwen0_block.asm")).read()
    got = dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                           basedir=sdir)["OUT"])
    d = psnr(got, REF)
    check("qwen0-block-parity", d >= 40.0,
          f"{d:.2f}dB full-L0 (no-qb) vs torch (peak=1.0, "
          f"outmax={np.abs(REF).max():.2f})")
    dhf = psnr(REF, REFHF)
    check("qwen0-qbias-cost", abs(dhf - 23.3) < 3.0,
          f"stated HF divergence {dhf:.1f}dB (pins the 23.3 cost)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
