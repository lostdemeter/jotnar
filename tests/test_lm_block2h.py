"""Two-head gate: two-head LM block parity vs torch (construction).

Listing programs/lm_block2h.asm (embeddings GATHER + RMSNorm + two-head
RoPE attention via TSHIFT+SOFTMAX_WIDE + SwiGLU MLP + unembedding) vs an
INDEPENDENT torch.float64 mirror (no phi-core). Frozen data/lm_block2h.npz
(seed 0, D=16/Dff=32/V=513, uniform 0.10). Bar 40dB peak=1.0.
Usage: python3 tests/test_lm_block2h.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

BAR_DB = 40.0
FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def torch_mirror(d, toks, pos):
    import torch
    dt = torch.float64
    E = torch.tensor(d["emb"][toks], dtype=dt)

    def rms(x, w, eps=1e-6):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * torch.tensor(
            np.ascontiguousarray(w), dtype=dt)

    def rope(x, p, base=10000.0):
        D = x.shape[-1]
        i = torch.arange(D // 2, dtype=dt)
        th = base ** (-2 * i / D)
        ang = torch.tensor(np.ascontiguousarray(p), dtype=dt).unsqueeze(1) * th.unsqueeze(0)
        c, s = torch.cos(ang), torch.sin(ang)
        y = torch.empty_like(x)
        y[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
        y[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
        return y

    XN = rms(E, d["rms1"])
    Q = XN @ torch.tensor(d["wq"], dtype=dt)
    K = XN @ torch.tensor(d["wk"], dtype=dt)
    V = XN @ torch.tensor(d["wv"], dtype=dt)

    def head(q, k, v, h):
        qq, kk, vv = q[:, h * 8:(h + 1) * 8], k[:, h * 8:(h + 1) * 8], v[:, h * 8:(h + 1) * 8]
        qr, kr = rope(qq, pos), rope(kk, pos)
        sc = qr @ kr.T
        return torch.softmax(sc, dim=-1) @ vv, float(sc.abs().max())

    C1, s1 = head(Q, K, V, 0)
    C2, s2 = head(Q, K, V, 1)
    CTX = torch.cat([C1, C2], dim=1)
    SCMAX = max(s1, s2)
    O = CTX @ torch.tensor(d["wo"], dtype=dt)
    H = E + O
    HN = rms(H, d["rms2"])
    UP = HN @ torch.tensor(d["wup"], dtype=dt)
    GATE = HN @ torch.tensor(d["wgate"], dtype=dt)
    GS = GATE * torch.sigmoid(GATE)
    DOWN = (GS * UP) @ torch.tensor(d["wdown"], dtype=dt)
    H2 = H + DOWN
    LOG = H2 @ torch.tensor(d["wlog"], dtype=dt)
    return H2.detach().numpy(), LOG.detach().numpy(), SCMAX


def run_listing(d, toks, pos):
    text = open(os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
                             "programs", "lm_block2h.asm")).read()
    return ASM.run_text(text, REGISTRY,
                        {"tok": toks, "pos": pos,
                         "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                         "wdown": enc(d["wdown"]),
                         "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                         "wlog": enc(d["wlog"])}, sigs=SIGS,
                        basedir=os.path.join(os.path.normpath(
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
                            "programs"))


def psnr(got, ref):
    mse = float(np.mean((got - ref) ** 2))
    return (float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)), mse


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    d = np.load(os.path.join(root, "data", "lm_block1.npz"))
    man = json.load(open(os.path.join(root, "data", "lm_block1_manifest.json")))
    toks = np.array(man["toks"], dtype=np.int64)
    pos = np.arange(len(toks), dtype=np.int64)
    refH, refL, scmax = torch_mirror(d, toks, pos)
    check("block1-scores-in-contract", scmax <= 1.0,
          f"scoremax={scmax:.3f} (proves barred row non-vacuous)")
    feeds = run_listing(d, toks, pos)
    dh, mseh = psnr(dec(feeds["H2"]), refH)
    check("block1-hidden-parity", dh >= BAR_DB,
          f"{dh:.2f}dB (bar {BAR_DB}, mse={mseh:.2e})")
    dl, msel = psnr(dec(feeds["LOGITS"]), refL)
    check("block1-logits-parity", dl >= BAR_DB,
          f"{dl:.2f}dB (bar {BAR_DB}, mse={msel:.2e})")
    feeds2 = run_listing(d, toks, pos)
    same = all(bool((feeds["OUT"][k] == feeds2["OUT"][k]).all()) for k in (0, 1, 2)) \
        if isinstance(feeds["OUT"], tuple) else bool((feeds["OUT"] == feeds2["OUT"]).all())
    check("block1-deterministic", same, "two runs identical ids")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
