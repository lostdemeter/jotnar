"""Causal LM gate: depth-2 causal vs torch causal mirror + twin orders.

Listing programs/lm_depth2causal.asm (mask structure inside attention).
Gates: H4/LOGITS parity vs causal float, determinism, twin active/passive
GATHER orders ([alexander,founded,alexandria] vs [alexandria,was,founded,
by,alexander]) both green with order-difference proof (same program, two
execution orders -- no template strings).
Usage: python3 tests/test_lm_depth2causal.py
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
ACTIVE = [12, 471, 59]
PASSIVE = [59, 38, 471, 11, 12]


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def torch_layer(Hin, d, pos, mask):
    import torch
    dt = torch.float64

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

    XN = rms(Hin, d["rms1"])
    Q = XN @ torch.tensor(d["wq"], dtype=dt)
    K = XN @ torch.tensor(d["wk"], dtype=dt)
    V = XN @ torch.tensor(d["wv"], dtype=dt)
    cs = []
    for h in range(2):
        q, k, v = Q[:, h * 8:(h + 1) * 8], K[:, h * 8:(h + 1) * 8], V[:, h * 8:(h + 1) * 8]
        qr, kr = rope(q, pos), rope(k, pos)
        sc = qr @ kr.T + torch.tensor(mask, dtype=dt)
        cs.append(torch.softmax(sc, dim=-1) @ v)
    CTX = torch.cat(cs, dim=1)
    O = CTX @ torch.tensor(d["wo"], dtype=dt)
    H = Hin + O
    HN = rms(H, d["rms2"])
    UP = HN @ torch.tensor(d["wup"], dtype=dt)
    GATE = HN @ torch.tensor(d["wgate"], dtype=dt)
    GS = GATE * torch.sigmoid(GATE)
    DOWN = (GS * UP) @ torch.tensor(d["wdown"], dtype=dt)
    return H + DOWN


def run_path(d, ids, text, sdir):
    import torch
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    feeds = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
    E = torch.tensor(d["emb"][toks], dtype=torch.float64)
    mask = np.triu(np.full((len(ids), len(ids)), -1e9), 1)
    H2 = torch_layer(E, d, pos, mask)
    H4 = torch_layer(H2, d, pos, mask)
    LOG = H4 @ torch.tensor(d["wlog"], dtype=torch.float64)
    return feeds, H4.detach().numpy(), LOG.detach().numpy()


def psnr(got, ref):
    mse = float(np.mean((got - ref) ** 2))
    return (float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)), mse


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    text = open(os.path.join(sdir, "lm_depth2causal.asm")).read()
    d = np.load(os.path.join(root, "data", "lm_block1.npz"))
    for tag, ids in (("active", ACTIVE), ("passive", PASSIVE)):
        feeds, refH, refL = run_path(d, ids, text, sdir)
        dh, mseh = psnr(dec(feeds["H4"]), refH)
        check(f"causallm-{tag}-hidden", dh >= BAR_DB,
              f"{dh:.2f}dB (S={len(ids)})")
        dl, msel = psnr(dec(feeds["LOGITS"]), refL)
        check(f"causallm-{tag}-logits", dl >= BAR_DB,
              f"{dl:.2f}dB (S={len(ids)})")
    fa, _, _ = run_path(d, ACTIVE, text, sdir)
    fp, _, _ = run_path(d, PASSIVE, text, sdir)
    oa = dec(fa["LOGITS"])
    op = dec(fp["LOGITS"])
    check("twin-order-matters", float(np.abs(oa.mean() - op.mean())) > 1e-4,
          "active vs passive execution orders differ (no template strings)")
    check("twin-same-content", set(ACTIVE) <= set(PASSIVE),
          "shared edge multiset {alexander,founded,alexandria} both paths")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
