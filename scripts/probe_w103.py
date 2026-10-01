"""Probe: w103 transfer parity + behavior (one-shot, not a gate)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DI = os.path.join(ROOT, "data_ingest")


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    import torch
    dt = torch.float64
    d = np.load(os.path.join(ROOT, "data", "lm_svd_IvoQ.npz"))
    E = np.load(os.path.join(DI, "w103_ends.npz"))
    b = np.load(os.path.join(DI, "w103_bank.npz"))
    B = {k: np.array(d[k]) for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown", "rms1", "rms2"]}
    ukt, evb = np.array(b["ukt"]), np.array(b["evb"])
    text = ("CONFIG m_acc 35492\nCONFIG m_cov 35048\n"
            + open(os.path.join(ROOT, "programs", "lm_bankhn2.asm")).read())
    toks = np.array([1, 223, 2], dtype=np.int64)
    pos = np.arange(3, dtype=np.int64)
    cm = np.tril(np.ones((3, 3), dtype=np.int64))
    pay = {"tok": toks, "pos": pos, "cmask": cm, "emb": enc(E["emb"]),
           "wq": enc(B["wq"]), "wk": enc(B["wk"]), "wv": enc(B["wv"]),
           "wo": enc(B["wo"]), "wup": enc(B["wup"]), "wgate": enc(B["wgate"]),
           "wdown": enc(B["wdown"]), "rms_w1": enc(B["rms1"]),
           "rms_w2": enc(B["rms2"]), "wlog": enc(E["wlog"]),
           "ukt": enc(ukt), "evb": enc(evb)}
    feeds = ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                         basedir=os.path.join(ROOT, "programs"))
    Ep = torch.tensor(np.ascontiguousarray(E["emb"][toks]), dtype=dt)
    MASK = torch.tensor(np.triu(np.full((3, 3), -1e9), 1), dtype=dt)

    def rms(x, ww, eps=1e-6):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * torch.tensor(
            np.ascontiguousarray(ww), dtype=dt)

    def rope(x, p, base=10000.0):
        Dd = x.shape[-1]
        i = torch.arange(Dd // 2, dtype=dt)
        th = base ** (-2 * i / Dd)
        ang = torch.tensor(np.ascontiguousarray(p), dtype=dt).unsqueeze(1) * th.unsqueeze(0)
        c, s = torch.cos(ang), torch.sin(ang)
        y = torch.empty_like(x)
        y[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
        y[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
        return y

    def attn(Hin):
        XN = rms(Hin, B["rms1"])
        Q = XN @ torch.tensor(np.ascontiguousarray(B["wq"]), dtype=dt)
        K = XN @ torch.tensor(np.ascontiguousarray(B["wk"]), dtype=dt)
        Vv = XN @ torch.tensor(np.ascontiguousarray(B["wv"]), dtype=dt)
        cs = []
        for h in range(2):
            q = Q[:, h * 8:(h + 1) * 8]
            k = K[:, h * 8:(h + 1) * 8]
            v = Vv[:, h * 8:(h + 1) * 8]
            qr, kr = rope(q, pos), rope(k, pos)
            sc = qr @ kr.T + MASK
            cs.append(torch.softmax(sc, dim=-1) @ v)
        CTX = torch.cat(cs, dim=1)
        O = CTX @ torch.tensor(np.ascontiguousarray(B["wo"]), dtype=dt)
        return Hin + O

    def bankmlp(H):
        HN = rms(H, B["rms2"])
        HNn = HN.detach().numpy()
        BP = np.exp(HNn @ ukt - (HNn @ ukt).max(-1, keepdims=True))
        BP = BP / BP.sum(-1, keepdims=True)
        return H + torch.tensor(np.ascontiguousarray(BP @ evb), dtype=dt)

    H2 = bankmlp(attn(Ep))
    H4 = bankmlp(attn(H2))
    LOG = H4 @ torch.tensor(np.ascontiguousarray(E["wlog"]), dtype=dt)
    for name, ref, got in (("H4", H4.detach().numpy(), dec(feeds["H4"])),
                           ("LOGITS", LOG.detach().numpy(), dec(feeds["LOGITS"]))):
        mse = float(np.mean((got - ref) ** 2))
        db = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        print(name, round(db, 2))


if __name__ == "__main__":
    main()
