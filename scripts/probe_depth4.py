"""Probe: depth-4 tied parity (one-shot, not a gate)."""
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


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    import torch
    dt = torch.float64
    d = np.load(os.path.join(ROOT, "data", "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(ROOT, "data", "bankhn.npz"))
    ukt, evb = np.array(b["ukt"]), np.array(b["evb"])
    man = json.load(open(os.path.join(ROOT, "data", "lm_block1_manifest.json")))
    toks = np.array(man["toks"], dtype=np.int64)
    pos = np.arange(len(toks), dtype=np.int64)
    Sq = len(toks)
    cm = np.tril(np.ones((Sq, Sq), dtype=np.int64))
    text = ("CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
            + open(os.path.join(ROOT, "programs", "lm_depth4.asm")).read())
    pay = {"tok": toks, "pos": pos, "cmask": cm, "emb": enc(d["emb"]),
           "wq": enc(d["wq"]), "wk": enc(d["wk"]), "wv": enc(d["wv"]),
           "wo": enc(d["wo"]), "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
           "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
           "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
           "ukt": enc(ukt), "evb": enc(evb)}
    feeds = ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                         basedir=os.path.join(ROOT, "programs"))
    E = torch.tensor(np.ascontiguousarray(d["emb"][toks]), dtype=dt)
    MASK = torch.tensor(np.triu(np.full((Sq, Sq), -1e9), 1), dtype=dt)

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
        XN = rms(Hin, d["rms1"])
        Q = XN @ torch.tensor(np.ascontiguousarray(d["wq"]), dtype=dt)
        K = XN @ torch.tensor(np.ascontiguousarray(d["wk"]), dtype=dt)
        Vv = XN @ torch.tensor(np.ascontiguousarray(d["wv"]), dtype=dt)
        cs = []
        for h in range(2):
            q = Q[:, h * 8:(h + 1) * 8]
            k = K[:, h * 8:(h + 1) * 8]
            v = Vv[:, h * 8:(h + 1) * 8]
            qr, kr = rope(q, pos), rope(k, pos)
            sc = qr @ kr.T + MASK
            cs.append(torch.softmax(sc, dim=-1) @ v)
        CTX = torch.cat(cs, dim=1)
        O = CTX @ torch.tensor(np.ascontiguousarray(d["wo"]), dtype=dt)
        return Hin + O

    def bankmlp(H):
        HN = rms(H, d["rms2"])
        HNn = HN.detach().numpy()
        BP = np.exp(HNn @ ukt - (HNn @ ukt).max(-1, keepdims=True))
        BP = BP / BP.sum(-1, keepdims=True)
        return H + torch.tensor(np.ascontiguousarray(BP @ evb), dtype=dt)

    H = E
    for _ in range(4):
        H = bankmlp(attn(H))
    LOG = H @ torch.tensor(np.ascontiguousarray(d["wlog"]), dtype=dt)
    for name, ref, got in (("H8", H.detach().numpy(), dec(feeds["H8"])),
                           ("LOGITS", LOG.detach().numpy(), dec(feeds["LOGITS"]))):
        mse = float(np.mean((got - ref) ** 2))
        db = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        print(name, round(db, 2))


if __name__ == "__main__":
    main()
