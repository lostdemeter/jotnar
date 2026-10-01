"""Probe: parity at body gain 2.0 (one-shot, not a gate)."""
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
    d = np.load(os.path.join(ROOT, "data", "lm_svd.npz"))
    man = json.load(open(os.path.join(ROOT, "data", "lm_block1_manifest.json")))
    toks = np.array(man["toks"], dtype=np.int64)
    pos = np.arange(len(toks), dtype=np.int64)
    Sq = len(toks)
    cm = np.tril(np.ones((Sq, Sq), dtype=np.int64))
    g = 2.0
    w = {k: d[k] * g for k in
         ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown"]}
    text = ("CONFIG m_acc 35492\nCONFIG m_cov 35048\n"
            + open(os.path.join(ROOT, "programs", "lm_depth2causal.asm")).read())
    feeds = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                          "wdown": enc(w["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS,
                         basedir=os.path.join(ROOT, "programs"))
    dt = torch.float64
    E = torch.tensor(d["emb"][toks], dtype=dt)
    MASK = torch.tensor(np.triu(np.full((Sq, Sq), -1e9), 1), dtype=dt)

    def rms(x, ww, eps=1e-6):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * torch.tensor(
            np.ascontiguousarray(ww), dtype=dt)

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

    def layer(Hin):
        XN = rms(Hin, d["rms1"])
        Q = XN @ torch.tensor(w["wq"], dtype=dt)
        K = XN @ torch.tensor(w["wk"], dtype=dt)
        Vv = XN @ torch.tensor(w["wv"], dtype=dt)
        cs, scmax = [], 0.0
        for h in range(2):
            q = Q[:, h * 8:(h + 1) * 8]
            k = K[:, h * 8:(h + 1) * 8]
            v = Vv[:, h * 8:(h + 1) * 8]
            qr, kr = rope(q, pos), rope(k, pos)
            sc = qr @ kr.T + MASK
            scmax = max(scmax, float(sc.abs().max()))
            cs.append(torch.softmax(sc, dim=-1) @ v)
        CTX = torch.cat(cs, dim=1)
        O = CTX @ torch.tensor(w["wo"], dtype=dt)
        H = Hin + O
        HN = rms(H, d["rms2"])
        UP = HN @ torch.tensor(w["wup"], dtype=dt)
        GATE = HN @ torch.tensor(w["wgate"], dtype=dt)
        GS = GATE * torch.sigmoid(GATE)
        DOWN = (GS * UP) @ torch.tensor(w["wdown"], dtype=dt)
        return H + DOWN, scmax

    H2, s1 = layer(E)
    H4, s2 = layer(H2)
    LOG = H4 @ torch.tensor(d["wlog"], dtype=dt)
    sm = max(s1, s2)
    print(f"scoremax {sm:.3f} {'(in-contract)' if sm <= 1.0 else '(WIDE regime)'}")
    for name, ref, got in (("H4", H4.detach().numpy(), dec(feeds["H4"])),
                           ("LOGITS", LOG.detach().numpy(), dec(feeds["LOGITS"]))):
        mse = float(np.mean((got - ref) ** 2))
        db = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        print(name, round(db, 2))


if __name__ == "__main__":
    main()
