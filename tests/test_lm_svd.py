"""SVD weights gate: counts-injected transformer (construction paths A+B).

Frozen data/lm_svd.npz (log1p-bigram rank-16 SVD: emb=U√s spectrum 7.3x,
wlog=√sVᵀ; body seed-0 random; m_acc 35492 + m_cov 35048 per goldilocks).
Listing programs/lm_depth2causal.asm + CONFIG overrides (caller-supplied,
proven path). Gates: parity vs torch mirror (SVD factors, causal),
behavior beats random (top1>0.15, ppl<300 on 10-line sample; full 40-line
numbers in the commit note, not the gate).
Usage: python3 tests/test_lm_svd.py (slow: ~100 listing runs)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

BAR_DB = 40.0
FAIL = []
CFG = "CONFIG m_acc 35492\nCONFIG m_cov 35048\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    man = json.load(open(os.path.join(root, "data", "lm_svd_manifest.json")))
    check("svd-manifest", man["k"] == 16 and man["spec_ratio"] > 5.0,
          f"rank-16 spectral init (ratio {man['spec_ratio']}, "
          f"flat-random was 1.4x, real 17x)")
    d = np.load(os.path.join(root, "data", "lm_svd.npz"))
    vocab = json.load(open(os.path.join(root, "data", "lm_vocab.json")))
    text = CFG + open(os.path.join(sdir, "lm_depth2causal.asm")).read()
    toks = np.array(man_tokens(), dtype=np.int64)
    pos = np.arange(len(toks), dtype=np.int64)
    cm = np.tril(np.ones((len(toks), len(toks)), dtype=np.int64))
    feeds = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
    refH, refL = torch_ref(d, toks, pos)
    for name, key, ref in (("hidden", "H4", refH), ("logits", "LOGITS", refL)):
        got = dec(feeds[key])
        mse = float(np.mean((got - ref) ** 2))
        db = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        check(f"svd-{name}-parity", db >= BAR_DB, f"{db:.2f}dB")

    def logits_of(ids):
        ctx = ids[-8:]
        p = np.arange(len(ctx), dtype=np.int64)
        c = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": p, "cmask": c,
                          "emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
                          "wv": enc(d["wv"]), "wo": enc(d["wo"]),
                          "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
                          "wdown": enc(d["wdown"]),
                          "rms_w1": enc(d["rms1"]), "rms_w2": enc(d["rms2"]),
                          "wlog": enc(d["wlog"])}, sigs=SIGS, basedir=sdir)
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(root, "data", "lm_test.txt")).read().split("\n")[:10]
    nll, ntok, top1, tot = 0.0, 0, 0, 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            lg = logits_of(ids[max(0, k - 8):k])
            e = np.exp(lg - lg.max())
            p = e / e.sum()
            nll += -np.log(max(p[ids[k]], 1e-12))
            ntok += 1
            top1 += (int(np.argmax(lg)) == ids[k])
            tot += 1
    ppl = float(np.exp(nll / max(ntok, 1)))
    check("svd-beats-random-top1", top1 / tot > 0.15,
          f"top1={top1 / tot:.3f} (random 0.0, bigram 0.406)")
    check("svd-beats-random-ppl", ppl < 300,
          f"ppl={ppl:.1f} (random 512, bigram 47)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


def man_tokens():
    return json.load(open(os.path.join(
        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
        "data", "lm_block1_manifest.json")))["toks"]


def torch_ref(d, toks, pos):
    import torch
    dt = torch.float64
    Sq = len(toks)
    E = torch.tensor(d["emb"][toks], dtype=dt)
    MASK = torch.tensor(np.triu(np.full((Sq, Sq), -1e9), 1), dtype=dt)

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

    def layer(Hin):
        XN = rms(Hin, d["rms1"])
        Q = XN @ torch.tensor(d["wq"], dtype=dt)
        K = XN @ torch.tensor(d["wk"], dtype=dt)
        V = XN @ torch.tensor(d["wv"], dtype=dt)
        cs = []
        for h in range(2):
            q, k, v = Q[:, h * 8:(h + 1) * 8], K[:, h * 8:(h + 1) * 8], V[:, h * 8:(h + 1) * 8]
            qr, kr = rope(q, pos), rope(k, pos)
            sc = qr @ kr.T + MASK
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

    H2 = layer(E)
    H4 = layer(H2)
    LOG = H4 @ torch.tensor(d["wlog"], dtype=dt)
    return H4.detach().numpy(), LOG.detach().numpy()


if __name__ == "__main__":
    main()
