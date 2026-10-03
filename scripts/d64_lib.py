"""D64 piece refit battery harness (mission tooling, NOT a gate). Part 1: lib."""

import glob
import html
import json
import math
import os
import re
import sys
import time
from html.parser import HTMLParser

import numpy as np

sys.path.insert(0, '/home/thorin/Documents/OpenCode/phi-core')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
SD = os.path.join(ROOT, "programs")
WIN = 16


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def load_bpe():
    vocab = json.load(open(os.path.join(DD, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(DD, "bpe_merges.json")))
    return vocab, {tuple(m): i for i, m in enumerate(merges)}


def encode(s, vocab, rank):
    out = []
    for w in re.findall(r"[a-z0-9']+", s.lower()):
        syms = [c for c in w] + ["</w>"]
        while len(syms) > 1:
            best = None
            for i in range(len(syms) - 1):
                r = rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best[0]):
                    best = (r, i)
            if best is None:
                break
            _, i = best
            syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
        out.extend(vocab[p] for p in syms)
    return out


class _T(HTMLParser):
    def __init__(self):
        super().__init__()
        self.p = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        self.skip = tag in ("script", "style", "nav", "header", "footer",
                            "aside")

    def handle_endtag(self, tag):
        self.skip = False

    def handle_data(self, d):
        if not self.skip:
            self.p.append(d)


def load_probe():
    sents = []
    for f in sorted(glob.glob("/home/thorin/Documents/OpenCode/"
                               "Echion_Revisted/data/grokipedia/*.html")):
        t = _T()
        t.feed(open(f, encoding="utf-8", errors="replace").read())
        txt = html.unescape(" ".join(t.p))
        sents += [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt)
                  if len(s.strip().split()) >= 4]
    rng = np.random.default_rng(0)
    idx = np.arange(len(sents))
    rng.shuffle(idx)
    return [sents[i] for i in idx[int(0.8 * len(sents)):][:10]]


def bodies():
    seed = dict(np.load(os.path.join(DD, "lm_d64.npz")))
    G2 = {k: (2 * seed[k] if k in
               ["wq", "wk", "wv", "wo", "wup", "wgate", "wdown"]
               else seed[k]) for k in seed}
    WF = dict(G2)
    WF["wq"] = np.eye(64) * 0.2
    WF["wk"] = np.eye(64) * 0.2
    return seed, G2, WF


def run_ids(ids, E_, W_, b_, text):
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
    f = ASM.run_text(text, REGISTRY,
                     {"tok": toks, "pos": pos, "cmask": cm,
                      "emb": enc(E_["emb"]), "wq": enc(W_["wq"]),
                      "wk": enc(W_["wk"]), "wv": enc(W_["wv"]),
                      "wo": enc(W_["wo"]), "wup": enc(W_["wup"]),
                      "wgate": enc(W_["wgate"]), "wdown": enc(W_["wdown"]),
                      "rms_w1": enc(W_["rms1"]), "rms_w2": enc(W_["rms2"]),
                      "wlog": enc(E_["wlog"]), "ukt": enc(b_["ukt"]),
                      "evb": enc(b_["evb"])}, sigs=SIGS, basedir=SD)
    return f


def _enc_pair(vocab, rank):
    a = encode("alexander founded alexandria", vocab, rank)
    p = encode("alexandria was founded by alexander", vocab, rank)
    return a, p


def twin_num(E_, W_, b_, text, vocab, rank):
    a, p = _enc_pair(vocab, rank)
    fa = dec(run_ids(a, E_, W_, b_, text)["H4"])
    fp = dec(run_ids(p, E_, W_, b_, text)["H4"])
    return float(np.linalg.norm(fa[-1] - fp[-1])
                 / max(np.linalg.norm(fa[-1]), 1e-12))


def rms_t(x, w, eps=1e-6):
    import torch
    dt = torch.float64
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) \
        * torch.tensor(np.ascontiguousarray(w), dtype=dt)


def rope_t(x, p, base=10000.0):
    import torch
    dt = torch.float64
    D = x.shape[-1]
    i = torch.arange(D // 2, dtype=dt)
    th = base ** (-2 * i / D)
    pp = torch.tensor(np.ascontiguousarray(p), dtype=dt)
    ang = pp.unsqueeze(1) * th.unsqueeze(0)
    c, s = torch.cos(ang), torch.sin(ang)
    y = torch.empty_like(x)
    y[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    y[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return y


def mirror(E_, W_, b_, ids, beta_b=1.0, temper_L2=False, Dh=32):
    import torch
    dt = torch.float64
    toks = np.array(ids, dtype=np.int64)
    pos = np.arange(len(ids), dtype=np.int64)
    Sq = len(ids)
    E = torch.tensor(np.ascontiguousarray(E_["emb"][toks]), dtype=dt)
    MASK = torch.tensor(np.triu(np.full((Sq, Sq), -1e9), 1), dtype=dt)
    Wq = torch.tensor(np.ascontiguousarray(W_["wq"]), dtype=dt)
    Wk = torch.tensor(np.ascontiguousarray(W_["wk"]), dtype=dt)
    Wv = torch.tensor(np.ascontiguousarray(W_["wv"]), dtype=dt)
    Wo = torch.tensor(np.ascontiguousarray(W_["wo"]), dtype=dt)
    ukt = torch.tensor(np.ascontiguousarray(b_["ukt"]), dtype=dt)
    evb = torch.tensor(np.ascontiguousarray(b_["evb"]), dtype=dt)
    Wl = torch.tensor(np.ascontiguousarray(E_["wlog"]), dtype=dt)

    def attn(Hin, temper2=1.0):
        XN = rms_t(Hin, W_["rms1"])
        Q, K, Vv = XN @ Wq, XN @ Wk, XN @ Wv
        cs = []
        for h in range(2):
            q, k, v = (Q[:, h * Dh:(h + 1) * Dh],
                       K[:, h * Dh:(h + 1) * Dh],
                       Vv[:, h * Dh:(h + 1) * Dh])
            sc = rope_t(q, pos) @ rope_t(k, pos).T + MASK
            if h == 1:
                sc = sc * temper2
            cs.append(torch.softmax(sc, dim=-1) @ v)
        return Hin + torch.cat(cs, dim=1) @ Wo

    def bank(Hin):
        HN = rms_t(Hin, W_["rms2"])
        HNn = HN.detach().numpy()
        Z = HNn @ np.ascontiguousarray(b_["ukt"])
        BP = np.exp(Z - Z.max(-1, keepdims=True))
        BP = BP / BP.sum(-1, keepdims=True)
        return Hin + torch.tensor(np.ascontiguousarray(BP @
                                  np.ascontiguousarray(b_["evb"])),
                                  dtype=dt)

    H2 = bank(attn(E, beta_b))
    H4 = bank(attn(H2, 1.0 if not temper_L2 else beta_b))
    return H4, H4 @ Wl, ukt, evb


def par_db(E_, W_, b_, ids, key="H4", listing="lm_d64.asm", beta_b=1.0,
           macc=36118, mcov=35048):
    f = run_ids(ids, E_, W_, b_,
                "CONFIG m_acc %d\nCONFIG m_cov %d\nCONFIG beta -30.0\n"
                "CONFIG beta_b %s\n" % (macc, mcov, beta_b)
                + open(os.path.join(SD, listing)).read())
    temper = beta_b if "headt" in listing else 1.0
    H4t, LOGt, _, _ = mirror(E_, W_, b_, ids, beta_b=temper)
    ref = {"H4": H4t.detach().numpy(), "LOGITS": LOGt.detach().numpy()}[key]
    got = dec(f[key])
    mse = float(np.mean((got - ref) ** 2))
    db = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    return db, float(np.abs(ref).max()), float(np.abs(got).max())
