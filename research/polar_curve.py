"""Polarization curve: install coexistence vs value angle (nonlinear path).

Two stores, SAME key (Italy HN key twice -> softmax splits exactly
50/50 by construction): values Rome-row 8x + companion(theta) 8x,
companion = cos t*romeU + sin t*u_perp. Through the layer-1 bank
(lm_bankhn.asm: attention+MLP downstream, NONLINEAR): does Rome rank
fall monotonically with angle (Malus-like EMERGENCE -- not exact
linearity, which can't produce angle dependence beyond the dot)?
t=0: constructive (2x Rome, predicts best). t=180: cancellation
(predicts base -- superpos control). t=90: orthogonal passage?
Monotonic falloff = polarization law for installs; flat = no angle
dependence (direction irrelevant, only dose).
Usage: python3 research/polar_curve.py (CPU lattice, fast)
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
THETAS = [0, 30, 60, 90, 120, 180]
DOSE = 8.0


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt, evb):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog),
                             "ukt": enc(ukt), "evb": enc(evb)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    iprompt = ids_of("The capital of Italy is")[-8:]
    fkey = run(iprompt, ukt0, evb0)
    hn = dec(fkey["HN"])[-1]
    key = (hn / np.linalg.norm(hn))[:, None]
    ukt2 = np.concatenate([key, key], axis=1)
    romeU = np.ascontiguousarray(wlog[:, 98])
    romeU /= np.linalg.norm(romeU)
    rng = np.random.default_rng(0)
    u = rng.normal(size=16)
    u -= (u @ romeU) * romeU
    u /= np.linalg.norm(u)
    f0 = run(iprompt, ukt0, evb0)
    lg0 = dec(f0["LOGITS"])[-1]
    print(f"base: Rome-rank={int((lg0 > lg0[98]).sum()) + 1} "
          f"top={inv.get(int(lg0.argmax()), '?')}", flush=True)
    for deg in THETAS:
        th = float(np.radians(deg))
        comp = float(np.cos(th)) * romeU + float(np.sin(th)) * u
        evb2 = np.concatenate([(DOSE * evn * romeU)[None, :],
                               (DOSE * evn * comp)[None, :]], axis=0)
        f2 = run(iprompt, ukt2, evb2)
        lg = dec(f2["LOGITS"])[-1]
        bp = dec(f2["BP1"])[-1]
        o = np.argsort(-lg)[:3]
        print(f"theta={deg:3}: rank={int((lg > lg[98]).sum()) + 1} "
              f"top3={[(inv.get(int(i), '?'), round(float(lg[int(i)]), 1)) for i in o]} "
              f"P={np.round(bp, 3).tolist()}", flush=True)


if __name__ == "__main__":
    main()
