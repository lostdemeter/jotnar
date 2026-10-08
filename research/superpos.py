"""Superposition arithmetic: predict ranks before running.

Wave theory graduates iff it predicts. Linear post-L2 path (corr
1.000): deltas are EXACTLY routing-weighted value sums, so:
(1) Cancellation: same key, values +v/-v. Equal dose -> net zero
irrespective of routing split (silence by construction); 2:1 ->
predicted half-effect. Any deviation is interference to price.
(2) Decision boundary: same key, Rome vs Alexandria values over a
dose grid. Predicted winner per cell = argmax(base + a*(w_r.v)
+ b*(w_a.v)) with measured routing; measured map must match cell
for cell. Deviations locate where superposition breaks (saturation?
cross-terms?).
Program lm_bankhn2.asm (linear), Italy HNB key, unit-tier values.
Usage: python3 research/superpos.py (CPU lattice, fast)
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
    wlogU = np.ascontiguousarray(np.load(os.path.join(dd, "wlogU.npz"))["wlogU"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt2, evb2):
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
                             "wlog": enc(wlogU),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    iprompt = ids_of("The capital of Italy is")[-8:]
    f0 = run(iprompt, ukt0[:, :1], np.zeros((1, 16)))
    lg0 = dec(f0["LOGITS2"])[-1]
    hnb = dec(f0["HNB"])[-1]
    key = (hnb / np.linalg.norm(hnb))[:, None]
    print(f"base flat top={inv.get(int(lg0.argmax()), '?')} "
          f"rome-r={int((lg0 > lg0[98]).sum()) + 1} "
          f"alex-r={int((lg0 > lg0[59]).sum()) + 1}", flush=True)
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    U2 = np.concatenate([key, key], axis=1)

    # (1) cancellation
    for da, db in ((4, 4), (8, 4), (4, 0)):
        E2 = np.concatenate([(da * evn * romeU)[None, :],
                             (-db * evn * romeU)[None, :]], axis=0)
        f2 = run(iprompt, U2, E2)
        lg = dec(f2["LOGITS2"])[-1]
        bp = dec(f2["BP2"])[-1]
        pred = lg0 + (bp[0] * da - bp[1] * db) * evn * (romeU @ wlogU)
        pm = int(np.argmax(pred))
        print(f"cancel {da}:{-db}: top={inv.get(int(lg.argmax()), '?')} "
              f"pred-top={inv.get(pm, '?')} "
              f"P={np.round(bp, 3).tolist()} "
              f"{'EXACT' if int(lg.argmax()) == pm else 'DEVIATES'}", flush=True)

    # (2) decision boundary grid (doses in evn units)
    print("boundary grid (measured top / predicted top):", flush=True)
    mism = 0
    for da in (0, 2, 4, 8):
        row = []
        for db in (0, 2, 4, 8):
            E2 = np.concatenate([(da * evn * romeU)[None, :],
                                 (db * evn * alexU)[None, :]], axis=0)
            f2 = run(iprompt, U2, E2)
            lg = dec(f2["LOGITS2"])[-1]
            bp = dec(f2["BP2"])[-1]
            pred = lg0 + (bp[0] * da * evn) * (romeU @ wlogU) \
                + (bp[1] * db * evn) * (alexU @ wlogU)
            m, p = inv.get(int(lg.argmax()), '?'), inv.get(int(np.argmax(pred)), '?')
            mism += m != p
            row.append(f"{m}/{p}")
        print(f"  rome{da}x: " + " ".join(f"{c:>14}" for c in row), flush=True)
    print(f"boundary mismatches: {mism}/16 "
          f"{'(superposition predictive)' if mism == 0 else '(interference to price)'}",
          flush=True)


if __name__ == "__main__":
    main()
