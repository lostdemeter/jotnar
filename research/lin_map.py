"""Linearize the layer-2 map at the Italy operating point, then aim through it.

Probes: same ks=32 e261 key (own-weight ~1 on the Italy prompt), values
= eps * basis_i (16 runs) + zero baseline. M[c,i] = dlogit_c / eps.
Checks: (a) M @ rome8 predicts the measured 8x-Rome delta (linearity);
(b) maximin value direction over the 7 blockers through M + its dose;
(c) install it: rank + holds. Falsifier: (a) fails -> layer-2 is
essentially nonlinear here, stop; (c) stalls -> value-geometry ceiling.
Usage: python3 research/lin_map.py
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain.engram import yarnball_bank

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
EPS = 0.5
KEY_SCALE = 32.0


def maximin(A, seeds=6, iters=4000):
    rng = np.random.default_rng(1)
    best_t, best_v = -1e18, None
    for _ in range(seeds):
        v = rng.normal(size=A.shape[1])
        v /= np.linalg.norm(v)
        step = 0.2
        for _it in range(iters):
            s = A.dot(v)
            j = int(np.argmin(s))
            g = A[j] - (A[j].dot(v)) * v
            n = np.linalg.norm(g)
            if n > 1e-12:
                v = v + step * g / n
                v /= np.linalg.norm(v)
            step *= 0.9995
        t = float(np.min(A.dot(v)))
        if t > best_t:
            best_t, best_v = t, v.copy()
    return best_v, best_t


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
    K0 = ukt0.shape[1]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    text = CFG + open(os.path.join(sdir, "lm_yarnball.asm")).read()

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

    def Pkey(f):
        c = [k for k in f if "yarnball_apply" in k and k.endswith(".P")]
        assert len(c) == 1, c
        return c[0]

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    e261 = np.ascontiguousarray(dE["emb"][261]) / np.linalg.norm(dE["emb"][261])
    iprompt = ids_of("The capital of Italy is")[-8:]

    # NOTE: yarnball_bank normalizes value to unit then scales by dose;
    # for raw probe vectors pass dose=||v|| with value=v (keeps v exact).
    def bank_raw(v):
        n = float(np.linalg.norm(v))
        return yarnball_bank(ukt0, evb0, [{"key": e261, "value": v,
                                           "dose": n, "tier": "opt",
                                           "support": "linmap-probe"}],
                             key_scale=KEY_SCALE)

    ku = (e261 / np.linalg.norm(e261) * KEY_SCALE)[:, None]
    Ua0 = np.concatenate([ukt0, ku], axis=1)
    Vc0 = np.concatenate([evb0, np.zeros((1, 16))], axis=0)
    f0 = run(iprompt, Ua0, Vc0)
    lg0 = dec(f0["LOGITS"])[-1]
    w0 = float(dec(f0[Pkey(f0)])[-1][K0])
    print(f"baseline: own-weight={w0:.3f} (need ~1)", flush=True)

    M = np.zeros((wlog.shape[1], 16))
    for i in range(16):
        v = np.zeros(16)
        v[i] = EPS
        Ua, Vc, _ = bank_raw(v)
        fi = run(iprompt, Ua, Vc)
        M[:, i] = (dec(fi["LOGITS"])[-1] - lg0) / EPS
    print(f"M mapped: |M|_fro={float(np.linalg.norm(M)):.3f} "
          f"vs |wlog|_fro={float(np.linalg.norm(wlog)):.3f}", flush=True)

    # (a) linearity check: predict the 8x-Rome delta
    romeU = np.ascontiguousarray(wlog[:, 98]) / np.linalg.norm(wlog[:, 98])
    v8 = romeU * (8.0 * evn)
    Ua, Vc, _ = bank_raw(v8)
    fact = dec(run(iprompt, Ua, Vc)["LOGITS"])[-1] - lg0
    pred = M.dot(v8)
    print(f"linearity corr={float(np.corrcoef(fact, pred)[0, 1]):.3f} "
          f"rms-fact={float(np.sqrt((fact ** 2).mean())):.3f} "
          f"rms-pred={float(np.sqrt((pred ** 2).mean())):.3f}", flush=True)

    o = np.argsort(-(lg0 + fact))
    top = [int(i) for i in o[:9] if int(i) != 98][:7]
    print("blockers=" + str([(inv.get(c, "?"), round(float(lg0[c] + fact[c]), 2))
                             for c in top]), flush=True)
    A = np.stack([M[98] - M[c] for c in top], axis=0)
    deficits = np.array([float((lg0[c] + fact[c]) - (lg0[98] + fact[98]))
                         for c in top])
    vstar, t = maximin(A)
    print(f"maximin t/unit={t:.4f} deficits={np.round(deficits, 2)}", flush=True)
    Av = A.dot(vstar)
    if np.any(Av <= 1e-9):
        print("INFEASIBLE: some blocker un-beatable through M", flush=True)
        return
    req = float(np.max(deficits / Av))
    print(f"required ||v||={req:.2f} (8x-Rome norm={float(np.linalg.norm(v8)):.2f})",
          flush=True)
    np.savez("/tmp/linmap.npz", M=M, vstar=vstar, req=np.array(req),
             base8=lg0 + fact)

    # (c) install vstar at req norm
    Ua, Vc, _ = bank_raw(vstar * req)
    fi = run(iprompt, Ua, Vc)
    li = dec(fi["LOGITS"])[-1]
    r = int((li > li[98]).sum()) + 1
    oo = np.argsort(-li)
    print(f"vstar install: rank={r} top5="
          + str([(inv.get(int(i), "?"), round(float(li[int(i)]), 2))
                 for i in oo[:5]]), flush=True)
    cprompts = []
    for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    hold = sum(1 for ids, tgt in cprompts
               if int(dec(run(ids, Ua, Vc)["LOGITS"])[-1].argmax())
               == int(dec(run(ids, ukt0, evb0)["LOGITS"])[-1].argmax()))
    print(f"holds {hold}/16", flush=True)


if __name__ == "__main__":
    main()
