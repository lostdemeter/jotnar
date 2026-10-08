"""Synthesis validation: solver vs ladders, then Caesar install.

(1) Validate qpsolve on the amid full-vocab case (maximin-direction
req was 39.85 with proof): QP optimum must satisfy all 512-row
constraints at req <= 39.85 (else the solver, not the theory, is
wrong -- fail loud with the gap).
(2) Synthesize the Caesar value (flat head, top blockers incl. egypt,
mass-adjusted) and INSTALL through lm_siphon (9-store bank, no negs:
the configuration where maximin won): predict rank-1 before running.
Gates: solver receipt (constraints + bound) + install FLIP.
Usage: python3 research/synth_test.py (CPU lattice, fast)
"""
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

from chain.synth import qpsolve, synth_value

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
    text = CFG + open(os.path.join(sdir, "lm_siphon.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run0(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        return ASM.run_text(text0, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt0[:, :1]),
                             "evb2": enc(np.zeros((1, 16)))},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    # (1) solver receipt on amid full-vocab (margin 0 baseline for comparability)
    actx = [0, 6, 0, 3, 137, 0, 28, 0]
    fa = run0(actx)
    lga = dec(fa["LOGITS2"])[-1]
    V = wlogU.shape[1]
    keep = [c for c in range(V) if c != 74]
    A = np.stack([wlogU[:, 74] - wlogU[:, c] for c in keep], axis=0)
    d = np.array([float(lga[c] - lga[74]) for c in keep])
    posm = d > 0
    v, req = qpsolve(A[posm], d[posm])
    Av = A[posm].dot(v)
    print(f"solver receipt: req={req:.2f} (maximin-dir was 39.85) "
          f"min-gain={float(Av.min()):.3f} min-def={float(d[posm].min()):.3f} "
          f"{'SATISFIES' if bool(np.all(Av + 1e-6 >= d[posm])) else 'VIOLATED'}",
          flush=True)
    assert req <= 39.85 + 1e-6, "solver worse than maximin-direction bound"

    # (2) Caesar synthesis + install (9-store bank, unity E-cosine program)
    cprompt = ids_of("Julius Caesar was assassinated in the city of")[-8:]
    f0 = run0(cprompt)
    h4 = dec(f0["H4"])[-1]
    lgf = h4 @ wlogU
    o = np.argsort(-lgf)[:8]
    comps = [int(i) for i in o if int(i) != 98]
    vS, reqS, info = synth_value(wlogU, lgf, 98, comps, margin=0.5, mass=0.6)
    print(f"caesar synth: req={reqS:.2f} ncomp={info['n_comp']} "
          f"maxdef={info['max_deficit']:.2f} (raw-row 12x norm="
          f"{12 * evn:.2f})", flush=True)

    epos = {"italy": 3, "alex": 0, "caesar": 1}
    eids = {"italy": 261, "alex": 12, "caesar": 40}
    order = ["italy", "alex", "caesar"]
    prompts = {"italy": ids_of("The capital of Italy is")[-8:],
               "alex": ids_of("Alexander the Great founded the city of")[-8:],
               "caesar": cprompt}
    emb = np.ascontiguousarray(dE["emb"])
    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    fkeys = {}
    for f in order:
        ids = prompts[f]
        h = dec(run0(ids)["HN"])[epos[f]]
        fkeys[f] = h / np.linalg.norm(h)
    bgHN, bgE = [], []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or 12 in ctx or 40 in ctx or len(ctx) < 3 \
                    or len(bgHN) >= 6:
                continue
            h = dec(run0(ctx)["HN"])[-1]
            bgHN.append(h / np.linalg.norm(h))
        if len(bgHN) >= 6:
            break
    for w in ["city", "son", "syria", "egypt", "her", "his"]:
        if w in vocab and len(bgE) < 6:
            e = np.ascontiguousarray(emb[vocab[w]])
            bgE.append(e / np.linalg.norm(e))
    KS = 2.0
    Uhn = np.concatenate([k[:, None] for k in bgHN]
                         + [fkeys[f][:, None] for f in order], axis=1) * KS
    Ue = np.concatenate([k[:, None] for k in bgE]
                        + [(np.ascontiguousarray(emb[eids[f]])
                            / np.linalg.norm(emb[eids[f]]))[:, None]
                           for f in order], axis=1) * KS
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    Vc = np.concatenate([np.zeros((6, 16)), (8.0 * evn * romeU)[None, :],
                         (16.0 * evn * alexU)[None, :], vS[None, :]], axis=0)

    def runS(ids, Uhn, Ue, Vc):
        toks = np.array(ids, dtype=np.int64)
        n = len(ids)
        pos = np.arange(n, dtype=np.int64)
        cm = np.tril(np.ones((n, n), dtype=np.int64))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog), "wlogU": enc(wlogU),
                             "Uhn": enc(Uhn), "Ue": enc(Ue), "Vc": enc(Vc),
                             "unity": enc(np.ones(16)),
                             "onesS1": enc(np.ones((n, 1))),
                             "ones1V": enc(np.ones((1, V))),
                             "onesSV": enc(np.ones((n, V))),
                             "ukt": enc(ukt0), "evb": enc(evb0)},
                            sigs=SIGS, basedir=sdir)

    for name, ids, t in (("italy", prompts["italy"], 98),
                         ("alex", prompts["alex"], 59),
                         ("caesar", cprompt, 98)):
        fi = runS(ids, Uhn, Ue, Vc)
        li = dec(fi["LOGITS2"])[-1]
        print(f"synth install {name}: rank={int((li > li[t]).sum()) + 1} "
              f"top={inv.get(int(li.argmax()), '?')} "
              f"{'FLIP' if int(li.argmax()) == t else ''}", flush=True)


if __name__ == "__main__":
    main()
