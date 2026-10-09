"""Negative-mined background: append false-positive keys as nulls.

blend_verify.py located two flip species: blend-attenuation razor
(doctrine) and false-positive retrieval (w=1.0 on generic contexts --
addressable). This closes the loop: run the prior split with the 7-
store bank, collect flipped UNK contexts, mine their HNB rows, append
as null stores AFTER Italy (index 6 baked in the program stays valid),
re-gate everything. Predicts UNK recovery with install intact;
receipt (Italy w>0.9) is the tripwire -- more competitors can only
hurt retrieval, never help it.
Usage: python3 research/negmine.py (CPU lattice, ~450 runs)
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
MARGIN_BAR = 1.0
NBG = 6
KS = 32.0
DOSE = 4.0
MAXNEG = 12


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
    V = wlog.shape[1]
    wlogU = np.ascontiguousarray(np.load(os.path.join(dd, "wlogU.npz"))["wlogU"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_dualhead5.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run2(ids, ukt2, evb2):
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
                             "ones1V": enc(np.ones((1, V))),
                             "thr": enc(np.full((n, V), 0.5)),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

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

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    iprompt = ids_of("The capital of Italy is")[-8:]
    hnb = dec(run0(iprompt)["HNB"])[-1]
    ikey = hnb / np.linalg.norm(hnb)
    bgkeys = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or len(ctx) < 3 or len(bgkeys) >= NBG:
                continue
            h = dec(run0(ctx)["HNB"])[-1]
            bgkeys.append(h / np.linalg.norm(h))
        if len(bgkeys) >= NBG:
            break
    romeU = np.ascontiguousarray(wlogU[:, 98])

    def bank(negkeys):
        U = np.concatenate([k[:, None] for k in bgkeys] + [ikey[:, None]]
                           + [k[:, None] for k in negkeys], axis=1) * KS
        E = np.concatenate([np.zeros((NBG, 16)), (DOSE * evn * romeU)[None, :]]
                           + [np.zeros((1, 16))] * len(negkeys), axis=0)
        return U, E

    Ua, Vc = bank([])
    # pass 1: collect flipped UNK contexts
    flips = []
    gh = gt = ch = ct = uh = ut = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            ctx = ids[max(0, k - 8):k]
            lg = dec(run2(ctx, Ua, Vc)["LOGITS2"])[-1]
            hit = int(lg.argmax()) == truth
            tot += 1
            ut += hit
            if truth == 0:
                uh += hit
                t0 = int(dec(run0(ctx)["LOGITS2"])[-1].argmax())
                if int(lg.argmax()) != t0 and len(flips) < MAXNEG:
                    flips.append(ctx)
                continue
            if freq[truth] >= fcut:
                gh += hit
                gt += 1
            else:
                ch += hit
                ct += 1
    print(f"pass1: top1={ut / max(tot, 1):.3f} glue={gh}/{gt} content={ch}/{ct} "
          f"unk={uh} negs={len(flips)}", flush=True)
    negkeys = []
    for ctx in flips:
        h = dec(run0(ctx)["HNB"])[-1]
        negkeys.append(h / np.linalg.norm(h))
    Ua2, Vc2 = bank(negkeys)
    print(f"bank: {Ua2.shape[1]} stores ({len(negkeys)} neg-mined)", flush=True)

    fi = run2(iprompt, Ua2, Vc2)
    li = dec(fi["LOGITS2"])[-1]
    bp = dec(fi["BP2"])[-1]
    print(f"install: rank={int((li > li[98]).sum()) + 1} "
          f"top={inv.get(int(li.argmax()), '?')} w_italy={float(bp[NBG]):.3f} "
          f"{'FLIP' if int(li.argmax()) == 98 else ''}", flush=True)
    gh = gt = ch = ct = uh = ut = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            ctx = ids[max(0, k - 8):k]
            lg = dec(run2(ctx, Ua2, Vc2)["LOGITS2"])[-1]
            hit = int(lg.argmax()) == truth
            tot += 1
            ut += hit
            if truth == 0:
                uh += hit
                continue
            if freq[truth] >= fcut:
                gh += hit
                gt += 1
            else:
                ch += hit
                ct += 1
    print(f"pass2: top1={ut / max(tot, 1):.3f} glue={gh}/{gt} content={ch}/{ct} "
          f"unk={uh}", flush=True)


if __name__ == "__main__":
    main()
