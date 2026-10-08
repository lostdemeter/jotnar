"""Scale-2 gate: two native facts through one self-routing bank + negmine loop.

Bank: 6 background nulls + Italy(6, unit-Rome 4x) + Alex(7,
unit-Alexandria 4x) + neg-mined nulls. Blend weight = w6+w7 (either
install context reads flat). Loop: pass-1 split -> collect flipped
UNK -> mine keys -> append nulls -> pass-2 split + both installs +
receipts + holds. Gates: BOTH FLIP + prior == skewed base + receipts
w>0.9 on own prompts + holds. Cross-talk (shared flat head, shared
H5) is the question: Rome must win on Italy rows, Alexandria on
Alex rows.
Usage: python3 research/scale2.py (CPU lattice, ~500 runs)
"""
import json
import os
import re
import sys

import numpy as np

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
MARGIN_BAR = 1.0
NBG = 6
KS = 32.0
DOSE = 4.0
MAXNEG = 16


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
    text = CFG + open(os.path.join(sdir, "lm_dualhead3.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run3(ids, ukt2, evb2):
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
                             "onesSV": enc(np.ones((n, V))),
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
    prompts = {"italy": ids_of("The capital of Italy is")[-8:],
               "alex": ids_of("Alexander the Great founded the city of")[-8:]}
    tids = {"italy": 98, "alex": 59}
    fkeys, fvals = {}, {}
    for f, ids in prompts.items():
        h = dec(run0(ids)["HNB"])[-1]
        fkeys[f] = h / np.linalg.norm(h)
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    fvals = {"italy": (DOSE * evn * romeU)[None, :],
             "alex": (DOSE * evn * alexU)[None, :]}
    bgkeys = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or 12 in ctx or len(ctx) < 3 or len(bgkeys) >= NBG:
                continue
            h = dec(run0(ctx)["HNB"])[-1]
            bgkeys.append(h / np.linalg.norm(h))
        if len(bgkeys) >= NBG:
            break

    def bank(negkeys):
        U = np.concatenate([k[:, None] for k in bgkeys]
                           + [fkeys["italy"][:, None], fkeys["alex"][:, None]]
                           + [k[:, None] for k in negkeys], axis=1) * KS
        E = np.concatenate([np.zeros((NBG, 16)), fvals["italy"], fvals["alex"]]
                           + [np.zeros((1, 16))] * len(negkeys), axis=0)
        return U, E

    def split(Ua, Vc):
        gh = gt = ch = ct = uh = ut = tot = 0
        flips = []
        for s in lines:
            ids = ids_of(s)
            for k in range(1, len(ids)):
                truth = ids[k]
                ctx = ids[max(0, k - 8):k]
                lg = dec(run3(ctx, Ua, Vc)["LOGITS2"])[-1]
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
        return (ut / max(tot, 1), gh, gt, ch, ct, uh, flips)

    Ua, Vc = bank([])
    m1 = split(Ua, Vc)
    print(f"pass1: top1={m1[0]:.3f} glue={m1[1]}/{m1[2]} content={m1[3]}/{m1[4]} "
          f"unk={m1[5]} negs={len(m1[6])}", flush=True)
    negkeys = []
    for ctx in m1[6]:
        h = dec(run0(ctx)["HNB"])[-1]
        negkeys.append(h / np.linalg.norm(h))
    Ua2, Vc2 = bank(negkeys)
    print(f"bank: {Ua2.shape[1]} stores ({len(negkeys)} neg-mined)", flush=True)
    for f, ids in prompts.items():
        fi = run3(ids, Ua2, Vc2)
        li = dec(fi["LOGITS2"])[-1]
        bp = dec(fi["BP2"])[-1]
        t = tids[f]
        w = float(bp[NBG + (0 if f == "italy" else 1)])
        check(f"scale2-install-{f}", int(li.argmax()) == t,
              f"rank={int((li > li[t]).sum()) + 1} w={w:.3f}")
        check(f"scale2-receipt-{f}", w > 0.9, f"w={w:.3f}")
    m2 = split(Ua2, Vc2)
    print(f"pass2: top1={m2[0]:.3f} glue={m2[1]}/{m2[2]} content={m2[3]}/{m2[4]} "
          f"unk={m2[5]}", flush=True)
    check("scale2-prior", (round(m2[0], 3), m2[1], m2[2], m2[3], m2[4], m2[5])
          == (0.337, 6, 87, 0, 40, 77),
          f"{m2[0]:.3f} {m2[1]}/{m2[2]} {m2[3]}/{m2[4]} {m2[5]}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
