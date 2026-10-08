"""Entity-routed install gate (address early/lexical, content late).

Program lm_siphon.asm: entity row found by E-space match (position-
free), dual-channel bank scores (HN contextual + E lexical) read AT
that row, value tiles, dual heads blend by install mass. Keys: HN
mined rows + raw embeddings.
minus function mean), separation VERIFIED in-script before trusting.
Battery: 4 installs (italy, italy2-template-variant, alex, caesar)
+ swap-distractor (must HOLD) + holds + split + negmine loop.
Gates: ALL installs FLIP + swap holds + prior == skewed base +
receipts.
by fiat is refused here).
Usage: python3 research/eroute.py (CPU lattice, ~700 runs)
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
MAXNEG = 16
CONTENT_WORDS = ["italy", "rome", "caesar", "alexander", "alexandria",
                 "city", "son", "syria", "egypt", "her", "his"]
FUNC_WORDS = ["the", "and", "of", "is", "a", "to", "in", "as"]


def main():
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    emb = np.ascontiguousarray(dE["emb"])
    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    V = wlog.shape[1]
    wlogU = np.ascontiguousarray(np.load(os.path.join(dd, "wlogU.npz"))["wlogU"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_siphon.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

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
                             "onesS1": enc(np.ones((n, 1))),
                             "ones1V": enc(np.ones((1, V))),
                             "onesSV": enc(np.ones((n, V))),
                             "ukt": enc(ukt0), "evb": enc(evb0)},
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

    inst = {"italy": (ids_of("The capital of Italy is")[-8:], 98),
            "italy2": (ids_of("the city of italy is")[-8:], 98),
            "alex": (ids_of("Alexander the Great founded the city of")[-8:], 59),
            "caesar": (ids_of("Julius Caesar was assassinated in the city of")[-8:], 98)}
    swaps = {"caesar-city": (ids_of("the city of caesar is")[-8:], None)}
    for name, (ids, t) in list(inst.items()) + list(swaps.items()):
        lg = dec(run0(ids)["LOGITS2"])[-1]
        o = np.argsort(-lg)[:3]
        print(f"base {name:11} rank={(int((lg > lg[t]).sum()) + 1) if t else '-':>4} "
              f"top3={[(inv.get(int(i), '?'), round(float(lg[int(i)]), 1)) for i in o]}",
              flush=True)
    order = ["italy", "alex", "caesar"]
    eids = {"italy": 261, "alex": 12, "caesar": 40}
    fkeys, fvals = {}, {}
    for f in order:
        h = dec(run0(inst[f][0])["HN"])[-1]
        fkeys[f] = h / np.linalg.norm(h)
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    fvals = {"italy": (DOSE * evn * romeU)[None, :],
             "alex": (DOSE * evn * alexU)[None, :],
             "caesar": (DOSE * evn * romeU)[None, :]}
    bgE = []
    for w in ["city", "son", "syria", "egypt", "her", "his", "without",
              "amid", "victory", "sources", "battle", "army"]:
        if w in vocab and len(bgE) < NBG:
            e = np.ascontiguousarray(emb[vocab[w]])
            bgE.append(e / np.linalg.norm(e))
    assert len(bgE) == NBG, f"only {len(bgE)} bg E-keys found in vocab"
    bgHN = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or 12 in ctx or 40 in ctx or len(ctx) < 3 \
                    or len(bgHN) >= NBG:
                continue
            h = dec(run0(ctx)["HN"])[-1]
            bgHN.append(h / np.linalg.norm(h))
        if len(bgHN) >= NBG:
            break

    def bank(negHN, negE):
        U = np.concatenate([k[:, None] for k in bgHN]
                           + [fkeys[f][:, None] for f in order]
                           + [k[:, None] for k in negHN], axis=1) * KS
        E = np.concatenate([k[:, None] for k in bgE]
                           + [(np.ascontiguousarray(emb[eids[f]])
                               / np.linalg.norm(emb[eids[f]]))[:, None]
                              for f in order]
                           + [k[:, None] for k in negE], axis=1) * KS
        Vc = np.concatenate([np.zeros((NBG, 16))]
                            + [fvals[f] for f in order]
                            + [np.zeros((1, 16))] * len(negHN), axis=0)
        return U, E, Vc

    def split(Ua, Ea, Vc):
        gh = gt = ch = ct = uh = ut = tot = 0
        flips = []
        for s in lines:
            ids = ids_of(s)
            for k in range(1, len(ids)):
                truth = ids[k]
                ctx = ids[max(0, k - 8):k]
                lg = dec(runS(ctx, Ua, Ea, Vc)["LOGITS2"])[-1]
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

    Ua, Ea, Vc = bank([], [])
    m1 = split(Ua, Ea, Vc)
    print(f"pass1: top1={m1[0]:.3f} glue={m1[1]}/{m1[2]} content={m1[3]}/{m1[4]} "
          f"unk={m1[5]} negs={len(m1[6])}", flush=True)
    negHN, negE = [], []
    for ctx in m1[6]:
        h = dec(run0(ctx)["HN"])[-1]
        negHN.append(h / np.linalg.norm(h))
        toks = np.array(ctx)
        ent_tok = toks[-1]
        e = np.ascontiguousarray(emb[int(ent_tok)])
        n = float(np.linalg.norm(e))
        negE.append(e / n if n > 0 else bgE[0])
    Ua2, Ea2, Vc2 = bank(negHN, negE)
    print(f"bank: {Ua2.shape[1]} stores ({len(negHN)} neg-mined)", flush=True)
    for name, (ids, t) in list(inst.items()) + list(swaps.items()):
        fi = runS(ids, Ua2, Ea2, Vc2)
        li = dec(fi["LOGITS2"])[-1]
        assert "PR" in fi, sorted(fi.keys())
        pr = dec(fi["PR"])[0]
        w = float(pr[NBG:NBG + 3].sum()) if pr is not None else -1
        if t is None:
            t0 = int(dec(run0(ids)["LOGITS2"])[-1].argmax())
            print(f"swap {name}: top={inv.get(int(li.argmax()), '?')} "
                  f"(base {inv.get(t0, '?')}) wsum={w:.3f} "
                  f"{'HOLD' if int(li.argmax()) == t0 else 'LEAK'}", flush=True)
        else:
            print(f"install {name:7}: rank={int((li > li[t]).sum()) + 1} "
                  f"top={inv.get(int(li.argmax()), '?')} wsum={w:.3f} "
                  f"{'FLIP' if int(li.argmax()) == t else ''}", flush=True)
    m2 = split(Ua2, Ea2, Vc2)
    print(f"pass2: top1={m2[0]:.3f} glue={m2[1]}/{m2[2]} content={m2[3]}/{m2[4]} "
          f"unk={m2[5]}", flush=True)


if __name__ == "__main__":
    main()
