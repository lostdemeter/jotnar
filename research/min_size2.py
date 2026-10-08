"""Minimum size, curve 2 (organized ball): K sweep with install.

Curve 1 (min_size.py) found the floor: content 0/40 at every K, UNK
carries all, size irrelevant -- so curve 2 must CREATE, not preserve.
Same truncation rule (top-K by evb row-norm), but each K gets the
organized ball: yarnball_bank + the MVYB Italy store (native emb key
x8, Rome value 8x) through programs/lm_yarnball.asm. Per K: Italy
Rome-rank + retrieval + 16-prompt holds (stable/razor split) + the
curve-1 top1/glue/content/unk split for comparability.
Question: down to what K do install + holds coexist? A cliff here
(after none in curve 1) IS the price -- or proof -- of organization.
Usage: python3 research/min_size2.py (CPU lattice, ~100 runs)
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
KS = [128, 64, 32, 16, 8]
MARGIN_BAR = 1.0


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
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
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

    e261 = np.ascontiguousarray(dE["emb"][261])
    e261 /= np.linalg.norm(e261)
    romeU = np.ascontiguousarray(wlog[:, 98])
    romeU /= np.linalg.norm(romeU)
    iprompt = ids_of("The capital of Italy is")[-8:]
    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    cprompts = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    order = np.argsort(-np.linalg.norm(evb0, axis=1))
    for K in KS:
        keep = np.sort(order[:K])
        uk, ev = ukt0[:, keep], evb0[keep, :]
        Ua, Vc, _ = yarnball_bank(uk, ev, [
            {"key": e261, "value": romeU, "dose": 8.0 * evn,
             "tier": "assoc", "support": "Italy->Rome MVYB"}],
            key_scale=8.0)
        Kk = Ua.shape[1] - 1
        fi = run(iprompt, Ua, Vc)
        li = dec(fi["LOGITS"])[-1]
        rank = int((li > li[98]).sum()) + 1
        ret = int(dec(fi[Pkey(fi)])[-1].argmax())
        base_tops, hold, shold, nst = {}, 0, 0, 0
        for ids, tgt in cprompts:
            lg0 = dec(run(ids, uk, ev)["LOGITS"])[-1]
            o = np.argsort(-lg0)
            base_tops[tuple(ids)] = (int(lg0.argmax()),
                                     float(lg0[o[0]] - lg0[o[1]]))
        for ids, tgt in cprompts:
            t1 = int(dec(run(ids, Ua, Vc)["LOGITS"])[-1].argmax())
            if t1 == base_tops[tuple(ids)][0]:
                hold += 1
                if base_tops[tuple(ids)][1] >= MARGIN_BAR:
                    shold += 1
            if base_tops[tuple(ids)][1] >= MARGIN_BAR:
                nst += 1
        gh = gt = ch = ct = uh = ut = tot = 0
        for s in lines:
            ids = ids_of(s)
            for k in range(1, len(ids)):
                truth = ids[k]
                lg = dec(run(ids[max(0, k - 8):k], Ua, Vc)["LOGITS"])[-1]
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
        print(f"K={K:3}: Rome-rank={rank:3} ret={ret}/{Kk} "
              f"hold={hold}/16 stable={shold}/{nst} "
              f"top1={ut / max(tot, 1):.3f} glue={gh}/{gt} "
              f"content={ch}/{ct} unk={uh}", flush=True)


if __name__ == "__main__":
    main()
