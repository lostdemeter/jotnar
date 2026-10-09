"""Dual-head v2 gate: in-listing blend, no host mask.

Bank: 6 background null stores (HNB-space keys from control prompts)
+ Italy store LAST (unit-Rome 4x value). Readout blends by the bank's
own Italy weight (SLICE BP2 col 6 -> tile -> MUL/SUB/ADD): w~1 reads
flat (installs), w~0 reads skewed (prior). ones1V/thr frozen per
geometry, cmask class.
Gates: Italy FLIP + weight receipt (w_italy>0.9 Italy, <0.1 mean base)
+ full split == skewed base + stable holds. Failures locate: receipt
fails -> addressing; split differs -> tier interference; FLIP fails
with receipt green -> blend arithmetic (quantum).
Usage: python3 research/dualhead2.py (CPU lattice)
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
        o1 = np.ones((1, V))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog), "wlogU": enc(wlogU),
                             "ones1V": enc(o1),
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

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:int(os.environ.get("NLINES", 10))]
    iprompt = ids_of("The capital of Italy is")[-8:]

    # background keys: HNB rows on non-Italy controls (frozen address data)
    bgids = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or len(ctx) < 3:
                continue
            bgids.append(ctx)
            if len(bgids) >= NBG:
                break
        if len(bgids) >= NBG:
            break
    bgkeys = []
    for ctx in bgids:
        h = dec(run0(ctx)["HNB"])[-1]
        bgkeys.append(h / np.linalg.norm(h))
    hnb = dec(run0(iprompt)["HNB"])[-1]
    ikey = hnb / np.linalg.norm(hnb)
    Ua = np.concatenate([k[:, None] for k in bgkeys] + [ikey[:, None]],
                        axis=1) * KS
    romeU = np.ascontiguousarray(wlogU[:, 98])
    Vc = np.concatenate([np.zeros((NBG, 16)), (DOSE * evn * romeU)[None, :]],
                        axis=0)
    print(f"bank: {Ua.shape[1]} stores (Italy last idx {NBG})", flush=True)

    fi = run2(iprompt, Ua, Vc)
    li = dec(fi["LOGITS2"])[-1]
    bp = dec(fi["BP2"])[-1]
    print(f"install: rank={int((li > li[98]).sum()) + 1} "
          f"top={inv.get(int(li.argmax()), '?')} w_italy={float(bp[NBG]):.3f} "
          f"{'FLIP' if int(li.argmax()) == 98 else ''}", flush=True)

    # prior split + weight receipt on battery + holds
    cprompts = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    ws = []
    for ids, tgt in cprompts:
        ws.append(float(dec(run2(ids, Ua, Vc)["BP2"])[-1][NBG]))
    print(f"receipt: Italy w={float(bp[NBG]):.3f}, battery mean w={np.mean(ws):.3f} "
          f"max={np.max(ws):.3f}", flush=True)
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
                continue
            if freq[truth] >= fcut:
                gh += hit
                gt += 1
            else:
                ch += hit
                ct += 1
    print(f"prior: top1={ut / max(tot, 1):.3f} glue={gh}/{gt} content={ch}/{ct} "
          f"unk={uh} (skewed base: 0.337, 6/87, 0/40, 77)", flush=True)
    base_tops, hold, shold, nst = {}, 0, 0, 0
    for ids, tgt in cprompts:
        lg = dec(run0(ids)["LOGITS2"])[-1]
        o = np.argsort(-lg)
        base_tops[tuple(ids)] = (int(lg.argmax()), float(lg[o[0]] - lg[o[1]]))
    for ids, tgt in cprompts:
        t1 = int(dec(run2(ids, Ua, Vc)["LOGITS2"])[-1].argmax())
        if t1 == base_tops[tuple(ids)][0]:
            hold += 1
            if base_tops[tuple(ids)][1] >= MARGIN_BAR:
                shold += 1
        if base_tops[tuple(ids)][1] >= MARGIN_BAR:
            nst += 1
    print(f"holds: {hold}/16 stable={shold}/{nst}", flush=True)


if __name__ == "__main__":
    main()
