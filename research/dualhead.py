"""Dual-head gate: prior intact + install flips, one program.

Post-L2 bank ALWAYS carries the unit-Rome 4x store (broadcast drug);
readout routes per prompt: install mask -> flat head (probe-2B
flips), else skewed head (probe-1 4x holds). wlogU from /tmp/wlogU.npy
(repo data untouched); mask tiled host-side like cmask.
Gates: Italy FLIP + full curve split == skewed base (prior:
top1/glue/content/unk) + stable holds green. Either gate fails ->
route leaks (mask mechanics) or tiers interfere (design).
Usage: python3 research/dualhead.py (CPU lattice)
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
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_dualhead.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, mask1, ukt2, evb2):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        im = np.tile(np.asarray(mask1, dtype=np.int64)[:, None],
                     (1, wlog.shape[1]))
        return ASM.run_text(text, REGISTRY,
                            {"tok": toks, "pos": pos, "cmask": cm,
                             "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                             "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                             "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                             "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                             "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                             "wlog": enc(wlog), "wlogU": enc(wlogU),
                             "imask": im,
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    iprompt = ids_of("The capital of Italy is")[-8:]
    # drug keyed at Italy HNB (single store; broadcast by design)
    fkey = run(iprompt, np.zeros(len(iprompt), dtype=np.int64),
               ukt0[:, :1], np.zeros((1, 16)))
    hnb = dec(fkey["HNB"])[-1]
    key = (hnb / np.linalg.norm(hnb))[:, None]
    romeU = np.ascontiguousarray(wlogU[:, 98])
    drug = (4.0 * evn * romeU)[None, :]

    def zero2():
        return ukt0[:, :1], np.zeros((1, 16))

    # prior: full split, mask 0 everywhere (must equal skewed base)
    gh = gt = ch = ct = uh = ut = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            ctx = ids[max(0, k - 8):k]
            lg = dec(run(ctx, np.zeros(len(ctx), dtype=np.int64), key,
                         drug)["LOGITS2"])[-1]
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
    print(f"prior (mask0+drug): top1={ut / max(tot, 1):.3f} glue={gh}/{gt} "
          f"content={ch}/{ct} unk={uh} (skewed base: 0.337, 6/87, 0/40, 77)",
          flush=True)

    # install: Italy mask 1
    m1 = np.zeros(len(iprompt), dtype=np.int64)
    m1[-1] = 1
    fi = run(iprompt, m1, key, drug)
    li = dec(fi["LOGITS2"])[-1]
    print(f"install (mask1): rank={int((li > li[98]).sum()) + 1} "
          f"top={inv.get(int(li.argmax()), '?')} "
          f"{'FLIP' if int(li.argmax()) == 98 else ''}", flush=True)

    # holds: battery, mask 0 (drug present, skewed head)
    cprompts = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    base_tops, hold, shold, nst = {}, 0, 0, 0
    for ids, tgt in cprompts:
        lg = dec(run(ids, np.zeros(len(ids), dtype=np.int64),
                     *zero2())["LOGITS2"])[-1]
        o = np.argsort(-lg)
        base_tops[tuple(ids)] = (int(lg.argmax()), float(lg[o[0]] - lg[o[1]]))
    for ids, tgt in cprompts:
        t1 = int(dec(run(ids, np.zeros(len(ids), dtype=np.int64), key,
                         drug)["LOGITS2"])[-1].argmax())
        if t1 == base_tops[tuple(ids)][0]:
            hold += 1
            if base_tops[tuple(ids)][1] >= MARGIN_BAR:
                shold += 1
        if base_tops[tuple(ids)][1] >= MARGIN_BAR:
            nst += 1
    print(f"holds (mask0+drug): {hold}/16 stable={shold}/{nst}", flush=True)


if __name__ == "__main__":
    main()
