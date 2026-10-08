"""Rung-2 probe 2: column-norm ablation (unit-norm readout).

Probe 1 exonerated depth: the wall is norm structure (|w_rome| 0.84
vs unk 6.98). This zeroes the skew: wlogU = unit columns, everything
else identical (same banks, same program, same keys). Two questions:
(A) does glue survive on alignment alone (top1/glue/content/unk
split, curve-comparable)? (B) does the post-L2 Rome install dose
collapse (rank ladder + holds)? Variant lives in /tmp (repo data
untouched). Readings: A-yes + B-collapse -> redesign = tier norms;
A-no -> norms ARE the prior, redesign must replace it another way.
Usage: python3 research/norm_ablate.py (CPU lattice, ~300 runs)
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
DOSES = [1, 2, 4, 8]


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
    norms = np.linalg.norm(wlog, axis=0, keepdims=True)
    wlogU = np.ascontiguousarray(wlog / np.maximum(norms, 1e-12))
    np.save("/tmp/wlogU.npy", wlogU)
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    freq = np.asarray(counts.sum(axis=0)).ravel()
    fcut = np.sort(freq)[-64]
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt2, evb2, W):
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
                             "wlog": enc(W),
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    zero2 = (ukt0[:, :1], np.zeros((1, 16)))

    # (A) base split under unit-norm readout
    gh = gt = ch = ct = uh = ut = tot = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            lg = dec(run(ids[max(0, k - 8):k], *zero2, wlogU)["LOGITS2"])[-1]
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
    print(f"(A) unit-norm base: top1={ut / max(tot, 1):.3f} glue={gh}/{gt} "
          f"content={ch}/{ct} unk={uh} (skewed base: 0.337, 6/87, 0/40, 77)",
          flush=True)

    # (B) install ladder under unit-norm readout
    iprompt = ids_of("The capital of Italy is")[-8:]
    f0 = run(iprompt, *zero2, wlogU)
    lg0 = dec(f0["LOGITS2"])[-1]
    hnb0 = dec(f0["HNB"])[-1]
    print(f"(B) unit base: Rome-rank={int((lg0 > lg0[98]).sum()) + 1} "
          f"top={inv.get(int(lg0.argmax()), '?')}", flush=True)
    key = (hnb0 / np.linalg.norm(hnb0))[:, None]
    romeU = np.ascontiguousarray(wlogU[:, 98])  # already unit
    cprompts = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    base_tops = {}
    for ids, tgt in cprompts:
        base_tops[tuple(ids)] = int(
            dec(run(ids, *zero2, wlogU)["LOGITS2"])[-1].argmax())
    for dx in DOSES:
        evb2 = (dx * evn * romeU)[None, :]
        f2 = run(iprompt, key, evb2, wlogU)
        lg = dec(f2["LOGITS2"])[-1]
        o = np.argsort(-lg)[:5]
        hold = sum(1 for ids, tgt in cprompts
                   if int(dec(run(ids, key, evb2, wlogU)["LOGITS2"])[-1].argmax())
                   == base_tops[tuple(ids)])
        print(f"dose={dx}x: rank={int((lg > lg[98]).sum()) + 1} "
              f"top5={[(inv.get(int(i), '?'), round(float(lg[int(i)]), 2)) for i in o]} "
              f"hold={hold}/16 {'FLIP' if int(lg.argmax()) == 98 else ''}",
              flush=True)


if __name__ == "__main__":
    main()
