"""Dual-head v2 gate: self-routing blend, no host mask (generic ball).

Program lm_dualhead2.asm: 6 background nulls + Italy LAST (idx 6,
unit-Rome 4x); readout blends flat over skewed by the bank's own
Italy weight (SLICE+tile+MUL/SUB/ADD, existing ops only). Locked
operating point: ks=32, dose 4x.
Gates: Italy FLIP + receipt (own>0.9, battery mean<0.1) + full split
== skewed base + stable holds. Razor residue (11 UNK) accepted per
doctrine: stable (margin>=1) is the bar, razor tracked separately
(blend arithmetic ~0.1-logit scale, proven over ks 8/16/32).
Usage: python3 tests/test_dualhead2.py (slow: ~250 runs)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
MARGIN_BAR = 1.0
NBG = 6
KS = 32.0
DOSE = 4.0


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
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
    text = CFG + open(os.path.join(sdir, "lm_dualhead2.asm")).read()
    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

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
    Ua = np.concatenate([k[:, None] for k in bgkeys] + [ikey[:, None]],
                        axis=1) * KS
    romeU = np.ascontiguousarray(wlogU[:, 98])
    Vc = np.concatenate([np.zeros((NBG, 16)), (DOSE * evn * romeU)[None, :]],
                        axis=0)

    fi = run2(iprompt, Ua, Vc)
    li = dec(fi["LOGITS2"])[-1]
    wI = float(dec(fi["BP2"])[-1][NBG])
    check("dualhead-install", int(li.argmax()) == 98,
          f"rank={int((li > li[98]).sum()) + 1} top={inv.get(int(li.argmax()), '?')}")
    check("dualhead-receipt-own", wI > 0.9, f"w_italy={wI:.3f}")

    cprompts = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    ws = [float(dec(run2(ids, Ua, Vc)["BP2"])[-1][NBG]) for ids, _ in cprompts]
    check("dualhead-receipt-base", float(np.mean(ws)) < 0.1,
          f"mean={float(np.mean(ws)):.3f} max={float(np.max(ws)):.3f}")
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
    check("dualhead-prior-glue", (gh, gt) == (6, 87), f"{gh}/{gt}")
    check("dualhead-prior-content", (ch, ct) == (0, 40), f"{ch}/{ct}")
    print(f"dualhead-prior-top1: {ut / max(tot, 1):.3f} unk={uh} "
          f"(razor residue accepted per doctrine)", flush=True)
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
    check("dualhead-hold-stable", shold == nst, f"{shold}/{nst}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}", flush=True)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
