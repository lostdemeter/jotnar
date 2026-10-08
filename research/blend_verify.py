"""Blend verification: decode the v2 intermediates on flipped positions.

Attribution claims v2's 11 UNK flips come from blend arithmetic
(x(1-w) attenuation + w.xFLAT injection, ~0.1-logit scale). Verify,
don't assert: on Italy (install) + the first flipped UNK positions,
decode WI/W/NW/FW/SW/LOGITS2 and check (a) W tiles BP2[6], (b) NW =
1-W, (c) FW+SW = LOGITS2 path, (d) which term dominates the flip:
|FW| vs |SW-LOGSKEW| at the flipped top-2. A mismatch anywhere is a
CODE bug (error-source audit with a target); all-match closes the
mechanism to arithmetic (no trawl needed).
Usage: python3 research/blend_verify.py (CPU lattice)
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
    wlogU = np.ascontiguousarray(np.load("/tmp/wlogU.npy"))
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    text = CFG + open(os.path.join(sdir, "lm_dualhead2.asm")).read()
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
    nbg = 0
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or len(ctx) < 3:
                continue
            h = dec(run0(ctx)["HNB"])[-1]
            bgkeys.append(h / np.linalg.norm(h))
            nbg += 1
            if nbg >= NBG:
                break
        if nbg >= NBG:
            break
    Ua = np.concatenate([k[:, None] for k in bgkeys] + [ikey[:, None]],
                        axis=1) * KS
    romeU = np.ascontiguousarray(wlogU[:, 98])
    Vc = np.concatenate([np.zeros((NBG, 16)), (DOSE * evn * romeU)[None, :]],
                        axis=0)

    # Italy receipt first
    fi = run2(iprompt, Ua, Vc)
    bpi = dec(fi["BP2"])[-1]
    print(f"Italy: w= {float(bpi[NBG]):.4f} top={inv.get(int(dec(fi['LOGITS2'])[-1].argmax()), '?')}",
          flush=True)

    # find flipped UNK positions
    flips = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, len(ids)):
            truth = ids[k]
            if truth != 0:
                continue
            ctx = ids[max(0, k - 8):k]
            t0 = int(dec(run0(ctx)["LOGITS2"])[-1].argmax())
            f2 = run2(ctx, Ua, Vc)
            t1 = int(dec(f2["LOGITS2"])[-1].argmax())
            if t0 != t1:
                flips.append((ctx, t0, t1, f2))
                if len(flips) >= 3:
                    break
        if len(flips) >= 3:
            break
    print(f"flipped UNK positions: {len(flips)} (showing up to 3)", flush=True)
    for ctx, t0, t1, f2 in flips:
        n = len(ctx)
        WI = dec(f2["WI"])[-1][0]
        W = dec(f2["W"])[-1]
        NW = dec(f2["NW"])[-1]
        FW = dec(f2["FW"])[-1]
        SW = dec(f2["SW"])[-1]
        LG = dec(f2["LOGITS2"])[-1]
        LS = dec(f2["LOGSKEW"])[-1]
        LF = dec(f2["LOGFLAT"])[-1]
        BP = dec(f2["BP2"])[-1]
        w = float(BP[NBG])
        print(f"ctx={[inv.get(i, '?') for i in ctx]} {inv.get(t0, '?')}->{inv.get(t1, '?')} "
              f"w={w:.4f}", flush=True)
        print(f"  (a) W tiles WI: max|W-WI|={float(np.abs(W - WI).max()):.2e} "
              f"(b) NW=1-W: max|NW-(1-W)|={float(np.abs(NW - (1 - W)).max()):.2e}",
              flush=True)
        print(f"  (c) FW=w.FLAT: max|FW-w.LF|={float(np.abs(FW - w * LF).max()):.2e} "
              f"SW=(1-w).SKEW: max|SW-(1-w).LS|={float(np.abs(SW - (1 - w) * LS).max()):.2e}",
              flush=True)
        print(f"  (d) at flipped pair: |FW[t0]|={abs(float(FW[t0])):.3f} "
              f"|FW[t1]|={abs(float(FW[t1])):.3f} "
              f"|SW[t0]-LS[t0]|={abs(float(SW[t0] - LS[t0])):.3f} "
              f"|SW[t1]-LS[t1]|={abs(float(SW[t1] - LS[t1])):.3f}", flush=True)


if __name__ == "__main__":
    main()
