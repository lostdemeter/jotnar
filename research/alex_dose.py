"""Alex dose ladder: retrieval is partial (wsum 0.14), margin is 10.08.

Entity-row keys moved Alex 6->4. Remaining wall is dose: at full
mass, 4x covers ~4 margin units vs 10.08 needed. Ladder Alex value
4/8/12/16x (others fixed 4x): rank + receipt + holds per cell.
Routing confines dose to Alex rows (base mass ~0 there), so holds
should survive where broadcast dose died. Live bank, no loop.
Usage: python3 research/alex_dose.py (CPU lattice, ~100 runs)
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

    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    emb = np.ascontiguousarray(dE["emb"])
    aprompt = ids_of("Alexander the Great founded the city of")[-8:]
    order = ["italy", "alex", "caesar"]
    epos = {"italy": 3, "alex": 0, "caesar": 1}
    eids = {"italy": 261, "alex": 12, "caesar": 40}
    prompts = {"italy": ids_of("The capital of Italy is")[-8:],
               "alex": aprompt,
               "caesar": ids_of("Julius Caesar was assassinated in the city of")[-8:]}
    fkeys = {}
    for f in order:
        ids = prompts[f]
        assert ids[epos[f]] == eids[f], (f, ids)
        h = dec(run0(ids)["HN"])[epos[f]]
        fkeys[f] = h / np.linalg.norm(h)
    romeU = np.ascontiguousarray(wlogU[:, 98])
    alexU = np.ascontiguousarray(wlogU[:, 59])
    bgHN, bgE = [], []
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
    for w in ["city", "son", "syria", "egypt", "her", "his"]:
        if w in vocab and len(bgE) < NBG:
            e = np.ascontiguousarray(emb[vocab[w]])
            bgE.append(e / np.linalg.norm(e))
    Uhn = np.concatenate([k[:, None] for k in bgHN]
                         + [fkeys[f][:, None] for f in order], axis=1) * KS
    Ue = np.concatenate([k[:, None] for k in bgE]
                        + [(np.ascontiguousarray(emb[eids[f]])
                            / np.linalg.norm(emb[eids[f]]))[:, None]
                           for f in order], axis=1) * KS
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
        lg = dec(run0(ids)["LOGITS2"])[-1]
        o = np.argsort(-lg)
        base_tops[tuple(ids)] = (int(lg.argmax()), float(lg[o[0]] - lg[o[1]]))
    nst = sum(1 for v in base_tops.values() if v[1] >= MARGIN_BAR)
    for dx in (4, 8, 12, 16):
        Vc = np.concatenate([np.zeros((NBG, 16)),
                             (4.0 * evn * romeU)[None, :],
                             (dx * evn * alexU)[None, :],
                             (4.0 * evn * romeU)[None, :]], axis=0)
        fi = runS(aprompt, Uhn, Ue, Vc)
        li = dec(fi["LOGITS2"])[-1]
        pr = dec(fi["PR"])[0]
        h = hs = 0
        for ids, tgt in cprompts:
            t1 = int(dec(runS(ids, Uhn, Ue, Vc)["LOGITS2"])[-1].argmax())
            if t1 == base_tops[tuple(ids)][0]:
                h += 1
                if base_tops[tuple(ids)][1] >= MARGIN_BAR:
                    hs += 1
        print(f"alex-dose={dx}x: rank={int((li > li[59]).sum()) + 1} "
              f"top={inv.get(int(li.argmax()), '?')} wsum={float(pr[NBG:NBG + 3].sum()):.3f} "
              f"hold={h}/16 stable={hs}/{nst} "
              f"{'FLIP' if int(li.argmax()) == 59 else ''}", flush=True)


if __name__ == "__main__":
    main()
