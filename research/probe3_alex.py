"""Rung-2 probe 3: high-rank-target install through the SKEWED head.

Probe 2 proved installability is norm-geometry (flat head flips at
4x) and the prior is the same knob (glue dies flat). Remaining path
to flip WITH prior intact: targets already near the top, where
modest dose suffices despite skew. Alexander prompt (base rank ~6):
Alexandria readout value through the linear post-L2 bank, ladder
1/2/4/8x + holds. Gates: rank 1 at any dose with stable holds green
-> native top-1 WITH prior (target selection is the product form);
stall -> skewed head blocks even rank-6 targets -> dual-head design.
Usage: python3 research/probe3_alex.py (CPU lattice)
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
PROMPT = "Alexander the Great founded the city of"
TID = 59


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
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt2, evb2):
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
                             "ukt": enc(ukt0), "evb": enc(evb0),
                             "ukt2": enc(ukt2), "evb2": enc(evb2)},
                            sigs=SIGS, basedir=sdir)

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    iprompt = ids_of(PROMPT)[-8:]
    print(f"prompt ids={iprompt} tid={TID} ({inv.get(TID, '?')})", flush=True)
    f0 = run(iprompt, ukt0[:, :1], np.zeros((1, 16)))
    lg0 = dec(f0["LOGITS2"])[-1]
    hnb0 = dec(f0["HNB"])[-1]
    o0 = np.argsort(-lg0)[:8]
    print(f"base: rank={int((lg0 > lg0[TID]).sum()) + 1} "
          f"top8={[(inv.get(int(i), '?'), round(float(lg0[int(i)]), 2)) for i in o0]}",
          flush=True)
    key = (hnb0 / np.linalg.norm(hnb0))[:, None]
    vd = np.ascontiguousarray(wlog[:, TID])
    vd /= np.linalg.norm(vd)
    cprompts = []
    for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    base_tops = {}
    for ids, tgt in cprompts:
        lg = dec(run(ids, ukt0[:, :1], np.zeros((1, 16)))["LOGITS2"])[-1]
        o = np.argsort(-lg)
        base_tops[tuple(ids)] = (int(lg.argmax()), float(lg[o[0]] - lg[o[1]]))
    nst = sum(1 for v in base_tops.values() if v[1] >= 1.0)
    for dx in DOSES:
        evb2 = (dx * evn * vd)[None, :]
        f2 = run(iprompt, key, evb2)
        lg = dec(f2["LOGITS2"])[-1]
        o = np.argsort(-lg)[:5]
        h = hs = 0
        for ids, tgt in cprompts:
            t1 = int(dec(run(ids, key, evb2)["LOGITS2"])[-1].argmax())
            if t1 == base_tops[tuple(ids)][0]:
                h += 1
                if base_tops[tuple(ids)][1] >= 1.0:
                    hs += 1
        print(f"dose={dx}x: rank={int((lg > lg[TID]).sum()) + 1} "
              f"top5={[(inv.get(int(i), '?'), round(float(lg[int(i)]), 2)) for i in o]} "
              f"hold={h}/16 stable={hs}/{nst} "
              f"{'FLIP' if int(lg.argmax()) == TID else ''}", flush=True)


if __name__ == "__main__":
    main()
