"""Rung-2 probe 1: install late (Rome readout row, post-layer-2 bank).

Qwen-late accepts top-1 installs; our layer-1 bank stalls at rank 8
(blockers outclimb, lin_map saturates). Difference candidates: depth
after install, norm structure. This isolates depth: lm_bankhn2.asm's
post-layer-2 bank is LINEAR to logits (HNB norm preserves direction,
head matmul exact -- corr ~0.92 proven), one norm + unembed from the
decision, like Qwen L27-of-28. Single store (broadcast by design:
retrieval is trivial, rank is the question; holds reported, not
gated -- addressing follows if rank works).
Value: Rome readout row (exact native content, the wrow species).
Ladder: value dose 1/2/4/8/16x evn on the Italy prompt; Rome rank +
top + lattice fidelity (corr actual vs evb2@wlog). Falsifier: rank
stalls with blockers outclimbing (as layer-1) -> the wall is norm
structure, not depth; rank hits 1 -> "install late" is a design law.
Usage: python3 research/postl2_install.py (CPU lattice)
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
DOSES = [1, 2, 4, 8, 16]


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

    iprompt = ids_of("The capital of Italy is")[-8:]
    f0 = run(iprompt, ukt0[:, :1], np.zeros((1, 16)))
    lg0 = dec(f0["LOGITS2"])[-1]
    hnb0 = dec(f0["HNB"])[-1]
    print(f"base: Rome-rank={int((lg0 > lg0[98]).sum()) + 1} "
          f"top={inv.get(int(lg0.argmax()), '?')} "
          f"|w_rome|={float(np.linalg.norm(wlog[:, 98])):.2f} evn={evn:.3f}",
          flush=True)
    key = (hnb0 / np.linalg.norm(hnb0))[:, None]
    romeU = np.ascontiguousarray(wlog[:, 98])
    romeU /= np.linalg.norm(romeU)
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
        base_tops[tuple(ids)] = int(
            dec(run(ids, ukt0[:, :1], np.zeros((1, 16)))["LOGITS2"])[-1].argmax())
    for dx in DOSES:
        evb2 = (dx * evn * romeU)[None, :]
        f2 = run(iprompt, key, evb2)
        lg = dec(f2["LOGITS2"])[-1]
        actual, pred = lg - lg0, evb2[0] @ wlog
        o = np.argsort(-lg)[:5]
        hold = sum(1 for ids, tgt in cprompts
                   if int(dec(run(ids, key, evb2)["LOGITS2"])[-1].argmax())
                   == base_tops[tuple(ids)])
        print(f"dose={dx}x: rank={int((lg > lg[98]).sum()) + 1} "
              f"top5={[(inv.get(int(i), '?'), round(float(lg[int(i)]), 2)) for i in o]} "
              f"corr={float(np.corrcoef(actual, pred)[0, 1]):.3f} "
              f"hold={hold}/16 {'FLIP' if int(lg.argmax()) == 98 else ''}",
              flush=True)


if __name__ == "__main__":
    main()
