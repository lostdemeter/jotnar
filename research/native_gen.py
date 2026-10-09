"""Native generation survival: dual-head install through greedy decoding.

Teacher installs degenerate generation (per-step adds compound
via feedback). Same question natively, cheap: dualhead2 routed
install vs pure base, greedy loop (short prompt + 4 steps stays
S<=8), mask1 vs mask0 trajectories side by side. Metric-setting run:
report strings + per-step tops; survival = Rome appears and text
stays on-distribution (no quantitative coherence meter yet -- the
strings decide what meter to build).
Usage: python3 research/native_gen.py (CPU lattice)
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
N_GEN = 4


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
    text = CFG + open(os.path.join(sdir, "lm_dualhead2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt2, evb2):
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

    text0 = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

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
    f0 = run0([1, 3, 261, 146])
    hnb = dec(f0["HNB"])[-1]
    ikey = hnb / np.linalg.norm(hnb)
    lines = open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]
    bgkeys = []
    for s in lines:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            ctx = ids[max(0, k - 7):k]
            if 261 in ctx or len(ctx) < 3 or len(bgkeys) >= 6:
                continue
            h = dec(run0(ctx)["HNB"])[-1]
            bgkeys.append(h / np.linalg.norm(h))
        if len(bgkeys) >= 6:
            break
    Ua = np.concatenate([k[:, None] for k in bgkeys] + [ikey[:, None]],
                        axis=1) * 32.0
    romeU = np.ascontiguousarray(wlogU[:, 98])
    Vc = np.concatenate([np.zeros((6, 16)), (4.0 * evn * romeU)[None, :]],
                        axis=0)
    # NOTE: dualhead2 bakes SLICE(BP2,1,6,7): bank MUST be 6bg + Italy.
    for tag, fn in (("base", lambda ids: dec(run0(ids)["LOGITS2"])[-1]),
                    ("ball", lambda ids: dec(run(ids, Ua, Vc)["LOGITS2"])[-1])):
        seq = [1, 3, 261, 146]
        outs = []
        for _ in range(N_GEN):
            lg = fn(seq)
            top = int(lg.argmax())
            outs.append((inv.get(top, "?"), round(float(lg[98]), 2)))
            seq = (seq + [top])[-8:]
        print(f"native-gen {tag}: " +
              " ".join(f"{w}(rome={r})" for w, r in outs), flush=True)


if __name__ == "__main__":
    main()
