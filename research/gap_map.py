"""Gap mapping + seriation: remove stores, read what fills the hole.

For the K131 catalog bank (128 native + Italy/Caesar/Alexander):
- remove each catalog store (and one random native control): fact
  ranks (necessity), other facts (entanglement), who retrieves the
  orphaned queries (gap-fill = backup coverage), holds (redundancy).
- spectral seriation of the key matrix into a ring (no privileged
  start: composition proved order-free); ring-neighbors of catalog
  stores vs measured cross-talk (layout predicts interference?).
Usage: python3 research/gap_map.py
"""
import os
import re
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
FACTS = [
    ("Italy", "The capital of Italy is", 261, 98),
    ("Caesar", "Julius Caesar was assassinated in the city of", 40, 98),
    ("Alexander", "Alexander the Great founded the city of", 12, 59),
]


def main():
    import json
    dd = os.path.join(ROOT, "data")
    sdir = os.path.join(ROOT, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    dE = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    ukt0, evb0 = b["ukt"], b["evb"]
    wlog = np.ascontiguousarray(dE["wlog"])
    evn = float(np.linalg.norm(evb0, axis=1).mean())
    z = np.load("/tmp/cat_keys.npz")
    keys = [z[f"k{i}"] * 2.0 for i in range(len(FACTS))]
    vals = []
    for (name, prompt, xid, tid) in FACTS:
        vd = np.ascontiguousarray(wlog[:, tid])
        vals.append(vd / np.linalg.norm(vd) * evn)
    K128 = ukt0.shape[1]
    ukt3 = np.concatenate([ukt0] + [k[:, None] for k in keys], axis=1)
    evb3 = np.concatenate([evb0] + [v[None, :] for v in vals], axis=0)

    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    text = CFG + open(os.path.join(sdir, "lm_bankhn.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    def run(ids, ukt, evb):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(dE["emb"]), "wq": enc(dE["wq"]),
                          "wk": enc(dE["wk"]), "wv": enc(dE["wv"]),
                          "wo": enc(dE["wo"]), "wup": enc(dE["wup"]),
                          "wgate": enc(dE["wgate"]), "wdown": enc(dE["wdown"]),
                          "rms_w1": enc(dE["rms1"]), "rms_w2": enc(dE["rms2"]),
                          "wlog": enc(wlog),
                          "ukt": enc(ukt), "evb": enc(evb)},
                         sigs=SIGS, basedir=sdir)
        return f

    def ids_of(s):
        return [vocab.get(w, 0) for w in re.findall(r"[a-z0-9']+", s.lower())]

    fprompts = [(name, ids_of(prompt)[-8:], tid)
                for name, prompt, xid, tid in FACTS]
    cprompts = []
    for s in open(os.path.join(dd, "lm_test.txt")).read().split("\n")[:10]:
        ids = ids_of(s)
        for k in range(1, min(len(ids), 4)):
            cprompts.append((ids[max(0, k - 7):k], ids[k]))
            if len(cprompts) >= 16:
                break
        if len(cprompts) >= 16:
            break
    # full-bank reference
    ref = {}
    for name, ids, tid in fprompts:
        f = run(ids, ukt3, evb3)
        lg = dec(f["LOGITS"])[-1]
        bp = dec(f["BP1"])[-1]
        ref[name] = (int((lg > lg[tid]).sum()) + 1, int(np.argmax(bp)))
    print(f"full bank ranks: "
          f"{[(n, r[0], 'ret' + str(r[1])) for n, r in ref.items()]}",
          flush=True)
    base_tops = {}
    for ids, tgt in cprompts:
        f = run(ids, ukt0, evb0)
        base_tops[tuple(ids)] = int(dec(f["LOGITS"])[-1].argmax())
    # removals: each catalog store + one native control (store 7).
    # NOTE: ukt3 columns are [128 native | 3 catalog]; evb3 rows match.
    def dropbank(drop):
        if drop < K128:
            uk = np.concatenate(
                [ukt0[:, [i for i in range(K128) if i != drop]]] +
                [k[:, None] for k in keys], axis=1)
            ev = np.concatenate(
                [evb0[[i for i in range(K128) if i != drop]]] +
                [v[None, :] for v in vals], axis=0)
        else:
            j = drop - K128
            uk = np.concatenate(
                [ukt0] + [keys[i][:, None] for i in range(3) if i != j],
                axis=1)
            ev = np.concatenate(
                [evb0] + [vals[i][None, :] for i in range(3) if i != j],
                axis=0)
        return uk, ev

    for drop, dname in [(128, "Italy"), (129, "Caesar"), (130, "Alexander"),
                        (7, "native-ctrl")]:
        ukt, evb = dropbank(drop)
        row = []
        for name, ids, tid in fprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            rk = int((lg > lg[tid]).sum()) + 1
            bp = dec(f["BP1"])[-1]
            row.append(f"{name}:r{rk}<-{int(np.argmax(bp))}")
        hold = 0
        broken = []
        for ids, tgt in cprompts:
            f = run(ids, ukt, evb)
            lg = dec(f["LOGITS"])[-1]
            top = int(lg.argmax())
            if top == base_tops[tuple(ids)]:
                hold += 1
            else:
                broken.append(inv.get(top, "?"))
        print(f"drop-{dname}: " + " ".join(row) + f" hold {hold}/16 "
              f"broken->[{','.join(broken[:6])}]", flush=True)
    # seriation: spectral ring order of the 131 keys (cosine graph)
    U = ukt3 / (np.linalg.norm(ukt3, axis=0, keepdims=True) + 1e-12)
    G = U.T @ U
    np.fill_diagonal(G, 0)
    D = np.diag(G.sum(1))
    L = D - G
    w, V = np.linalg.eigh(L)
    fied = V[:, 1]
    order = np.argsort(fied)
    ring = list(order)
    for i, ci in enumerate([128, 129, 130]):
        p = ring.index(ci)
        nb = [ring[(p + d) % len(ring)] for d in (-2, -1, 1, 2)]
        print(f"store{ci} ring-neighbors: {nb} "
              f"(cross-talk predicts geography here)", flush=True)


if __name__ == "__main__":
    main()
