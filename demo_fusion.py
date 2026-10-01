"""Demo: full functional system (retrieve + bankhn2 + boundary rules).

Word-cue -> assoc recall -> edge words prepended -> bankhn2 causal LM
with UNK-mask + no-repeat blocking + topk sampling (all host-boundary
decoding rules; listings untouched). The most functional composition
of everything built this session.
Run: python3 demo_fusion.py [seed words...] [--n 20] [--topk 12 --seed 7]
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
WIN = 8
NREP = 3


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    # retrieval side (word-cue bridge, exact)
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    z = np.load(os.path.join(dd, "edge_keys.npz"))
    keys = z["keys"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()
    # generation side (bankhn2 stack)
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    lmtext = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def cue_of(words):
        pats = [(k, wpat[w]) for k, w in enumerate(words) if w in wpat]
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        return cue, len(words) - len(pats)

    cue, skipped = cue_of(words_of(" ".join(sys.argv[1:])))
    f = ASM.run_text(assoc, REGISTRY,
                     {"cue": enc(cue.reshape(1, -1)),
                      "keys": enc(keys.T.copy()),
                      "values": enc(z["values"])},
                     sigs=SIGS, basedir=sdir)
    _ = f
    ki = int(np.argmax(keys @ cue))
    acc = 0
    rec = recs[0]
    for r in recs:
        if ki < acc + r["n_keys"]:
            rec = r
            break
        acc += r["n_keys"]
    P = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
         "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
         "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def logits_of(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        g = ASM.run_text(lmtext, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          **P}, sigs=SIGS, basedir=sdir)
        t = g["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    args, n, i, topk, seedn = [], 20, 1, 12, 7
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--topk" and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        else:
            args.append(a)
            i += 1
    seed = " ".join(args) if args else "caesar conquered gaul"
    pre = [vocab.get(w, 0) for w in rec["canon"]]
    out = pre + [vocab.get(w, 0) for w in words_of(seed)]
    rng = np.random.default_rng(seedn)
    for _ in range(n):
        lg = logits_of(out).copy()
        lg[0] = -1e9
        if len(out) >= NREP - 1:
            seen = {tuple(out[k:k + NREP]) for k in range(len(out) - NREP + 1)}
            prefix = tuple(out[-(NREP - 1):]) if NREP > 1 else ()
            for j in range(len(lg)):
                if prefix + (j,) in seen:
                    lg[j] = -1e9
        keep = np.argsort(-lg)[:topk]
        w = np.zeros_like(lg)
        w[keep] = np.exp(lg[keep] - lg[keep].max())
        w = w / w.sum()
        out.append(int(rng.choice(len(w), p=w)))
    print(f"retrieved: {rec['id']} [{rec['tag']}] "
          f"subj={rec['subj']!r} pred={rec['pred']!r} obj={rec['obj']!r}")
    print("fact:", " ".join(inv.get(j, "<unk>") for j in pre))
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))
    print(f"({len(out)} tokens, retrieve+generate, no-repeat {NREP})")


if __name__ == "__main__":
    main()
