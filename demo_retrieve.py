"""Demo: retrieve-then-generate (path C bridge, construction).

Context words -> order-sensitive cue (freeze_edges combiner) -> assoc_mem
recall -> edge record words prepended -> causal transformer generates.
Stated limits: cue uses edge-lexicon intersection (context words outside
the 138-lexicon are skipped, reported); retrieval exact on key sentences,
nearest otherwise; transformer weights SVD (top1 0.312 class).
Run: python3 demo_retrieve.py [context words...] [--n 12]
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

DIM = 64
CFG = "CONFIG m_acc 35492\nCONFIG m_cov 35048\n"


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    lex = list(zw["lex"])
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(lex)}
    z = np.load(os.path.join(dd, "edge_keys.npz"))
    keys, values = z["keys"], z["values"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))]
    edges = recs[1:]
    d = np.load(os.path.join(dd, "lm_svd.npz"))
    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()
    lmtext = CFG + open(os.path.join(sdir, "lm_depth2causal.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def cue_of(words):
        pats = [(k, wpat[w]) for k, w in enumerate(words) if w in wpat]
        skipped = len(words) - len(pats)
        cue = np.sign(sum(np.roll(p, k) for k, p in pats))
        cue[cue == 0] = 1.0
        return cue, skipped, len(pats)

    def retrieve(cue):
        f = ASM.run_text(assoc, REGISTRY,
                         {"cue": enc(cue.reshape(1, -1)),
                          "keys": enc(keys.T.copy()), "values": enc(values)},
                         sigs=SIGS, basedir=sdir)
        got = np.asarray(f["OUT"][0]).reshape(-1) > 0
        sims = keys @ cue
        return int(np.argmax(sims)), got

    eb, wq, wk, wv, wo = (enc(d["emb"]), enc(d["wq"]), enc(d["wk"]),
                          enc(d["wv"]), enc(d["wo"]))
    wup, wgate, wdown = enc(d["wup"]), enc(d["wgate"]), enc(d["wdown"])
    r1, r2, wl = enc(d["rms1"]), enc(d["rms2"]), enc(d["wlog"])

    def step(ids):
        ctx = ids[-8:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(lmtext, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          "emb": eb, "wq": wq, "wk": wk, "wv": wv, "wo": wo,
                          "wup": wup, "wgate": wgate, "wdown": wdown,
                          "rms_w1": r1, "rms_w2": r2, "wlog": wl},
                         sigs=SIGS, basedir=sdir)
        return int(np.ascontiguousarray(f["OUT"]).reshape(-1)[-1])

    args, n, i = [], 12, 1
    while i < len(sys.argv):
        if sys.argv[i] == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        else:
            args.append(sys.argv[i])
            i += 1
    seed = " ".join(args) if args else "caesar conquered gaul"
    ctx_words = words_of(seed)
    cue, skipped, used = cue_of(ctx_words)
    bi, got = retrieve(cue)
    # key -> edge: walk records' key counts (keys frozen in record order)
    sims = keys @ cue
    ki = int(np.argmax(sims))
    ei, acc = 0, 0
    for r in edges:
        if ki < acc + r["n_keys"]:
            ei = edges.index(r)
            break
        acc += r["n_keys"]
    rec = edges[ei]
    retr_words = rec["canon"]
    print(f"context: {seed} (cue from {used} lexicon words, skipped {skipped})")
    print(f"retrieved: {rec['id']} [{rec['tag']}] subj={rec['subj']!r} "
          f"pred={rec['pred']!r} obj={rec['obj']!r}")
    pre = [vocab.get(w, 0) for w in retr_words]
    gen = [vocab.get(w, 0) for w in ctx_words]
    out = pre + gen
    for _ in range(n):
        nxt = step(out)
        out.append(nxt)
        if nxt == 0:
            break
    print("seed(prepended):", " ".join(inv.get(j, "<unk>") for j in pre))
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))
    print(f"({len(out)} tokens, retrieve-then-generate, SVD weights)")


if __name__ == "__main__":
    main()
