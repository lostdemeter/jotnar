"""Selection demo: candidates scored by corpus fitness (Echion pattern).

Population: --cand N sampled continuations (seeded, replayable) from the
bankhn2 stack. Fitness (all frozen, all stated): bigram-hit (fraction of
output bigrams with count>0 in frozen counts -- our bloom equivalent,
exact), repeat-penalty (1 - unique-ratio shortfall), len-penalty
(Gricean quantity: |len - target|/target). Winner = max fitness.
Listings untouched; selection is host-boundary (same doctrine as
temperature/top-k). Run: python3 demo_select.py [seeds] [--cand 6]
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


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

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
        f = ASM.run_text(text, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          **P}, sigs=SIGS, basedir=sdir)
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    args, n, i = [], 20, 1
    topk, seedn, cand, nrep, gpen = 12, 7, 6, 3, 0.5
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
        elif a == "--cand" and i + 1 < len(sys.argv):
            cand = int(sys.argv[i + 1])
            i += 2
        elif a == "--gpen" and i + 1 < len(sys.argv):
            gpen = float(sys.argv[i + 1])
            i += 2
        else:
            args.append(a)
            i += 1
    seed = " ".join(args) if args else "alexander the great"
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    _GLUEIDS = {vocab[w] for w in ("the", "and", "of", "in", "a", "to",
                "with", "as", "for", "on", "by", "at", "is", "was") if w in vocab}
    try:
        attested = set(json.load(open(os.path.join(dd, "attested.json")))["attested_shapes"])
    except Exception:
        attested = set()

    def fitness(out):
        bh = sum(1 for x, y in zip(out[:-1], out[1:]) if counts[x, y] > 0)
        bloom = bh / max(len(out) - 1, 1)
        u = len(set(out)) / max(len(out), 1)
        rep = min(u / 0.8, 1.0)
        q = 1.0 - abs(len(out) - (len(ids) + n)) / max(len(ids) + n, 1)
        ws = [inv.get(j, "<unk>") for j in out]
        sh = 0
        for i in range(len(ws) - 2):
            for pat in (f"{ws[i]} {ws[i+1]} W", f"W {ws[i+1]} {ws[i+2]}"):
                if pat in attested:
                    sh += 1
        shape = min(sh / max(len(ws) - 2, 1) / 2.0, 1.0)
        _GLUE = {"the", "and", "of", "in", "a", "to", "with", "as", "for",
                 "on", "by", "at", "from", "is", "was", "were", "are", "be",
                 "it", "that", "this", "an", "or", "his", "her", "its"}
        _gr = sum(1 for x in ws if x in _GLUE) / max(len(ws), 1)
        glue = 1.0 - min(abs(_gr - 0.24) / 0.30, 1.0)
        return (0.35 * bloom + 0.20 * rep + 0.15 * q + 0.15 * shape
                + 0.15 * glue), bloom, rep, q, shape, glue

    best, best_f = None, -1.0
    for c in range(max(cand, 1)):
        rng = np.random.default_rng(seedn * 100003 + c)
        out = list(ids)
        for _ in range(n):
            lg = logits_of(out).copy()
            lg[0] = -1e9
            for _j in _GLUEIDS:
                lg[_j] -= gpen
            if len(out) >= nrep - 1:
                seen = {tuple(out[k:k + nrep]) for k in range(len(out) - nrep + 1)}
                prefix = tuple(out[-(nrep - 1):]) if nrep > 1 else ()
                for j in range(len(lg)):
                    if prefix + (j,) in seen:
                        lg[j] = -1e9
            keep = np.argsort(-lg)[:topk]
            w = np.zeros_like(lg)
            w[keep] = np.exp(lg[keep] - lg[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        f, bh, rp, q, sh, gl = fitness(out)
        if f > best_f:
            best, best_f, best_parts = list(out), f, (bh, rp, q, sh, gl)
    print(f"seed: {seed} (cand={cand} fitness={best_f:.3f} "
          f"bloom={best_parts[0]:.3f} rep={best_parts[1]:.3f} q={best_parts[2]:.3f} shape={best_parts[3]:.3f} glue={best_parts[4]:.3f})")
    print("out :", " ".join(inv.get(j, "<unk>") for j in best))


if __name__ == "__main__":
    main()
