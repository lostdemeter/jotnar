"""Demo: flagship composition (depth-4 + fusion + selection + guides).

Everything proven, one command: curated retrieval prepend -> depth-4
causal transformer (top1 0.472 stack) -> candidates scored by
bloom/rep/quantity/shape/glue fitness -> winner. Guides (host
boundary, declared): UNK-mask, no-repeat-4, gpen 0.5, topk 12.
Listings untouched. Run: python3 demo_flagship.py [seeds] [--cand 6]
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

CFG = "CONFIG m_acc 36849\nCONFIG m_cov 35048\n"
WIN = 8
GLUE = {"the", "and", "of", "in", "a", "to", "with", "as", "for", "on",
        "by", "at", "is", "was"}


def words_of(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    try:
        attested = set(json.load(open(os.path.join(dd, "attested.json")))["attested_shapes"])
    except Exception:
        attested = set()
    # retrieval side (curated 55)
    zw = np.load(os.path.join(dd, "edge_wordpats.npz"))
    wpat = {w: zw[f"w{i}"] for i, w in enumerate(list(zw["lex"]))}
    z = np.load(os.path.join(dd, "edge_keys.npz"))
    keys = z["keys"]
    recs = [json.loads(l) for l in open(os.path.join(dd, "edge_records.jsonl"))][1:]
    assoc = open(os.path.join(sdir, "assoc_mem.asm")).read()
    # generation side (depth-4)
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    lmtext = CFG + open(os.path.join(sdir, "lm_depth4.asm")).read()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    P = {"emb": enc(d["emb"]), "wq": enc(d["wq"]), "wk": enc(d["wk"]),
         "wv": enc(d["wv"]), "wo": enc(d["wo"]),
         "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
         "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
         "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
         "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}
    gids = {vocab[w] for w in GLUE if w in vocab}

    def retrieve(ctx_words):
        pats = [(k, wpat[w]) for k, w in enumerate(ctx_words) if w in wpat]
        if not pats:
            return []
        cue = np.sign(sum(np.roll(p, k) for k, p in pats)).astype(float)
        cue[cue == 0] = 1.0
        ki = int(np.argmax(keys @ cue))
        acc = 0
        for r in recs:
            if ki < acc + r["n_keys"]:
                return words_of(r["subj"]) + words_of(r["pred"]) + words_of(r["obj"])
            acc += r["n_keys"]
        return []

    def logits_of(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        cm = np.tril(np.ones((len(ctx), len(ctx)), dtype=np.int64))
        f = ASM.run_text(lmtext, REGISTRY,
                         {"tok": np.array(ctx, np.int64), "pos": pos, "cmask": cm,
                          **P}, sigs=SIGS, basedir=sdir)
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    def fitness(out, target):
        bh = sum(1 for x, y in zip(out[:-1], out[1:]) if counts[x, y] > 0)
        bloom = bh / max(len(out) - 1, 1)
        u = len(set(out)) / max(len(out), 1)
        rep = min(u / 0.8, 1.0)
        q = 1.0 - abs(len(out) - target) / max(target, 1)
        ws = [inv.get(j, "<unk>") for j in out]
        sh = sum(1 for i in range(len(ws) - 2)
                 for pat in (f"{ws[i]} {ws[i+1]} W", f"W {ws[i+1]} {ws[i+2]}")
                 if pat in attested)
        shape = min(sh / max(len(ws) - 2, 1) / 2.0, 1.0)
        gr = sum(1 for x in ws if x in GLUE) / max(len(ws), 1)
        glue = 1.0 - min(abs(gr - 0.24) / 0.30, 1.0)
        return (0.35 * bloom + 0.20 * rep + 0.15 * q + 0.15 * shape
                + 0.15 * glue), bloom, rep, shape, glue

    args, n, i = [], 18, 1
    topk, seedn, cand, nrep, gpen = 12, 7, 6, 4, 0.5
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
            i += 2
        elif a == "--cand" and i + 1 < len(sys.argv):
            cand = int(sys.argv[i + 1])
            i += 2
        elif a == "--seed" and i + 1 < len(sys.argv):
            seedn = int(sys.argv[i + 1])
            i += 2
        else:
            args.append(a)
            i += 1
    seed = " ".join(args) if args else "alexander the great"
    pre = [vocab.get(w, 0) for w in retrieve(words_of(seed))]
    if any(j == 0 for j in pre):
        print("(retrieval unspeakable in 512-vocab, seed-only)", flush=True)
        pre = []
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    base = pre + ids
    target = len(base) + n
    best, best_f, best_parts = None, -1.0, None
    for c in range(max(cand, 1)):
        rng = np.random.default_rng(seedn * 100003 + c)
        out = list(base)
        for _ in range(n):
            lg = logits_of(out).copy()
            lg[0] = -1e9
            for j in gids:
                lg[j] -= gpen
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
        f, bh, rp, sh, gl = fitness(out, target)
        if f > best_f:
            best, best_f, best_parts = list(out), f, (bh, rp, sh, gl)
    print(f"seed: {seed} (cand={cand} f={best_f:.3f} bloom={best_parts[0]:.3f} "
          f"rep={best_parts[1]:.3f} shape={best_parts[2]:.3f} glue={best_parts[3]:.3f})")
    print("out :", " ".join(inv.get(j, "<unk>") for j in best))


if __name__ == "__main__":
    main()
