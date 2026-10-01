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
        if not pats:
            return np.ones(64), len(words)
        cue = np.sign(sum(np.roll(p, k) for k, p in pats)).astype(float)
        cue[cue == 0] = 1.0
        return cue, len(words) - len(pats)

    cue, skipped = cue_of(words_of(" ".join(sys.argv[1:])))
    # banked retrieval: curated 55 + mined 202 + w103 (dot-arbitrate,
    # narrow wins ties -- demo_banks policy; full-triple cue shape)
    import hashlib as _hl
    banks = []
    # curated
    f = ASM.run_text(assoc, REGISTRY,
                     {"cue": enc(cue.reshape(1, -1)),
                      "keys": enc(keys.T.copy()),
                      "values": enc(z["values"])},
                     sigs=SIGS, basedir=sdir)
    _ = f
    sims = keys @ cue
    banks.append(("curated", float(sims.max()), None))
    # mined 202 (hash-lex cue)
    wpatM = dict(wpat)
    mined = [json.loads(l) for l in open(os.path.join(dd, "mined_edges.jsonl"))][1:]
    for w in sorted({x for e in mined for ff in ("subj", "pred", "obj")
                     for x in words_of(e[ff])} - set(wpatM)):
        h = int(_hl.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
        wpatM[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=64)
    zM = np.load(os.path.join(dd, "mined_keys.npz"))
    patsM = [(k, wpatM[w]) for k, w in enumerate(words_of(" ".join(sys.argv[1:]))) if w in wpatM]
    if patsM:
        cueM = np.sign(sum(np.roll(p, k) for k, p in patsM)).astype(float)
        cueM[cueM == 0] = 1.0
        simsM = zM["keys"] @ cueM
        kiM = int(np.argmax(simsM))
        acc = 0
        recM = None
        for r in mined:
            n = 1
            if kiM < acc + n:
                recM = r
                break
            acc += n
        banks.append(("mined", float(simsM.max()), recM))
    # w103 (hash-lex cue)
    wpatW = dict(wpat)
    w103e = [json.loads(l) for l in open(os.path.join(dd, "banks", "w103_edges.jsonl"))][1:]
    for w in sorted({x for e in w103e for ff in ("subj", "pred", "obj")
                     for x in words_of(e[ff])} - set(wpatW)):
        h = int(_hl.sha256(f"lex99:{w}".encode()).hexdigest()[:16], 16)
        wpatW[w] = np.random.default_rng(h).choice([-1.0, 1.0], size=64)
    zW = np.load(os.path.join(dd, "banks", "w103_keys.npz"))
    patsW = [(k, wpatW[w]) for k, w in enumerate(words_of(" ".join(sys.argv[1:]))) if w in wpatW]
    if patsW:
        cueW = np.sign(sum(np.roll(p, k) for k, p in patsW)).astype(float)
        cueW[cueW == 0] = 1.0
        simsW = zW["keys"] @ cueW
        kiW = int(np.argmax(simsW))
        recW = w103e[kiW] if kiW < len(w103e) else None
        banks.append(("w103", float(simsW.max()), recW))
    banks.sort(key=lambda t: -t[1])
    win_tag, win_score, win_rec = banks[0]
    # curated rec: key->edge walk over n_keys records (original path)
    ki = int(np.argmax(keys @ cue))
    acc = 0
    rec = recs[0]
    for r in recs:
        if ki < acc + r["n_keys"]:
            rec = r
            break
        acc += r["n_keys"]
    if win_rec is not None and win_tag != "curated":
        rec = {"id": win_rec.get("id", "?"), "tag": f"{win_tag}/{win_rec.get('id', '?')}",
               "subj": win_rec["subj"], "pred": win_rec["pred"],
               "obj": win_rec["obj"], "canon": words_of(win_rec["subj"]) + words_of(win_rec["pred"]) + words_of(win_rec["obj"])}
    print(f"banks: {[(t, round(s, 1)) for t, s, _ in banks]} -> {win_tag}", flush=True)
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
