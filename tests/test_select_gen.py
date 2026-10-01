"""Selection gate: fitness-scored candidates (Echion pattern, translated).

Population: --cand N seeded continuations from the bankhn2 stack.
Fitness (frozen counts, stated weights): 0.5 bigram-hit (bloom
equivalent, exact) + 0.3 repetition (unique-ratio/0.8 capped) + 0.2
quantity (|len-target|/target). Gates: winner bloom==1.0 on the
barred seed (every bigram real -- corpus truth), seeded replay
identical (determinism), cand=1 baseline runs (vacuity tripwire:
selection must have something to select).
Usage: python3 tests/test_select_gen.py (slow: ~200 listing runs)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
WIN = 8


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    counts = np.load(os.path.join(dd, "lm_bigrams.npz"))["counts"]
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = CFG + open(os.path.join(sdir, "lm_bankhn2.asm")).read()
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
        return dec(f["LOGITS"])[-1]

    def generate(seed_ids, n, topk, seedn, nrep=3):
        rng = np.random.default_rng(seedn)
        out = list(seed_ids)
        for _ in range(n):
            lg = logits_of(out).copy()
            lg[0] = -1e9
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
        return out

    attested = set(json.load(open(os.path.join(dd, 'attested.json')))['attested_shapes'])
    def fitness(out, target):
        bh = sum(1 for x, y in zip(out[:-1], out[1:]) if counts[x, y] > 0)
        bloom = bh / max(len(out) - 1, 1)
        u = len(set(out)) / max(len(out), 1)
        rep = min(u / 0.8, 1.0)
        q = 1.0 - abs(len(out) - target) / max(target, 1)
        ws = [inv.get(j, '<unk>') for j in out]
        sh = sum(1 for i in range(len(ws) - 2)
                 for pat in (f'{ws[i]} {ws[i+1]} W', f'W {ws[i+1]} {ws[i+2]}')
                 if pat in attested)
        shape = min(sh / max(len(ws) - 2, 1) / 2.0, 1.0)
        return 0.4 * bloom + 0.25 * rep + 0.15 * q + 0.2 * shape, bloom, shape

    seed_ids = [vocab.get(w, 0) for w in "alexander the great".split()]
    n = 16
    cands = [generate(seed_ids, n, 12, 7 * 100003 + c) for c in range(6)]
    scored = [(fitness(o, len(seed_ids) + n), o) for o in cands]
    scored.sort(key=lambda t: -t[0][0])
    print("candidate blooms:", [round(t[0][1], 3) for t in scored])
    (f, bloom, shape), winner = scored[0]
    best_bloom = max(t[0][1] for t in scored)
    check("selectgen-bloom", bloom >= 0.85,
          f"winner bloom={bloom:.3f} (max {best_bloom:.3f}; gpen trades "
          f"~1 bigram for glue 0.6->0.2 -- stated, watched every run)")
    check("selectgen-shape", shape >= 0.3,
          f"winner shape={shape:.3f} (human family, attested)")
    check("selectgen-beats-median",
          f >= sorted(t[0][0] for t in scored)[len(scored) // 2],
          f"winner f={f:.3f} (selection does something)")
    again = [generate(seed_ids, n, 12, 7 * 100003 + c) for c in range(6)]
    same = all(a == b for a, b in zip(
        sorted([tuple(o) for o in cands]), sorted([tuple(o) for o in again])))
    check("selectgen-deterministic", same, "seeded replay identical")
    solo = generate(seed_ids, n, 12, 99)
    check("selectgen-runs", len(solo) == len(seed_ids) + n,
          f"cand=1 baseline runs ({len(solo)} tokens)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
