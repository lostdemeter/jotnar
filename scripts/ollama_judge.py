"""Ollama-judge probe (offline): does external ranking agree with fitness?

Fixed candidate set (6 seeded continuations, bankhn2 stack) -> fitness
order (frozen counts) vs ollama order (prompted rank, best-first) ->
Spearman agreement. If agreement is high, ollama can pre-filter (cheap)
with fitness as final gate; if low, ollama judges nothing here.
Ollama NEVER gates, NEVER writes listings/weights (propose-only).
Usage: python3 scripts/ollama_judge.py [--model qwen2:latest] [--seed ...]
"""
import json
import os
import re
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DD = os.path.join(ROOT, "data")
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\n"
WIN = 8


def ollama_rank(model, seed, cands, timeout=300):
    prompt = (f"Rank these {len(cands)} continuations of the historical "
              f"seed '{seed}' by fluency and coherence, best first. Reply "
              f"with ONLY comma-separated numbers, e.g. 3,1,2.\n" +
              "".join(f"\n{i + 1}. {' '.join(c)}" for i, c in enumerate(cands)))
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt,
                         "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.load(r)["response"]
    except Exception as e:
        return None, f"ollama error: {e}"
    nums = [int(x) for x in re.findall(r"\d+", out)][:len(cands)]
    if sorted(nums) != list(range(1, len(cands) + 1)):
        return None, f"unparseable rank: {out[:120]!r}"
    return nums, out[:120]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2:latest")
    ap.add_argument("--seed", default="alexander the great")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cand", type=int, default=6)
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    vocab = json.load(open(os.path.join(DD, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    counts = np.load(os.path.join(DD, "lm_bigrams.npz"))["counts"]
    d = np.load(os.path.join(DD, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(DD, "bankhn.npz"))
    text = CFG + open(os.path.join(ROOT, "programs", "lm_bankhn2.asm")).read()

    def enc(x):
        return S.encode(np.ascontiguousarray(x, dtype=np.float64))

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
                         {"tok": np.array(ctx, np.int64), "pos": pos,
                          "cmask": cm, **P}, sigs=SIGS,
                         basedir=os.path.join(ROOT, "programs"))
        t = f["LOGITS"]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))[-1]

    ids = [vocab.get(w.lower(), 0) for w in a.seed.split()]
    cands = []
    for c in range(a.cand):
        rng = np.random.default_rng(7 * 100003 + c)
        out = list(ids)
        tk = [3, 5, 12, 12, 20, 30][c % 6] if a.spread else 12
        for _ in range(a.n):
            lg = logits_of(out).copy()
            lg[0] = -1e9
            keep = np.argsort(-lg)[:tk]
            w = np.zeros_like(lg)
            w[keep] = np.exp(lg[keep] - lg[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        cands.append([inv.get(j, "<unk>") for j in out])

    def fitness(ws):
        toks = [vocab.get(w, 0) for w in ws]
        bh = sum(1 for x, y in zip(toks[:-1], toks[1:]) if counts[x, y] > 0)
        return bh / max(len(toks) - 1, 1)

    fit = [fitness(c) for c in cands]
    fit_order = sorted(range(a.cand), key=lambda i: -fit[i])
    print("fitness order:", [i + 1 for i in fit_order],
          [round(fit[i], 3) for i in fit_order])
    order, note = ollama_rank(a.model, a.seed, cands)
    if order is None:
        print(note)
        return
    print("ollama order:", order, note)
    # Spearman between fitness ranks and ollama ranks
    fr = {c: r for r, c in enumerate(fit_order)}
    or_ = {c: r for r, c in enumerate([x - 1 for x in order])}
    n = a.cand
    d2 = sum((fr[i] - or_[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    print(f"agreement Spearman rho={rho:.2f} "
          f"({'USEFUL pre-filter' if rho >= 0.5 else 'judges nothing here'})")


if __name__ == "__main__":
    main()
