"""Demo: our transformer speaking (construction, honest machinery demo).

Seed ids -> sliding-window (<=8) through programs/lm_depth2tied.asm
(2x H=2 blocks, random frozen weights, no training) -> next-id loop.
Stated limits: weights are RANDOM (calibrated magnitudes only), attention
is NON-CAUSAL (no mask yet -- backlog), so output is ~uniform noise, not
fluency. The gate is machinery (parity 70dB+) + loop runs end-to-end;
fluency needs counts-injected or fitted weights (future construction).
Run: python3 demo_deep.py [seed words...] [--n 20] [--topk 12 --seed 7]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

WIN = 8


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    dd = os.path.join(root, "data")
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    d = np.load(os.path.join(dd, "lm_block1.npz"))
    text = open(os.path.join(root, "programs", "lm_depth2tied.asm")).read()
    sdir = os.path.join(root, "programs")
    words, n, i = [], 20, 1
    topk, seedn = 0, 7
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
            words.append(a)
            i += 1
    seed = " ".join(words) if words else "alexander the great in"

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    eb, wq, wk, wv, wo = (enc(d["emb"]), enc(d["wq"]), enc(d["wk"]),
                          enc(d["wv"]), enc(d["wo"]))
    wup, wgate, wdown = enc(d["wup"]), enc(d["wgate"]), enc(d["wdown"])
    r1, r2, wl = enc(d["rms1"]), enc(d["rms2"]), enc(d["wlog"])

    def step(ids):
        ctx = ids[-WIN:]
        pos = np.arange(len(ctx), dtype=np.int64)
        feeds = ASM.run_text(text, REGISTRY,
                             {"tok": np.array(ctx, np.int64), "pos": pos,
                              "emb": eb, "wq": wq, "wk": wk, "wv": wv, "wo": wo,
                              "wup": wup, "wgate": wgate, "wdown": wdown,
                              "rms_w1": r1, "rms_w2": r2, "wlog": wl},
                             sigs=SIGS, basedir=sdir)
        t = feeds["LOGITS"]
        logits = (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                  * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))
        return logits[-1], int(np.ascontiguousarray(feeds["OUT"]).reshape(-1)[-1])

    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    rng = np.random.default_rng(seedn)
    out = list(ids)
    for _ in range(n):
        logits, greedy = step(out)
        if topk > 0:
            keep = np.argsort(-logits)[:topk]
            w = np.zeros_like(logits)
            w[keep] = np.exp(logits[keep] - logits[keep].max())
            w = w / w.sum()
            out.append(int(rng.choice(len(w), p=w)))
        else:
            out.append(greedy)
    print("seed:", seed)
    print("out :", " ".join(inv.get(j, "<unk>") for j in out))
    print(f"({len(out)} tokens, depth-2 H=2, random weights, non-causal; "
          f"machinery demo, not fluency)")


if __name__ == "__main__":
    main()
