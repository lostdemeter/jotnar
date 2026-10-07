"""Search tiers: exact-hash vs hierarchical vs orthogonal bank.

Three ways to make the ball navigable, same 6-fact catalog, same keys:
A. EXACT tier: sign-quantized key dict (perfect-hit/no-false-hit on
   frozen keys; generality price measured on paraphrases).
B. HIERARCHICAL: data-driven clusters (coarse mean-key, then item
   keys); must match flat gating with fewer comparisons.
C. ORTHOGONAL bank: Gram-Schmidt directions; cross-talk must vanish
   (install quality separately priced -- whitening may cost dose).
Usage: python3 research/search_tiers.py
"""
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from qwen_torch import fwd, fwdH
from chain.qwen7b import load7b

COUNTRIES = ["France", "Germany", "Italy", "Spain", "Japan", "China"]
TMPL = "The capital of {c} is"
PARA = ["{c}'s capital is the city of", "What is the capital of {c}?"]


def key_of(prompt, country):
    t, ids = fwdH(prompt, keep="all")
    _, tok = load7b()
    toks = tok.convert_ids_to_tokens(ids)
    frag = country[1:].lower()
    pos = next((i for i, x in enumerate(toks) if frag in x.lower()),
               len(ids) - 1)
    v = t[2][pos]
    return v / np.linalg.norm(v)


def main():
    _, tok = load7b()
    prom = {c: TMPL.format(c=c) for c in COUNTRIES}
    base = {}
    for c in COUNTRIES:
        lg, _ = fwd(prom[c])
        base[c] = int(lg.argmax())
    print("unsteered:", {c: tok.decode([base[c]]) for c in COUNTRIES},
          flush=True)
    keys = {c: key_of(prom[c], c) for c in COUNTRIES}
    K = np.stack([keys[c] for c in COUNTRIES])
    print("key cosines (min off-diag):",
          round(float((K @ K.T)[np.triu_indices(6, 1)].min()), 3), flush=True)

    # -- A. exact tier: sign-quantized dict ---------------------------
    table = {np.packbits((keys[c] > 0).astype(np.uint8)).tobytes(): c
             for c in COUNTRIES}
    hits = sum(table.get(np.packbits((key_of(prom[c], c) > 0).astype(
        np.uint8)).tobytes()) == c for c in COUNTRIES)
    print(f"A exact-tier self-hits: {hits}/6 (expect 6)", flush=True)
    cross = 0
    for c in COUNTRIES:
        for p in COUNTRIES:
            if p == c:
                continue
            if table.get(np.packbits((key_of(prom[p], p) > 0).astype(
                    np.uint8)).tobytes()) == c:
                cross += 1
    print(f"A exact-tier cross-hits: {cross}/30 (expect 0)", flush=True)
    pmiss = 0
    for c in COUNTRIES:
        for t in PARA:
            if table.get(np.packbits((key_of(t.format(c=c), c) > 0).astype(
                    np.uint8)).tobytes()) != c:
                pmiss += 1
    print(f"A paraphrase misses: {pmiss}/12 (the generality price)",
          flush=True)

    # -- B. hierarchical: data-driven clusters ------------------------
    import itertools
    best, bestc = 1e9, None
    for a in range(1, 1 << 5):
        g0 = [COUNTRIES[0]] + [COUNTRIES[i + 1] for i in range(5) if a >> i & 1]
        g1 = [c for c in COUNTRIES if c not in g0]
        if not g1:
            continue
        m0 = np.mean([keys[c] for c in g0], axis=0)
        m1 = np.mean([keys[c] for c in g1], axis=0)
        s = sum(1 for c in g0 if keys[c] @ m0 < keys[c] @ m1)
        s += sum(1 for c in g1 if keys[c] @ m1 < keys[c] @ m0)
        if s < best:
            best, bestc = s, (g0, g1)
    print(f"B clusters: {bestc} misassigned={best}", flush=True)
    m = {0: np.mean([keys[c] for c in bestc[0]], axis=0),
         1: np.mean([keys[c] for c in bestc[1]], axis=0)}
    ok = 0
    for c in COUNTRIES:
        g = 0 if keys[c] @ m[0] > keys[c] @ m[1] else 1
        sub = bestc[g]
        pick = max(sub, key=lambda x: keys[c] @ keys[x])
        ok += pick == c
    print(f"B hier item accuracy: {ok}/6 (flat acids match + fewer cmps)",
          flush=True)

    # -- C. orthogonal bank -------------------------------------------
    trajs = {c: fwdH(prom[c]) for c in COUNTRIES}
    raw = {}
    for c in COUNTRIES:
        others = [trajs[o][0][27] for o in COUNTRIES if o != c]
        d = trajs[c][0][27] - np.mean(others, axis=0)
        raw[c] = d / np.linalg.norm(d)
    D = np.stack([raw[c] for c in COUNTRIES])
    Q, _ = np.linalg.qr(D.T)
    orth = {c: Q[:, i] / np.linalg.norm(Q[:, i]) for i, c in enumerate(COUNTRIES)}
    O = np.stack([orth[c] for c in COUNTRIES])
    print("C max off-diag |cos| raw vs orth:",
          round(float(np.abs((D @ D.T)[np.triu_indices(6, 1)]).max()), 3),
          round(float(np.abs((O @ O.T)[np.triu_indices(6, 1)]).max()), 3),
          flush=True)
    exp = {}
    for c in COUNTRIES:
        lg, _ = fwd(prom[c])
        exp[c] = int(lg.argmax())
    for c in ("Spain", "China"):
        lg, _ = fwd(prom[c], steer=(27, orth[c], 0.8))
        rk = int((lg > lg[exp[c]]).sum()) + 1
        print(f"C orth-install {c}: rank={rk} "
              f"(raw-diagonal was rank-1; whitening price shown)", flush=True)


if __name__ == "__main__":
    main()
