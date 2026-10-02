"""Twin-v2 gate: trajectory-aware comparison over voice pairs (Gate 4 instrument).

Endpoint twins compare full-content states (order/length differ);
trajectory twins compare content-ALIGNED states across all H rows
(same piece, histories differ) -- together they triangulate
content-invariance the way census+ranges corroborate (neither is
pure: endpoints carry order/length residue ~0.2, aligned pairs
carry history mismatch; measured v03 0.665/0.464 vs v02
0.335/0.651 -- neither bounds the other). Alignment rule (stated):
first-unused identical piece id; unaligned skipped; pair needs
>=3 aligned. Bars from first measurement + margin (LIB-015):
endpoint-mean < 0.50, endpoint-max < 0.75, traj-mean < 0.60,
traj-max < 0.75, min-aligned >= 3 (P32 fit stack).
Usage: python3 tests/test_twin_traj.py (fast: 16 listing runs)
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

FAIL = []
CFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\nCONFIG beta_b 0.25\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    vocab = json.load(open(os.path.join(dd, "bpe_vocab.json")))
    merges = json.load(open(os.path.join(dd, "bpe_merges.json")))
    rank = {tuple(m): i for i, m in enumerate(merges)}

    def encode(s):
        out = []
        for w in re.findall(r"[a-z0-9']+", s.lower()):
            syms = [c for c in w] + ["</w>"]
            while len(syms) > 1:
                best = None
                for i in range(len(syms) - 1):
                    r = rank.get((syms[i], syms[i + 1]))
                    if r is not None and (best is None or r < best[0]):
                        best = (r, i)
                if best is None:
                    break
                _, i = best
                syms = syms[:i] + [syms[i] + syms[i + 1]] + syms[i + 2:]
            out.extend(vocab[p] for p in syms)
        return out

    w = np.load(os.path.join(dd, "lm_piece32_fit.npz"))
    Ep = np.load(os.path.join(dd, "lm_piece32.npz"))
    b = np.load(os.path.join(dd, "bankp32_64.npz"))
    text = CFG + open(os.path.join(sdir, "lm_d32_headt.asm")).read()

    def run_H(ids):
        toks = np.array(ids, dtype=np.int64)
        pos = np.arange(len(ids), dtype=np.int64)
        cm = np.tril(np.ones((len(ids), len(ids)), dtype=np.int64))
        f = ASM.run_text(text, REGISTRY,
                         {"tok": toks, "pos": pos, "cmask": cm,
                          "emb": enc(Ep["emb"]), "wq": enc(w["wq"]), "wk": enc(w["wk"]),
                          "wv": enc(w["wv"]), "wo": enc(w["wo"]),
                          "wup": enc(w["wup"]), "wgate": enc(w["wgate"]),
                          "wdown": enc(w["wdown"]),
                          "rms_w1": enc(w["rms1"]), "rms_w2": enc(w["rms2"]),
                          "wlog": enc(Ep["wlog"]),
                          "ukt": enc(b["ukt"]), "evb": enc(b["evb"])},
                         sigs=SIGS, basedir=sdir)
        return dec(f["H4"])

    voice = json.load(open("/home/thorin/Documents/OpenCode/Echion_Revisted/data/voice_pairs.json"))
    eps, trs, als = [], [], []
    for p in voice:
        A, B = encode(p["active"]), encode(p["passive"])
        HA, HB = run_H(A), run_H(B)
        ep = float(np.linalg.norm(HA[-1] - HB[-1]) / max(np.linalg.norm(HA[-1]), 1e-12))
        used, pairs = set(), []
        for i, a in enumerate(A):
            for j, b_ in enumerate(B):
                if b_ == a and j not in used:
                    pairs.append((i, j))
                    used.add(j)
                    break
        ds = [float(np.linalg.norm(HA[i] - HB[j])
                    / max(np.linalg.norm(HA[i]), np.linalg.norm(HB[j]), 1e-12))
              for i, j in pairs]
        als.append(len(pairs))
        eps.append(ep)
        trs.append(float(np.mean(ds)))
    import statistics as st
    check("traj-endpoint-mean", st.mean(eps) < 0.50, f"mean {st.mean(eps):.3f}")
    check("traj-endpoint-max", max(eps) < 0.75, f"max {max(eps):.3f}")
    check("traj-aligned-mean", st.mean(trs) < 0.60, f"mean {st.mean(trs):.3f}")
    check("traj-aligned-max", max(trs) < 0.75, f"max {max(trs):.3f}")
    check("traj-coverage", min(als) >= 3, f"min aligned {min(als)}/8 pairs")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
