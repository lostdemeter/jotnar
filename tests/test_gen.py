"""Generation gate: the LLM speaks identically on all four backends.

Fixed prompt + 6 greedy UNK-masked steps on the D16 headt word model.
lattice == nonfpu is structural (same fixed-point math, must be exact);
c/cuda join by the leveled story (their float math is numpy-exact, and
the UNK mask keeps decisions off the lattice's ill-conditioned rows).
Golden string pins behavior against silent change.
Usage: python3 tests/test_gen.py (cuda valence ~1min: builds + steps).
"""
import os
import shutil
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []
GOLDEN = "alexander the great and the great and the great"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def gen(backend, seed="alexander the great", n=6):
    import demo_gen as G
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dd = os.path.join(root, "data")
    sdir = os.path.join(root, "programs")
    import json
    import numpy as np
    vocab = json.load(open(os.path.join(dd, "lm_vocab.json")))
    inv = {i: w for w, i in vocab.items()}
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    text = G.CFG + open(os.path.join(sdir, "lm_headt.asm")).read()
    trip = {"emb": G.enc(d["emb"]), "wq": G.enc(d["wq"]),
            "wk": G.enc(d["wk"]), "wv": G.enc(d["wv"]),
            "wo": G.enc(d["wo"]), "wup": G.enc(d["wup"]),
            "wgate": G.enc(d["wgate"]), "wdown": G.enc(d["wdown"]),
            "rms_w1": G.enc(d["rms1"]), "rms_w2": G.enc(d["rms2"]),
            "wlog": G.enc(d["wlog"]), "ukt": G.enc(b["ukt"]),
            "evb": G.enc(b["evb"])}
    work = f"/tmp/gen_test_{backend}"
    os.makedirs(work, exist_ok=True)
    if backend == "lattice":
        runner = G.LatticeRunner(text, trip, sdir)
    else:
        libs = [] if backend == "nonfpu" else "dflt"
        runner = G.ExeRunner(text, trip, sdir, work, "gen", backend, libs)
    ids = [vocab.get(w.lower(), 0) for w in seed.split()]
    out = list(ids)
    for _ in range(n):
        logits, _ = runner.step(out)
        logits = np.array(logits, dtype=np.float64)
        logits[0] = -np.inf
        out.append(int(np.argmax(logits)))
    return " ".join(inv.get(j, "<unk>") for j in out)


def main():
    outs = {}
    for backend in ("lattice", "c", "nonfpu"):
        outs[backend] = gen(backend)
        print(f"gen-{backend}: {outs[backend]}")
    check("gen-lattice-nonfpu-exact", outs["lattice"] == outs["nonfpu"],
          "same fixed-point math")
    check("gen-lattice-c-agree", outs["lattice"] == outs["c"],
          "conditioned decisions")
    check("gen-golden", outs["lattice"] == GOLDEN, GOLDEN[:60])
    if shutil.which("nvcc") is None:
        print("SKIP gen-cuda (no nvcc)")
    else:
        outs["cuda"] = gen("cuda")
        print(f"gen-cuda: {outs['cuda']}")
        check("gen-cuda-agree", outs["cuda"] == outs["lattice"], "gpu joins")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
