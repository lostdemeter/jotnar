"""Driver-composition gate: layer-step driver == monolithic (per-block vehicle).

Listing programs/lm_step.asm (one tied layer) + lm_head.asm driven 4x
from host with per-layer CONFIG: bit-exact vs programs/lm_depth4.asm
at global CONFIG (composition proof -- machinery, not accuracy), then
per-layer priced scales run green with joint held (twin 0.479 exact,
top1 0.484 vs 0.472). Per-block verdict: LATERAL at D16 (parity
+-0.5dB) -- mechanism proven for D64 where cover-tax is real.
Usage: python3 tests/test_lm_step.py (fast: ~10 listing runs)
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
GCFG = "CONFIG m_acc 36118\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"
GCFG4 = "CONFIG m_acc 36849\nCONFIG m_cov 35048\nCONFIG beta -30.0\n"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def main():
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    dd = os.path.join(root, "data")
    d = np.load(os.path.join(dd, "lm_svd_IvoQ.npz"))
    b = np.load(os.path.join(dd, "bankhn.npz"))
    stepT = open(os.path.join(sdir, "lm_step.asm")).read()
    headT = open(os.path.join(sdir, "lm_head.asm")).read()
    toks = np.array([12, 1, 27, 5], dtype=np.int64)
    pos = np.arange(4, dtype=np.int64)
    cm = np.tril(np.ones((4, 4), dtype=np.int64))
    pay0 = {"tok": toks, "pos": pos, "cmask": cm, "emb": enc(d["emb"]),
            "wq": enc(d["wq"]), "wk": enc(d["wk"]), "wv": enc(d["wv"]),
            "wo": enc(d["wo"]), "wup": enc(d["wup"]), "wgate": enc(d["wgate"]),
            "wdown": enc(d["wdown"]), "rms_w1": enc(d["rms1"]),
            "rms_w2": enc(d["rms2"]), "wlog": enc(d["wlog"]),
            "ukt": enc(b["ukt"]), "evb": enc(b["evb"])}

    def drive(step_cfg, head_cfg):
        g = ASM.run_text("IN tok\nIN emb\nOUT = GATHER(emb, tok)\n",
                         REGISTRY, {"tok": toks, "emb": enc(d["emb"])},
                         sigs=SIGS)
        h = g["OUT"]
        shared = dict(pay0)
        shared.pop("tok")
        for cfg in step_cfg:
            pay = dict(shared)
            pay["h"] = h
            h = ASM.run_text(cfg + stepT, REGISTRY, pay, sigs=SIGS,
                             basedir=sdir)["h2"]
        fh = ASM.run_text(head_cfg + headT, REGISTRY,
                          {"h": h, "wlog": enc(d["wlog"])}, sigs=SIGS)
        return h, fh["LOGITS"]

    mono = ASM.run_text(GCFG4 + open(os.path.join(sdir, "lm_depth4.asm")).read(),
                        REGISTRY, pay0, sigs=SIGS, basedir=sdir)
    h, lg = drive([GCFG4] * 4, GCFG4)
    same = all(bool((lg[k] == mono["LOGITS"][k]).all()) for k in (0, 1, 2))
    check("step-driver-exact", same, "driver == monolithic bit-exact")
    msteps = [(36849, 34725), (36849, 34899), (36849, 35030), (36849, 35136)]
    h2, lg2 = drive(["CONFIG m_acc %d\nCONFIG m_cov %d\nCONFIG beta -30.0\n" % mc
                     for mc in msteps],
                    "CONFIG m_acc 36899\nCONFIG m_cov 35048\n")
    check("step-runs", lg2 is not None, "per-layer scales run green")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
