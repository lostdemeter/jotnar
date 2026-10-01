"""Pruning demo (CRUD research Q1): delete by prediction, confirm in one run.

All 27 predicted-dead directions (docs/readout_down_896.csv, pred>55dB)
removed in ONE edit; combined cost predicted STATICALLY from linearity
(sum of per-direction delta vectors, no runs); single listing run confirms
inside +/-3dB (paper-first band, wider than single-dir 0.3dB: 27 components
accumulate quantum + the float-vs-lattice gap).
SKIPs without the local HF cache (needs-hardware precedent).
Usage: python3 tests/test_prune.py (~2 listing runs, seconds)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

FAIL = []
PROMPT = "The capital of France is Paris, and the capital of Germany is"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    from chain.qwen_mirror import load_layer0, build_H, mlp_forward, QWEN
    if not os.path.isfile(os.path.join(QWEN, "model.safetensors")):
        print("SKIP (needs Qwen2-0.5B in local HF cache)")
        sys.exit(0)
    try:
        import phi_core.lattice as S
        from chain import asm as ASM
        from chain.asm_ops import REGISTRY, SIGS
    except ImportError as e:
        print(f"SKIP ({e})")
        sys.exit(0)

    g, embed, tok = load_layer0()
    H, ids, toks = build_H(g, embed, tok, PROMPT)
    REF, MID, Wupf, Wgf, Wdf, ln2f = mlp_forward(H, g)

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    dead = []
    with open(os.path.join(root, "docs", "readout_down_896.csv")) as f:
        for row in csv.DictReader(f):
            if float(row["pred_db"]) > 55.0:
                dead.append(int(row["dir"]))
    # all pred>55 dirs (measured or not -- the COMBINATION was never run,
    # so combined cost is predicted either way; linearity is what's tested)
    check("prune-list", len(dead) == 27, f"{len(dead)} predicted-dead dirs")
    WdT = Wdf.T
    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)
    total = sum(s[i] * np.outer(MID @ U[:, i], Vt[i]) for i in dead)
    mse_pred = float((total ** 2).mean())
    d_pred = 10 * np.log10(1.0 / mse_pred) if mse_pred > 0 else float("inf")
    print(f"prune-predicted: removing {len(dead)} dirs costs {d_pred:.1f}dB "
          f"(statics only)")

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    pay0 = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}

    def run_down(W):
        pay = dict(pay0)
        pay["wdown"] = enc(W)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    base = run_down(WdT)
    Wp = WdT - sum(s[i] * np.outer(U[:, i], Vt[i]) for i in dead)
    got = run_down(Wp)
    mse = float(np.mean((got - base) ** 2))
    d = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    check("prune-confirm", abs(d - d_pred) <= 3.0,
          f"measured {d:.1f}dB vs predicted {d_pred:.1f}dB")
    # Q2 gain retune: doubling store 0's gain changes output by the SAME
    # magnitude as removing it (|+1x| == |-1x| of the component) -- predict
    # gain-x2 dB == ablation dB within 2dB, everything else fixed. One run.
    g2 = run_down(WdT + s[0] * np.outer(U[:, 0], Vt[0]))
    mse_g = float(np.mean((g2 - base) ** 2))
    d_g = 10 * np.log10(1.0 / mse_g) if mse_g > 0 else float("inf")
    ga = run_down(WdT - s[0] * np.outer(U[:, 0], Vt[0]))
    mse_a = float(np.mean((ga - base) ** 2))
    d_abl0 = 10 * np.log10(1.0 / mse_a) if mse_a > 0 else float("inf")
    check("retune-predict", abs(d_g - d_abl0) <= 2.0,
          f"gain-x2 {d_g:.1f}dB vs ablation {d_abl0:.1f}dB (same |delta|)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
