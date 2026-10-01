"""Storebank on real weights (v1.3 native): banks as listing data.

mlp_qwen0_storebank.asm replaces MATMUL(MID, wdown) with
storebank_apply(MID, Ub, Vb) (Ub gains-folded, built host-side via
chain/engram.py bank()). Gates: full-bank parity vs MATMUL form
(predict >=50dB: toy hit 96, coarser quanta cost decibels); pruned bank
(drop the 27 predicted-dead) lands inside float-linearity prediction
+/-3dB (assembler-side edit with numbers -- test_prune precedent, now in
bank form). Proves the same values flow through store-addressed data.
SKIPs without the local HF cache.
Usage: python3 test_storebank.py (~4 listing runs)
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []
PROMPT = "The capital of France is Paris, and the capital of Germany is"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    from chain.qwen_mirror import load_layer0, build_H, mlp_forward
    from chain import engram as EN
    if not os.path.isfile(os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub",
            "models--Qwen--Qwen2-0.5B", "snapshots",
            "91d2aff3f957f99e4c74c962f2f408dcc88a18d8",
            "model.safetensors")):
        print("SKIP (needs Qwen2-0.5B in local HF cache)")
        sys.exit(0)
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    g, embed, tok = load_layer0()
    H, _, _ = build_H(g, embed, tok, PROMPT)
    _, MID, Wupf, Wgf, Wdf, ln2f = mlp_forward(H, g)

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(root, "programs")
    ref_text = open(os.path.join(sdir, "mlp_qwen0.asm")).read()
    sb_text = open(os.path.join(sdir, "mlp_qwen0_storebank.asm")).read()
    pay0 = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}
    base = dec(ASM.run_text(ref_text, REGISTRY, dict(
        pay0, wdown=enc(Wdf.T)), sigs=SIGS, basedir=sdir)["OUT"])

    EN.freeze(Wdf.T, "qwen0_down")
    Ub_full, Vb_full = EN.bank("qwen0_down")
    pay = dict(pay0)
    pay["Ub"], pay["Vb"] = enc(Ub_full), enc(Vb_full)
    got = dec(ASM.run_text(sb_text, REGISTRY, pay, sigs=SIGS,
                           basedir=sdir)["OUT"])
    mse = float(np.mean((got - base) ** 2))
    d = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    check("storebank-real-parity", d >= 50.0,
          f"{d:.1f}dB bank-vs-matmul on real weights")
    dead = []
    with open(os.path.join(root, "docs", "readout_down_896.csv")) as f:
        for row in csv.DictReader(f):
            if float(row["pred_db"]) > 55.0:
                dead.append(int(row["dir"]))
    keep = [i for i in range(896) if i not in set(dead)]
    pay2 = dict(pay0)
    pay2["Ub"], pay2["Vb"] = enc(Ub_full[:, keep]), enc(Vb_full[keep])
    got_p = dec(ASM.run_text(sb_text, REGISTRY, pay2, sigs=SIGS,
                             basedir=sdir)["OUT"])
    mse_p = float(np.mean((got_p - base) ** 2))
    d_p = 10 * np.log10(1.0 / mse_p) if mse_p > 0 else float("inf")
    # linearity prediction from statics (same law as test_prune Q1)
    U, s, Vt, _ = EN.load("qwen0_down")
    total = sum(s[i] * np.outer(MID @ U[:, i], Vt[i]) for i in dead)
    mse_pr = float((total ** 2).mean())
    d_pr = 10 * np.log10(1.0 / mse_pr) if mse_pr > 0 else float("inf")
    check("storebank-prune-confirm", abs(d_p - d_pr) <= 3.0,
          f"bank-form pruned {d_p:.1f}dB vs predicted {d_pr:.1f}dB")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
