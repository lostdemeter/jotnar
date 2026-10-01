"""CREATE demo (CRUD research Q3): label-to-store on a fresh prompt.

Store built from the verified (dir24, France) label WITHOUT touching the
fresh prompt: key = mean France-MID rows from two OLD prompts (geo-pos3,
loop1-pos5), value = Vt[24], gain = s24. Added to FULL weights, tested on
a THIRD prompt (France at pos2, never used for key or label). Predict
(loop bands): France-pos top-3 + write-lead gap > 6dB. If the key carries
France-ness, France-pos2 responds most -- creation from knowledge, not fit.
SKIPs without the local HF cache.
Usage: python3 test_create.py (4 listing runs)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []
P_GEO = "The capital of France is Paris, and the capital of Germany is"
P_L1 = "My friends from school visited France yesterday morning"
P_FRESH = "Germany borders France to the southwest near Switzerland today"
FRANCE_FRESH = 2


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    from chain.qwen_mirror import load_layer0, build_H, mlp_forward
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

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    # key from OLD prompts only (fresh prompt never touches the key)
    keys = []
    for pr, pos in ((P_GEO, 3), (P_L1, 5)):
        Hx, _, _ = build_H(g, embed, tok, pr)
        _, MIDx, _, _, _, _ = mlp_forward(Hx, g)
        keys.append(MIDx[pos])
    key = sum(keys) / len(keys)
    key /= np.linalg.norm(key)
    Hf, _, toks = build_H(g, embed, tok, P_FRESH)
    assert "France" in toks[FRANCE_FRESH], toks
    print("fresh tokens:", toks)
    _, _, Wupf, Wgf, Wdf, ln2f = mlp_forward(Hf, g)
    # value/gain from the verified label (dir24 of down_proj)
    WdT = Wdf.T
    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)
    pay0 = {"H": enc(Hf), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}

    def run_down(W):
        pay = dict(pay0)
        pay["wdown"] = enc(W)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    base = run_down(WdT)
    created = WdT + s[24] * np.outer(key, Vt[24])
    got = run_down(created)
    d = ((got - base) ** 2).mean(-1)
    tokdb = np.array([float("inf") if v == 0 else 10 * np.log10(1.0 / v)
                      for v in d])
    rank = int((tokdb < tokdb[FRANCE_FRESH]).sum()) + 1
    others = [tokdb[i] for i in range(8) if i != FRANCE_FRESH]
    gap = float(np.mean(others) - tokdb[FRANCE_FRESH])
    print(f"create: France-pos dB {tokdb[FRANCE_FRESH]:.1f} rank {rank}/8 "
          f"gap {gap:.1f}dB; full {np.round(tokdb, 1)}")
    check("create-match", rank <= 3 and tokdb[FRANCE_FRESH] <= 45.0,
          f"France-pos rank {rank}/8 at {tokdb[FRANCE_FRESH]:.1f}dB")
    check("create-gap", gap > 6.0,
          f"created store leads field by {gap:.1f}dB")
    # Q4 shelf reuse (null prediction): the SAME write on pruned vs full
    # weights has the SAME effect (linearity -- MID identical, added term
    # identical in float; lattice quanta differ only). Shelves are a
    # CAPACITY story (free space + headroom), not interference: freeing
    # does not change what a write does. Bar >=40dB on the effect vectors.
    import csv as _csv
    dead = []
    with open(os.path.join(root, "docs", "readout_down_896.csv")) as f:
        for row in _csv.DictReader(f):
            if float(row["pred_db"]) > 55.0:
                dead.append(int(row["dir"]))
    UU, ss, VVt = np.linalg.svd(WdT, full_matrices=False)
    Wp = WdT - sum(ss[i] * np.outer(UU[:, i], VVt[i]) for i in dead)
    base_p = run_down(Wp)
    newstore = s[24] * np.outer(key, Vt[24])
    eff_full = run_down(WdT + newstore) - base
    eff_pruned = run_down(Wp + newstore) - base_p
    dd = eff_full - eff_pruned
    mse_dd = float((dd ** 2).mean())
    d_dd = 10 * np.log10(1.0 / mse_dd) if mse_dd > 0 else float("inf")
    check("shelf-null", d_dd >= 40.0,
          f"write effect pruned-vs-full identical at {d_dd:.1f}dB")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
