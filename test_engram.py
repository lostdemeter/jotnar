"""Native ENGRAM storage gates (v1.3): freeze/load/recompose + stability.

Freeze Qwen down_proj to stores/, reload, recompose within 1e-9
(maxabsdiff), and prove store IO adds nothing (listing with loaded stores
bit-exact vs direct weights). Regeneration determinism: freeze twice, same
bytes (shelves the gitignore decision -- blobs reproducible, not precious).
SKIPs without the local HF cache.
Usage: python3 test_engram.py (~3 listing runs + 2 SVDs)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAIL = []


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
    import hashlib
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain import engram as EN
    from chain.asm_ops import REGISTRY, SIGS

    g, embed, tok = load_layer0()
    H, _, _ = build_H(g, embed, tok,
                      "The capital of France is Paris, and the capital of Germany is")
    _, _, Wupf, Wgf, Wdf, ln2f = mlp_forward(H, g)
    WdT = Wdf.T
    p1 = EN.freeze(WdT, "_test_roundtrip")
    U, s, Vt, meta = EN.load("_test_roundtrip")
    err = float(np.abs(EN.recompose(U, s, Vt) - WdT).max())
    check("engram-roundtrip", err < 1e-9,
          f"maxabsdiff {err:.1e} (float64 SVD roundtrip)")
    h1 = hashlib.md5(open(p1, "rb").read()).hexdigest()
    EN.freeze(WdT, "_test_roundtrip")
    h2 = hashlib.md5(open(p1, "rb").read()).hexdigest()
    check("engram-deterministic", h1 == h2,
          "freeze twice, identical bytes (regenerable, not precious)")
    for stale in (p1, os.path.join(EN.STORE_DIR, "_test_roundtrip.json")):
        os.remove(stale)

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    pay0 = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}

    def run_down(W):
        pay = dict(pay0)
        pay["wdown"] = enc(W)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    EN.freeze(WdT, "qwen0_down")
    Ul, sl, Vtl, _ = EN.load("qwen0_down")
    a = run_down(WdT)
    b = run_down(EN.recompose(Ul, sl, Vtl))
    check("engram-stable", bool(((a - b) == 0).all()),
          "loaded stores bit-exact vs direct weights (IO adds nothing)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
