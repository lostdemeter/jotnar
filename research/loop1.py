"""Labeling loop, run one (one-shot, not a gate): (dir24, France).

Hypothesis (from MODEL_READ Q2: dir24 selective for France-pos, 28.2dB
spread): dir24 carries France-content. Match: fresh prompt with France at
position 5 (was 3) -- silence dir24, predict France-pos top-3 + <=45dB.
Verify by implant: write A*u24*wT (w = new MID[5]-aligned, functional aim)
-- predict France-pos leads field by >6dB (test_implant gap pattern with
label-derived aim). Bands stated here, before running.
"""
import os
import sys

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

PROMPT = "My friends from school visited France yesterday morning"
FRANCE_POS = 5
A_WRITE = 20.0


def main():
    from chain.qwen_mirror import load_layer0, build_H
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    g, embed, tok = load_layer0()
    H, ids, toks = build_H(g, embed, tok, PROMPT)
    print("tokens:", toks)
    assert "France" in toks[FRANCE_POS], toks

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    Wupf = g("model.layers.0.mlp.up_proj.weight")
    Wgf = g("model.layers.0.mlp.gate_proj.weight")
    Wdf = g("model.layers.0.mlp.down_proj.weight")
    ln2f = g("model.layers.0.post_attention_layernorm.weight")
    pay0 = {"H": enc(H), "wup": enc(Wupf.T), "wgate": enc(Wgf.T),
            "ln": enc(ln2f)}

    def run_down(WdT):
        pay = dict(pay0)
        pay["wdown"] = enc(WdT)
        feeds = ASM.run_text(text, REGISTRY, pay, sigs=SIGS, basedir=sdir)
        return dec(feeds["OUT"]), feeds

    base, feeds0 = run_down(Wdf.T)
    MID = dec(feeds0["MID"])
    WdT = Wdf.T
    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)

    def tokdb(got):
        d = ((np.ascontiguousarray(got, dtype=np.float64) - base) ** 2).mean(-1)
        return np.array([float("inf") if v == 0 else 10 * np.log10(1.0 / v)
                         for v in d])

    # MATCH: silence dir24
    d = tokdb(run_down(WdT - np.outer(U[:, 24] * s[24], Vt[24]))[0])
    rank = int((d < d[FRANCE_POS]).sum()) + 1
    print(f"MATCH silence-dir24: France-pos dB {d[FRANCE_POS]:.1f}, "
          f"rank {rank}/8, full {np.round(d, 1)}")
    print(f"MATCH verdict: {'CONFIRM' if d[FRANCE_POS] <= 45.0 and rank <= 3 else 'FALSIFY'}"
          f" (band: top-3 + <=45dB)")
    # VERIFY: write dir24 content at the new France position. Store view
    # (WdT = U S Vt: input keys U (4864, MID side), output values Vt (896)):
    # implant A*w*out24 with w = MID[5]-aligned key, out24 = Vt[24] value.
    # Token 5 projects maximally onto the key; what gets written IS dir24.
    w = MID[FRANCE_POS] / np.linalg.norm(MID[FRANCE_POS])
    out24 = Vt[24]
    got, _ = run_down(WdT + A_WRITE * np.outer(w, out24))
    dw = tokdb(got)
    others = [dw[i] for i in range(8) if i != FRANCE_POS]
    gap = float(np.mean(others) - dw[FRANCE_POS])
    print(f"VERIFY write-dir24: France-pos dB {dw[FRANCE_POS]:.1f}, "
          f"gap {gap:.1f}dB, full {np.round(dw, 1)}")
    print(f"VERIFY verdict: {'CONFIRM' if gap > 6.0 else 'FALSIFY'}"
          f" (band: gap > 6dB)")


if __name__ == "__main__":
    main()
