"""Cross-layer check (CRUD research Q5): do the laws hold past layer 0?

Layer-1 input = layer-0 output (post-MLP residual from mlp_forward);
layer-1 MLP weights + ln through the SAME listing (mlp_qwen0.asm is
generic over H/weights). Spot-check, not full sweep: spectrum decaying,
giant ablation strong, 8 sampled dirs spread + tracking. If the structure
is the architecture's (not layer-0's accident), the pattern reproduces.
SKIPs without the local HF cache.
Usage: python3 tests/test_layer1.py (~12 listing runs)
"""
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
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    g, embed, tok = load_layer0()
    H0, _, _ = build_H(g, embed, tok, PROMPT)
    REF0, _, _, _, _, _ = mlp_forward(H0, g)
    H1 = REF0  # layer-1 block input = layer-0 output
    import torch
    dt = torch.float64
    ln1 = torch.tensor(g("model.layers.1.post_attention_layernorm.weight"),
                       dtype=dt)
    Wup = g("model.layers.1.mlp.up_proj.weight")
    Wg = g("model.layers.1.mlp.gate_proj.weight")
    Wd = g("model.layers.1.mlp.down_proj.weight")

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    text = open(os.path.join(root, "programs", "mlp_qwen0.asm")).read()
    sdir = os.path.join(root, "programs")
    pay0 = {"H": enc(H1), "wup": enc(Wup.T), "wgate": enc(Wg.T),
            "ln": enc(ln1.numpy())}

    def run_down(W):
        pay = dict(pay0)
        pay["wdown"] = enc(W)
        return dec(ASM.run_text(text, REGISTRY, pay, sigs=SIGS,
                                basedir=sdir)["OUT"])

    WdT = Wd.T
    base = run_down(WdT)

    def psnr(a):
        mse = float(np.mean((a - base) ** 2))
        return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

    U, s, Vt = np.linalg.svd(WdT, full_matrices=False)
    check("layer1-spectrum", s[0] / s[-1] > 5.0,
          f"decaying {s[0]:.2f}..{s[-1]:.2f} (ratio {s[0]/s[-1]:.0f}x)")
    d0 = psnr(run_down(WdT - np.outer(U[:, 0] * s[0], Vt[0])))
    check("layer1-giant", d0 < 40.0,
          f"top-dir ablation moves output at {d0:.1f}dB")
    idx = [8, 100, 200, 328, 500, 700, 800, 888]
    dd = []
    for i in idx:
        dd.append(psnr(run_down(WdT - np.outer(U[:, i] * s[i], Vt[i]))))
    dd = np.array(dd)
    sv = s[idx]
    spread = float(dd.max() - dd.min())
    corr = float(np.corrcoef(dd, sv)[0, 1])
    check("layer1-spread", spread > 10.0,
          f"{spread:.1f}dB over 8 sampled dirs")
    check("layer1-tracks", corr < -0.5,
          f"corr(dir-dB, sval)={corr:.2f} (law reproduces on layer 1)")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
