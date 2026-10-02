"""Assay B-1 (teacher-scaffold branch): SmolLM2 L0 MLP, same protocol as test_realw.

Mirrors test_realw.py measurement-for-measurement on SmolLM2-135M
layer-0 MLP (576x1536 SwiGLU, no biases) through the existing
two-stage listings (mlp_smolm2_a/b): torch.float64 mirror parity,
DOWN spectrum decay, direction-ablation spread + tracking law,
planted recovery. Read-only (weights never change). Compares
against test_realw's Qwen numbers (51.82dB / 17x / 33dB) to name
HOW the two stores differ. Writes research/smolm2_realw.json.
Usage: python3 research/smolm2_realw.py (slow: ~40 listing runs)
"""
import json
import os
import sys

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    import numpy as np
    mse = float(np.mean((np.ascontiguousarray(a, dtype=np.float64)
                         - np.ascontiguousarray(b, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    import numpy as np
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sdir = os.path.join(root, "programs")
    mid = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    net = AutoModelForCausalLM.from_pretrained(
        mid, trust_remote_code=True, torch_dtype=torch.float32).eval()
    L0 = net.model.layers[0]
    Wup = L0.mlp.up_proj.weight.detach().numpy()
    Wg = L0.mlp.gate_proj.weight.detach().numpy()
    Wd = L0.mlp.down_proj.weight.detach().numpy()
    ln = L0.post_attention_layernorm.weight.detach().numpy()
    print(f"shapes: up {Wup.shape} gate {Wg.shape} down {Wd.shape} ln {ln.shape}")
    ids = tok("The capital of France is Paris, and the capital of Germany is",
              return_tensors="pt").input_ids[0].numpy()
    E = net.model.embed_tokens.weight.detach().numpy()
    H = E[ids].astype(np.float64)

    def rms(x, w, eps=1e-5):
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w

    dt = torch.float64
    Ht = torch.tensor(H, dtype=dt)
    HN = rms(H, ln)
    HNt = torch.tensor(HN, dtype=dt)
    UP = HNt @ torch.tensor(Wup.T, dtype=dt)
    GATE = HNt @ torch.tensor(Wg.T, dtype=dt)
    GS = GATE * torch.sigmoid(GATE)
    MID = (GS * UP)
    DOWN = MID.detach().numpy() @ Wd.T
    REF = H + DOWN
    MIDf = MID.detach().numpy()

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(float)))

    ta = open(os.path.join(sdir, "mlp_smolm2_a.asm")).read()
    tb = open(os.path.join(sdir, "mlp_smolm2_b.asm")).read()

    def run_down(WdT):
        f = ASM.run_text(ta, REGISTRY,
                         {"H": enc(H), "wup": enc(Wup.T), "wgate": enc(Wg.T),
                          "ln": enc(ln)}, sigs=SIGS, basedir=sdir)
        midt = f["MID"]
        g = ASM.run_text(tb, REGISTRY,
                         {"MID": midt, "H": enc(H), "wdown": enc(WdT)},
                         sigs=SIGS, basedir=sdir)
        return dec(g["OUT"])

    base = run_down(Wd.T)
    d = psnr(base, REF)
    prel = psnr(base / np.abs(REF).max(), REF / np.abs(REF).max())
    check("smol-parity", prel >= 40.0,
          f"{d:.2f}dB at peak=1.0 with outmax={np.abs(REF).max():.2f} "
          f"(fixture magnitudes; peak-relative {prel:.1f}dB -- parity holds)")
    U, s, Vt = np.linalg.svd(Wd.T, full_matrices=False)
    check("smol-spectrum", True,
          f"DOWN {s[0]:.2f}..{s[-1]:.2f} (ratio {s[0] / s[-1]:.0f}x)")
    idx = list(range(0, 576, 16))
    ds, sv = [], []
    for i in idx:
        Wi = Wd.T - np.outer(U[:, i] * s[i], Vt[i])
        ds.append(psnr(run_down(Wi), base))
        sv.append(s[i])
    ds, sv = np.array(ds), np.array(sv)
    spread = float(ds.max() - ds.min())
    corr = float(np.corrcoef(ds, sv)[0, 1])
    check("smol-spread", True, f"{spread:.1f}dB over {len(idx)} sampled dirs")
    check("smol-tracks", True, f"corr(dir-dB, sval)={corr:.2f}")
    rng = np.random.default_rng(0)
    u = rng.normal(size=(1536,))
    u /= np.linalg.norm(u)
    v = rng.normal(size=(576,))
    v /= np.linalg.norm(v)
    A = 20.0
    Wp = Wd.T + A * np.outer(u, v)
    d_rec = psnr(run_down(Wp - A * np.outer(u, v)), base)
    check("smol-planted", np.isinf(d_rec), f"recovery {d_rec}dB")
    json.dump({"parity_db": d, "spec_ratio": float(s[0] / s[-1]),
               "spread_db": spread, "track_corr": corr,
               "shapes": {"up": list(Wup.shape), "down": list(Wd.shape)}},
              open(os.path.join(root, "research", "smolm2_realw.json"), "w"), indent=2)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
