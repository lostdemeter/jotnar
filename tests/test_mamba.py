"""Mamba trajectory study (v1.6 anatomy-mamba): real selective scan in-assembly.

Qwen-style attention boundary left behind: here the recurrence ITSELF runs
in the listing (SCAN threaded via repeat(), STATE-carried h) on REAL
mamba-130m weights + embeddings. Boundary floats (stated): conv1d,
softplus-dt, abar-exp (no EXP/SOFTPLUS mnemonics exist -- same boundary
doctrine as attention scores); everything else (tmul-built Bx? no --
Bx also boundary this round: dt*B*x needs no LUT but the study pins the
scan core first, composition second) is triples. Hmm -- see body.
Gates: trajectory parity vs torch float64 (bar 40dB), ranges reported
(abar/h/Bx magnitudes + chosen m), repeat-threading determinism.
SKIPs without the local HF cache.
Usage: python3 tests/test_mamba.py
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
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import glob as _g
    _snaps = _g.glob(os.path.expanduser("~/.cache/huggingface/hub/models--state-spaces--"
                                        "mamba-130m-hf/snapshots/*/model.safetensors"))
    if not _snaps:
        print("SKIP (needs mamba-130m-hf in local HF cache)")
        sys.exit(0)
    snap = os.path.dirname(_snaps[0])
    os.environ["HF_HUB_OFFLINE"] = "1"
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS

    sd = load_file(os.path.join(snap, "model.safetensors"), device="cpu")
    g = lambda n: sd[n].float().double().numpy()
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"][0][:8].numpy()
    E = sd["backbone.embeddings.weight"][torch.tensor(ids)].float().double().numpy()
    dt = torch.float64
    xt = torch.tensor(E, dtype=dt)
    W_in = torch.tensor(g("backbone.layers.0.mixer.in_proj.weight"), dtype=dt)
    Wx = torch.tensor(g("backbone.layers.0.mixer.x_proj.weight"), dtype=dt)
    Wdt = torch.tensor(g("backbone.layers.0.mixer.dt_proj.weight"), dtype=dt)
    bdt = torch.tensor(g("backbone.layers.0.mixer.dt_proj.bias"), dtype=dt)
    A = -torch.exp(torch.tensor(g("backbone.layers.0.mixer.A_log"), dtype=dt))
    Dv = torch.tensor(g("backbone.layers.0.mixer.D"), dtype=dt)
    Wc = torch.tensor(g("backbone.layers.0.mixer.conv1d.weight"), dtype=dt)
    bc = torch.tensor(g("backbone.layers.0.mixer.conv1d.bias"), dtype=dt)
    # in_proj: 768 -> 3072 = [x(1536), z(1536)]; conv1d causal on x branch
    proj = xt @ W_in.T
    xx, zz = proj[:, :1536], proj[:, 1536:]
    Sq = xx.shape[0]
    xc = torch.nn.functional.conv1d(
        torch.nn.functional.pad(xx.T.unsqueeze(0), (3, 0)),
        Wc, bc, groups=1536).squeeze(0).T
    # x_proj: 1536 -> [dt(48), B(16), C(16)]; dt per dim via dt_proj+bias
    q = xc @ Wx.T
    dtb, Bb, Cb = q[:, :48], q[:, 48:64], q[:, 64:80]
    dts = torch.nn.functional.softplus(dt_b := (dtb @ Wdt.T + bdt))
    abar = torch.exp(dts[:, :, None] * A[None, :, :])  # (S,D,N)
    Bx = dts[:, :, None] * Bb[:, None, :] * xc[:, :, None]
    # torch trajectory
    h = torch.zeros(1536, 16, dtype=dt)
    traj = []
    for t in range(Sq):
        h = abar[t] * h + Bx[t]
        traj.append(h.clone())
    traj = torch.stack(traj).numpy()  # (S,D,N)
    print(f"abar range [{abar.min():.4f},{abar.max():.4f}] "
          f"h max {np.abs(traj).max():.4f} Bx max {np.abs(Bx.numpy()).max():.4f}")

    def enc(a):
        return S.encode(np.ascontiguousarray(a, dtype=np.float64))

    def dec(t):
        return S.decode(np.ascontiguousarray(t[0]),
                        np.ascontiguousarray(t[1])) * (
                            1 - np.ascontiguousarray(t[2]).astype(float))

    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    text = open(os.path.join(root, "programs", "scan_mamba.asm")).read()
    sdir = os.path.join(root, "programs")
    abl = [enc(abar[t].numpy()) for t in range(Sq)]
    bxl = [enc(Bx[t].numpy()) for t in range(Sq)]
    seed = (np.zeros((1536, 16), np.int8), np.zeros((1536, 16), np.int32),
            np.ones((1536, 16), np.uint8))
    hists, _, _ = ASM.repeat(text, REGISTRY,
                             {"h": seed, "ab": abl, "bx": bxl},
                             Sq, sigs=SIGS)
    got = np.stack([dec(h["h"]) for h in hists])
    mse = float(np.mean((got - traj) ** 2))
    d = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
    check("mamba-trajectory", d >= 40.0,
          f"{d:.1f}dB scan trajectory vs torch (peak=1.0)")
    # determinism: rerun identical
    hists2, _, _ = ASM.repeat(text, REGISTRY,
                              {"h": seed, "ab": abl, "bx": bxl},
                              Sq, sigs=SIGS)
    same = all(bool((hists[i]["h"][k] == hists2[i]["h"][k]).all())
               for i in range(Sq) for k in (0, 1, 2))
    check("mamba-deterministic", same, "repeat identical bit-exact")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
