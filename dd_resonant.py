"""Resonant-phase indexing over engrams (one-shot): does phase preserve neighborhoods?

Rule (content-derived, deterministic): phase_i = (Z1*||u_i|| + Z2*(u_i.r))
mod 2pi, with Z1,Z2 the first Riemann zeros and r a fixed universal vector
(seeded once, shared across models -- the coordination-free constant).
Test on Qwen L0 down_proj keys (U columns, 896 dirs): top-5 nearest-phase
neighbors vs top-5 cosine neighbors overlap. Chance ~= 0.03 dirs; bar:
mean overlap >= 2/5 (structure preserved) vs < 1 (phases scramble it).
Zero runs, statics only. Toward exact cross-model lookup over discovered
content (the resonant bridge proposal).
"""
import numpy as np

Z1, Z2 = 14.134725141734693, 21.022039638771555


def main():
    import torch
    from safetensors.torch import load_file
    import os
    D = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--"
                           "Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8")
    sd = load_file(D + "/model.safetensors", device="cpu")
    W = sd["model.layers.0.mlp.down_proj.weight"].float().double().numpy().T
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    rng = np.random.default_rng(7)  # the universal constant vector
    r = rng.normal(size=U.shape[0])
    r /= np.linalg.norm(r)
    nU = U / (np.linalg.norm(U, axis=0, keepdims=True) + 1e-30)
    ph = (Z1 * np.linalg.norm(U, axis=0) + Z2 * (nU.T @ r)) % (2 * np.pi)
    # cosine neighborhoods (ground truth structure)
    Cn = nU.T @ nU
    np.fill_diagonal(Cn, -2)
    cos_top = np.argsort(-Cn, axis=1)[:, :5]
    # phase neighborhoods (circular distance)
    Dp = np.abs(ph[:, None] - ph[None, :])
    Dp = np.minimum(Dp, 2 * np.pi - Dp)
    np.fill_diagonal(Dp, 1e9)
    ph_top = np.argsort(Dp, axis=1)[:, :5]
    ov = np.array([len(set(cos_top[i]) & set(ph_top[i])) for i in range(896)])
    print(f"mean top-5 overlap: {ov.mean():.2f}/5 (chance ~0.03; bar >= 2)")
    print(f"frac with overlap>=2: {(ov >= 2).mean():.2f}")
    print("RESONANT-INDEX:", "PRESERVES neighborhoods"
          if ov.mean() >= 2.0 else "SCRAMBLES neighborhoods")


if __name__ == "__main__":
    main()
