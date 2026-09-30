"""Substitution gates: identical columns swap at triple-direct seams.

Hypothesis under test (L3 architectures claim): two models whose structure
columns match at an op are mutually substitutable at that seam -- swap one
for the other and parity holds to the seam dividend (here: BIT-EXACT, since
integer ops are exact on every substrate).

Gate 1 (sig-swap): holo `sigmoid_trip` (numpy) vs RIFE `torch_emu.sigmoid_int`
(torch/CUDA) on shared integer vectors + a randomized triple sweep. Exact
(s,e,z) equality demanded -- same integers in, same integers out, across
repos AND frameworks. This is the first cross-model structure substitution
proven, not just used (bridge reuse was inherited; this is gated).

Gate 2 (chain-swap): the holo v4 chain with its sigmoid call swapped for
RIFE's (monkeypatched, no chain edit) must produce bit-identical Y-enh
triples on a real frame. Substitutability in situ, not just in isolation.

Gate 3 (edge-refusal, negative control): substitution must FAIL where columns
differ. holo conv pads replicate; phi-core phi_conv pads zero. On a
corner-impulse fixture the two MUST differ at the border and agree in the
interior -- characterizing the divergence contract precisely. A substitution
framework that can't say no is just aliasing.

Usage: python3 test_substitute.py (needs torch + the rife_reverse checkout
beside OpenCode root, CPU or CUDA -- integer ops are exact on both).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_RIFE = os.environ.get("RIFE_REVERSE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rife_reverse"))
sys.path.insert(0, _RIFE)

import phi_core.lattice as S
from chain import holo_phi as H

FAIL = []

RIFE = _RIFE


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def rife_sigmoid():
    import torch
    import torch_emu as R
    return torch, R


def main():
    try:
        torch, R = rife_sigmoid()
    except ImportError as e:
        print(f"SKIP (needs torch + rife_reverse checkout: {e})")
        sys.exit(0)
    dev = R.DEV
    print(f"rife backend: {dev} (integer ops exact on both)")

    # ---- Gate 1: shared vectors + randomized sweep ----
    vecs = [(-1, 35955, 0), (-1, 33505, 0), (-1, 32031, 0), (1, 0, 0),
            (1, 32031, 0), (1, 33505, 0), (1, 35955, 0)]
    all_ok = True
    for s, e, z in vecs:
        hn = H.sigmoid_trip((np.full(1, s, np.int8), np.full(1, e, np.int32),
                             np.full(1, z, np.uint8)))
        rt = (torch.full((1,), s, dtype=torch.int8, device=dev),
              torch.full((1,), e, dtype=torch.int32, device=dev),
              torch.full((1,), z, dtype=torch.uint8, device=dev))
        ro = R.sigmoid_int(rt)
        r = (ro[0].to("cpu").numpy(), ro[1].to("cpu").numpy().astype(np.int32),
             ro[2].to("cpu").numpy())
        if not (int(hn[0][0]) == int(r[0][0]) and int(hn[1][0]) == int(r[1][0])
                and int(hn[2][0]) == int(r[2][0])):
            all_ok = False
            print(f"    mismatch at {(s, e, z)}: holo={(int(hn[0][0]), int(hn[1][0]), int(hn[2][0]))} "
                  f"rife={(int(r[0][0]), int(r[1][0]), int(r[2][0]))}")
    check("sig-swap-vectors", all_ok, f"{len(vecs)} shared vectors, exact")

    rng = np.random.default_rng(41)
    n = 2000
    rs = rng.choice([-1, 1], n).astype(np.int8)
    re_ = rng.integers(0, 65536, n).astype(np.int32)
    rz = (rng.random(n) < 0.05).astype(np.uint8)
    # edge exponents forced in (rails + zero + BIAS region)
    re_[:6] = [0, 1, 32768, 65534, 65535, 100]
    hn = H.sigmoid_trip((rs, re_, rz))
    rt = (torch.from_numpy(rs).to(torch.int8).to(dev),
          torch.from_numpy(re_).to(torch.int32).to(dev),
          torch.from_numpy(rz).to(torch.uint8).to(dev))
    ro = R.sigmoid_int(rt)
    r = (ro[0].to("cpu").numpy(), ro[1].to("cpu").numpy(), ro[2].to("cpu").numpy())
    exact = bool((hn[0] == r[0].astype(np.int8)).all()
                 and (hn[1] == r[1].astype(np.int32)).all()
                 and (hn[2] == r[2].astype(np.uint8)).all())
    if not exact:
        bad = int(((hn[0] != r[0].astype(np.int8)) | (hn[1] != r[1].astype(np.int32))
                   | (hn[2] != r[2].astype(np.uint8))).sum())
        print(f"    {bad}/{n} triples diverge")
    check("sig-swap-sweep", exact, f"{n} random triples, exact (s,e,z)")

    # ---- Gate 2: pipeline-level swap, bit-identical Y-enh triples ----
    from PIL import Image
    _cand_ext = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rife_reverse", "samples", "f_012.png")
    cand = _cand_ext if os.path.isfile(_cand_ext) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "input_example.png")
    rgb01 = np.asarray(Image.open(cand).convert("RGB"), dtype=np.float64) / 255.0
    rgb_lin = np.power(rgb01, 2.2).astype(np.float32)
    y = (0.2126 * rgb_lin[:, :, 0] + 0.7152 * rgb_lin[:, :, 1]
         + 0.0722 * rgb_lin[:, :, 2]).astype(np.float64)

    ref, _ = H.enhance_luminance_int(y, beta=0.5, blur="splat_soft", ctrl="soft")

    orig = H.sigmoid_trip

    calls = {"n": 0}

    def rife_backed(t):
        calls["n"] += 1
        tt = (torch.from_numpy(np.ascontiguousarray(t[0])).to(torch.int8).to(dev),
              torch.from_numpy(np.ascontiguousarray(t[1])).to(torch.int32).to(dev),
              torch.from_numpy(np.ascontiguousarray(t[2])).to(torch.uint8).to(dev))
        o = R.sigmoid_int(tt)
        return (o[0].to("cpu").numpy().astype(np.int8),
                o[1].to("cpu").numpy().astype(np.int32),
                o[2].to("cpu").numpy().astype(np.uint8))

    H.sigmoid_trip = rife_backed
    try:
        got, _ = H.enhance_luminance_int(y, beta=0.5, blur="splat_soft", ctrl="soft")
    finally:
        H.sigmoid_trip = orig
    # tripwire: the swap must have EXECUTED (a gate that passes without
    # exercising the seam is green wallpaper -- doctrine).
    check("chain-swap-live", calls["n"] > 0,
          f"rife sigmoid invoked {calls['n']}x in-chain")
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    # NOTE: control.py does `from chain.holo_phi import sigmoid_trip` at call
    # time (inside beta_field_soft), so rebinding H.sigmoid_trip IS effective
    # (verified by the assertion below tripping if the swap didn't take).
    check("chain-swap", same, "swapped chain bit-identical (Y-enh triples)")

    # ---- Gate 3: negative control -- replicate vs zero pad ----
    m_acc, m_cov = H._load_scales()
    img = np.full((16, 16), 0.3)
    img[0, 0] = 1.0  # corner impulse: maximal border exposure
    it = S.encode(img)
    k = H.gaussian_kernel(radius=2, sigma=1.0)
    rep = H.conv_trip(it, k, m_acc, m_out=m_cov)

    def conv_zero_pad(img_trip, kernel, m_acc, m_out):
        r = kernel.shape[0] // 2
        Hh, Ww = img_trip[0].shape

        def pad(x, dt):
            return np.pad(x, ((r, r), (r, r)), mode="constant",
                          constant_values=dt)
        # zero triple: s=0,e=0,z=0 would poison tmul signs; instead pad the
        # DECODED float with 0 and re-encode (zero-pad semantics), keeping
        # everything else (tmul/to_fixed/accumulate) identical.
        dec = S.decode(img_trip[0], img_trip[1]) * (1 - img_trip[2].astype(np.float64))
        zp = np.pad(dec, ((r, r), (r, r)), mode="constant")
        acc = None
        ks, ke, kz = H.kernel_triples(kernel)
        for dy in range(kernel.shape[0]):
            for dx in range(kernel.shape[1]):
                tap = S.encode(zp[dy:dy + Hh, dx:dx + Ww])
                prod = H.tmul(tap, (np.full((Hh, Ww), ks[dy, dx], np.int8),
                                    np.full((Hh, Ww), ke[dy, dx], np.int32),
                                    np.full((Hh, Ww), kz[dy, dx], np.uint8)))
                q = S.to_fixed(prod[0], prod[1], prod[2], m_acc)
                acc = q if acc is None else acc + q
        return S.from_fixed(H.rescale_(acc, m_acc, m_out), m_out)

    zpad = conv_zero_pad(it, k, m_acc, m_cov)
    q_rep = S.to_fixed(rep[0], rep[1], rep[2], m_cov)
    q_zpad = S.to_fixed(zpad[0], zpad[1], zpad[2], m_cov)
    diff = q_rep != q_zpad
    border = np.zeros((16, 16), bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    check("edge-refuse-border", bool(diff[border].any()),
          "replicate vs zero MUST differ at the border")
    check("edge-refuse-interior", bool((~diff[~border]).all()),
          "and agree in the interior (only the edge rule differs)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
