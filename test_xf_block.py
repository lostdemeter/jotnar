"""Whole-block float parity (v1.0 Gate 1): xf_block.asm vs torch.

The listing programs/xf_block.asm (Qwen-style encoder block: RMSNorm, QKV,
RoPE, causal-free attention, out-proj, residual, SwiGLU MLP) runs through the
integer lattice and must hold >=40dB vs an INDEPENDENT torch.float64 reference
(no phi-core calls on the reference side -- same formulas, separate code).

Basis (declared): units are activation values, peak=1.0, PSNR =
10*log10(1/mse), bar 40dB (same bar as every other parity gate in the repo).

Operating envelope (stated, not hidden): the frozen holo scales
(chain/M.json: m_acc covers products ~=0.16, m_cov covers values ~=1.0) bound
the regime. The barred fixture (S=8, D=16, Dff=32, magnitudes *=0.10) keeps
attention scores <=1.0 abs (the softmax T-transformation contract: to_fixed
saturates above 1.0 at BIAS, so full-range attention needs the T-transform
path -- backlog) and matmul products inside m_acc coverage. Scores-max is
gated to prove the barred row is actually in-contract (not vacuous).

Out-of-contract row (scale 0.30, scores ~3.5) is MEASURED, not barred: it
documents where the current scales stop covering (softmax saturation +
matmul coverage), with the mechanism stated. Recalibrated scales for
real-model magnitudes (activations ~1-5, head dims 64-128, scores needing
1/sqrt(d) scaling the listing does not yet have) are backlog -- Gate 1
closes on the in-contract proof, the envelope is the contract.

Usage: python3 test_xf_block.py (needs torch).
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS

BAR_DB = 40.0
FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def dec(t):
    return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
            * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))


def enc(a):
    return S.encode(np.ascontiguousarray(a, dtype=np.float64))


def fixture(Sq=8, D=16, Dff=32, scale=0.10, seed=0, mlp_scale=None):
    rng = np.random.default_rng(seed)
    x_f = (rng.random((Sq, D)) - 0.5) * 2 * scale
    pos = np.arange(Sq, dtype=np.int64)

    def wf(shape, s=None):
        s = scale if s is None else s
        return (rng.random(shape) - 0.5) * 2 * s
    wq_f, wk_f, wv_f, wo_f = wf((D, D)), wf((D, D)), wf((D, D)), wf((D, D))
    ms = scale if mlp_scale is None else mlp_scale
    wup_f, wgate_f = wf((D, Dff), ms), wf((D, Dff), ms)
    wdown_f = wf((Dff, D), ms)
    rms1_f = np.ones(D) * 0.9 + (rng.random(D) - 0.5) * 0.1
    rms2_f = np.ones(D) * 0.9 + (rng.random(D) - 0.5) * 0.1
    weights = (wq_f, wk_f, wv_f, wo_f, wup_f, wgate_f, wdown_f)
    return x_f, pos, weights, rms1_f, rms2_f


def torch_block(x_f, pos, weights, rms1_f, rms2_f):
    """Independent reference: pure torch.float64, no phi-core anywhere."""
    import torch
    dt = torch.float64
    wq_f, wk_f, wv_f, wo_f, wup_f, wgate_f, wdown_f = weights
    xt = torch.tensor(x_f, dtype=dt)
    wqt = torch.tensor(wq_f, dtype=dt)
    wkt = torch.tensor(wk_f, dtype=dt)
    wvt = torch.tensor(wv_f, dtype=dt)
    wot = torch.tensor(wo_f, dtype=dt)
    wupt = torch.tensor(wup_f, dtype=dt)
    wgatet = torch.tensor(wgate_f, dtype=dt)
    wdownt = torch.tensor(wdown_f, dtype=dt)
    r1 = torch.tensor(rms1_f, dtype=dt)
    r2 = torch.tensor(rms2_f, dtype=dt)
    eps = 4514 / 2 ** 36  # rmsnorm eps_c in 2^-36 units (matches rmsnorm_int)

    def rmsnorm(x, w):
        return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * w

    def rope(x, p, base=10000.0):
        d = x.shape[-1]
        i = torch.arange(d // 2, dtype=dt)
        theta = base ** (-2 * i / d)
        ang = p.unsqueeze(1).to(dt) * theta.unsqueeze(0)
        c, s = torch.cos(ang), torch.sin(ang)
        y = torch.empty_like(x)
        y[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
        y[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
        return y

    post = torch.tensor(np.ascontiguousarray(pos), dtype=torch.int64)
    xn = rmsnorm(xt, r1)
    Q, K, V = xn @ wqt, xn @ wkt, xn @ wvt
    QR, KR = rope(Q, post), rope(K, post)
    SC = QR @ KR.T
    P = torch.softmax(SC, dim=-1)
    CTX = P @ V
    O = CTX @ wot
    H = xt + O
    HN = rmsnorm(H, r2)
    UP, GATE = HN @ wupt, HN @ wgatet
    GS = GATE * torch.sigmoid(GATE)
    DOWN = (GS * UP) @ wdownt
    OUT = H + DOWN
    return OUT.detach().numpy(), float(SC.abs().max())


def run_listing(x_f, pos, weights, rms1_f, rms2_f, extra_config=""):
    wq_f, wk_f, wv_f, wo_f, wup_f, wgate_f, wdown_f = weights
    text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "programs", "xf_block.asm")).read()
    feeds = ASM.run_text(extra_config + text, REGISTRY,
                         {"x": enc(x_f), "pos": pos,
                          "wq": enc(wq_f), "wk": enc(wk_f),
                          "wv": enc(wv_f), "wo": enc(wo_f),
                          "wup": enc(wup_f), "wgate": enc(wgate_f),
                          "wdown": enc(wdown_f),
                          "rms_w1": enc(rms1_f),
                          "rms_w2": enc(rms2_f)}, sigs=SIGS)
    return feeds


def psnr1(got, ref):
    mse = float(np.mean((got - ref) ** 2))
    return (float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)), mse


def main():
    # barred row: in-contract envelope (S=8, D=16, Dff=32, scale 0.10)
    x_f, pos, weights, r1, r2 = fixture()
    ref, scmax = torch_block(x_f, pos, weights, r1, r2)
    check("block-scores-in-contract", scmax <= 1.0,
          f"scoremax={scmax:.3f} (softmax contract; proves barred row non-vacuous)")
    feeds = run_listing(x_f, pos, weights, r1, r2)
    got = dec(feeds["OUT"])
    d, mse = psnr1(got, ref)
    check("block-parity", d >= BAR_DB,
          f"{d:.2f}dB (bar {BAR_DB}, mse={mse:.2e}, outmax={np.abs(ref).max():.3f})")
    # determinism: same payload twice, bit-identical triples
    feeds2 = run_listing(x_f, pos, weights, r1, r2)
    same = all(bool((feeds["OUT"][k] == feeds2["OUT"][k]).all()) for k in (0, 1, 2))
    check("block-deterministic", same, "two runs bit-identical (no hidden state)")

    # measured row: out-of-contract envelope (scale 0.30, scores ~3.5).
    # Not barred -- documents where frozen holo scales stop covering
    # (softmax saturation above 1.0 at BIAS + matmul products over m_acc).
    # Tripwire: assert the row actually leaves the contract (guards vacuity).
    x3, pos3, w3, r13, r23 = fixture(scale=0.30)
    ref3, scmax3 = torch_block(x3, pos3, w3, r13, r23)
    got3 = dec(run_listing(x3, pos3, w3, r13, r23)["OUT"])
    d3, mse3 = psnr1(got3, ref3)
    check("block-out-of-contract-vacuity", scmax3 > 1.0,
          f"scoremax={scmax3:.2f} > 1.0 (row truly out-of-contract)")
    print(f"block-out-of-contract-measured: {d3:.2f}dB (mse={mse3:.2e}, "
          f"mechanism: scores>1 saturate + products over m_acc; backlog: "
          f"recalibrated scales + 1/sqrt(d) scaling)")

    # mixed-scale row (roadmap gate 1): attention small (scores in-contract,
    # frozen-fine) + MLP big (products over frozen m_acc). Regimes differ
    # BY BLOCK (attn values ~0.4, mlp ~0.6); one listing spans both via
    # CONFIG m_acc (matmul family bridges big, everything else frozen).
    # m_acc 35492 = m_of(8.0), frozen literal (stated, cf #LIB-030).
    xm, posm, wm, r1m, r2m = fixture(scale=0.10, mlp_scale=0.30)
    refm, scmaxm = torch_block(xm, posm, wm, r1m, r2m)
    check("block-mixed-contract", scmaxm <= 1.0,
          f"scoremax={scmaxm:.3f} (attention in-contract throughout)")
    gotm_frozen = dec(run_listing(xm, posm, wm, r1m, r2m)["OUT"])
    dm_f, _ = psnr1(gotm_frozen, refm)
    gotm = dec(run_listing(xm, posm, wm, r1m, r2m,
                           extra_config="CONFIG m_acc 35492\n")["OUT"])
    dm, msem = psnr1(gotm, refm)
    check("block-mixed-scales", dm >= BAR_DB and dm_f < BAR_DB,
          f"mixed {dm:.1f}dB vs frozen {dm_f:.1f}dB (row non-vacuous)")
    check("block-mixed-touched", bool((gotm != gotm_frozen).any()),
          "override moves values (tripwire against silent ignore)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
