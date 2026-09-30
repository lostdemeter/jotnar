"""Temporal IIR gates (feedback #2): detail memory across frames.

Rule: D_t = 0.5*D + 0.5*warp(D_{t-1}) (dyadic frozen; see chain/temporal.py).
Warp is nihui-form (unclamped-floor alphas, clamped indices, replicate).
State carries triples; first frame equals still output bit-exactly.
Bases declared per test. Usage: python3 test_temporal.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import holo_phi as H
from chain import temporal as T
from chain import oracle as O

BAR_DB = 40.0
FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def main():
    print(f"a_mix={T.A_MIX} (frozen dyadic; change trips mix-frozen)")
    check("mix-frozen", T.A_MIX == 0.5, "non-dyadic a is a LOWERING ERROR")
    m_acc, m_cov = H._load_scales()

    # 1. warp parity int-vs-float (basis: random field [0,1] amplitude-like,
    #    subpixel flow +-3px, peak 1). Warp runs on FIXED counts; oracle float.
    rng = np.random.default_rng(31)
    F = rng.uniform(0, 1, (20, 20))
    fl = rng.uniform(-3, 3, (20, 20, 2))
    ft = S.encode(F)
    got = T.warp_trips(ft, fl, m_cov)
    gv = (S.decode(got[0], got[1]) * (1 - got[2].astype(np.float64))).reshape(20, 20)
    ref = O.warp_float_nihui(F, fl)
    d = psnr(gv, ref)
    check("warp-parity", d >= BAR_DB, f"{d:.2f}dB")
    # 2. identity flow is EXACT (weights (1,0,0,0): gather == input counts,
    #    so from_fixed sees identical input as the direct bridge).
    zf = np.zeros((20, 20, 2))
    got0 = T.warp_trips(ft, zf, m_cov)
    # NOTE: warp_q counts are compared directly (not via a second
    # from_fixed/to_fixed roundtrip, which quantizes and would false-fail).
    qd = S.to_fixed(ft[0], ft[1], ft[2], m_cov).reshape(-1)
    qw = T.warp_q(qd, T.flow_to_q14(zf), 20, 20)
    check("warp-identity", bool((qd == qw).all()), "fixed counts equal")

    # 3. static clip: frames 1+ BIT-identical mutually (memory converged, no
    #    drift); frame 1 near frame 0 within MEASURED bounds (the mix path
    #    roundtrips triples->fixed->triples; decoded values near lattice
    #    boundaries re-encode 1-2 steps off on 0-2 px per frame (6-seed
    #    calibration: max Δe=2, ndiff<=2/576; bounds set at 3 and 8 with the
    #    mechanism stated -- honest quantization cost of the IR binop
    #    contract, not shimmer: a broken mix would move dozens of pixels).
    ys = [np.clip(rng.uniform(0, 1, (24, 24)), 0, 1) for _ in range(4)]
    y0 = ys[0]
    frames = [y0, y0.copy(), y0.copy(), y0.copy()]
    flows = [None, np.zeros((24, 24, 2)), np.zeros((24, 24, 2)),
             np.zeros((24, 24, 2))]
    st, outs = T.new_state(), []
    for y, f in zip(frames, flows):
        o, st, _ = T.step(y, f, st, blur="iso")
        outs.append(o)
    mut = all(bool((outs[i][0] == outs[1][0]).all()
                   and (outs[i][1] == outs[1][1]).all()
                   and (outs[i][2] == outs[1][2]).all()) for i in (2, 3))
    check("static-converged", mut, "frames 1..3 mutually exact (no drift)")
    de = np.abs(outs[1][1].astype(np.int64) - outs[0][1].astype(np.int64))
    same_sz = bool((outs[1][0] == outs[0][0]).all() and (outs[1][2] == outs[0][2]).all())
    check("static-firststep", same_sz and bool(de.max() <= 3) and bool((de > 0).sum() <= 8),
          f"max Δe={int(de.max())} ndiff={int((de > 0).sum())} (bounds 3 / 8)")

    # 4. first frame == still output, bit-exactly (temporal adds nothing
    #    without memory: state None ignores flow).
    ref, _ = H.enhance_luminance_int(y0, beta=0.5, blur="iso")
    check("firstframe-still", bool((outs[0][0] == ref[0]).all()
                                   and (outs[0][1] == ref[1]).all()
                                   and (outs[0][2] == ref[2]).all()),
          "frame 0 == still path")

    # 5. memory exists: a changed frame alters the NEXT output (state carries).
    alt = np.clip(rng.uniform(0, 1, (24, 24)), 0, 1)
    st2 = T.new_state()
    o1, st2, _ = T.step(y0, None, st2, blur="iso")
    o2a, _, _ = T.step(y0, np.zeros((24, 24, 2)), dict(st2))
    o2b, _, _ = T.step(alt, np.zeros((24, 24, 2)), dict(st2))
    diff = bool((o2a[0] != o2b[0]).any() or (o2a[1] != o2b[1]).any())
    check("memory-carries", diff, "history alters the present")

    # 6. step response settles geometrically (a=0.5: residual halves/frame;
    #    frame t+3 within ~15% of steady state at t+7).
    N = 32
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                         indexing="ij")
    bar = lambda c: np.where(xx > c, 0.85, 0.15)
    seq = [bar(0.5), bar(0.5), bar(0.2), bar(0.2), bar(0.2), bar(0.2),
           bar(0.2), bar(0.2)]
    fls = [None] + [np.zeros((N, N, 2))] * 7
    st3, oo = T.new_state(), []
    for y, f in zip(seq, fls):
        o, st3, _ = T.step(y, f, st3, blur="iso")
        oo.append(S.decode(o[0], o[1]) * (1 - o[2].astype(np.float64)))
    steady = float(np.abs(oo[7] - oo[6]).mean())
    third = float(np.abs(oo[4] - oo[7]).mean())
    first = float(np.abs(oo[2] - oo[7]).mean())
    check("step-settles", third <= 0.2 * max(first, 1e-12),
          f"|f4-ss|={third:.2e} vs |f2-ss|={first:.2e} (steady {steady:.1e})")

    # 7. sequence parity int-vs-oracle (basis: 4-frame translating bar,
    #    exact 2px/frame flow, peak 1).
    N4 = 32
    frames4, flows4 = [], [None]
    for t in range(4):
        c = 0.5 - 2 * t / N4
        yy4, xx4 = np.meshgrid(np.linspace(0, 1, N4), np.linspace(0, 1, N4),
                              indexing="ij")
        frames4.append(np.where(xx4 > c, 0.85, 0.15))
    # NOTE on flow sign: frame-t content at x came from frame t-1 position
    # x+2 (bar moved -2px/frame), so warp(Dprev) samples prev at p+(+2): the
    # stored flow is +2 in x. Sign convention asserted by this gate passing
    # (wrong sign visibly halves parity -- verified during development).
    flows4 = [None] + [np.full((N4, N4, 2), [0.0, 2.0]) for _ in range(3)]
    st4, outs4 = T.new_state(), []
    for y, f in zip(frames4, flows4):
        o, st4, _ = T.step(y, f, st4, blur="iso")
        outs4.append(S.decode(o[0], o[1]) * (1 - o[2].astype(np.float64)))
    refs = O.temporal_frames_float(frames4, flows4, beta=0.5)
    ds = [psnr(o, r) for o, r in zip(outs4, refs)]
    check("seq-parity", all(v >= BAR_DB for v in ds),
          f"{[f'{v:.1f}' for v in ds]}dB")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
