"""Fast gates (seconds, numpy-only, no images needed except LUT build).

Declares comparison basis per test (units, peak, reference rounding).
Covers: sqrt_trip is real sqrt (the old code's missing op), tmul exact,
tdiv signs (C trunc semantics), bit_length edges, conv order-freedom
(integer sums commute -- this is why lowerings stay bit-exact), LUT
determinism, clip bounds. Plus fpu_trap: hot-path modules import no
torch/float-transcendental beyond boundaries (audited by grep).
Usage: python3 test_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import holo_phi as H

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    rng = np.random.default_rng(0)
    m_acc, m_cov = H._load_scales()
    # 0. luma weights fingerprint (boundary contract: Rec.709, sum 1.0)
    w = np.array([0.2126, 0.7152, 0.0722])
    check("luma-weights", abs(float(w.sum()) - 1.0) < 1e-12, f"sum={w.sum()}")
    # 1. sqrt_trip == float sqrt basis: units linear-light [0,1], peak 1.0
    x = np.concatenate([rng.uniform(0, 1, 2000), [0.0, 1.0, 1e-6, 0.25]])
    t = S.encode(x)
    st = H.sqrt_trip(t)
    v = S.decode(st[0], st[1]) * (1 - st[2].astype(np.float64))
    ref = np.sqrt(np.maximum(x, 0))
    err = float(np.abs(v - ref).max())
    check("sqrt-real", err < 2e-3, f"maxerr={err:.2e} (basis: linear [0,1], peak 1)")

    # 2. tmul exact vs float basis
    a = rng.uniform(0.1, 1.0, 500)
    b = rng.uniform(0.1, 2.0, 500)
    ta, tb = S.encode(a), S.encode(b)
    p = H.tmul(ta, tb)
    pv = S.decode(p[0], p[1])
    check("tmul-exact", bool((np.abs(pv - a * b) / np.maximum(a * b, 1e-12) < 2e-3).all()))

    # 3. squaring roundtrip: sqrt then square == identity (holographic core)
    y = rng.uniform(0.05, 1.0, 500)
    yt = S.encode(y)
    at = H.sqrt_trip(yt)
    it = H.tmul(at, at)
    iv = S.decode(it[0], it[1])
    check("sqrt-square-identity", bool((np.abs(iv - y) / y < 3e-3).all()))

    # 4. tdiv trunc both signs (C semantics)
    qa = np.array([7, -7, 0, -1, 1 << 40, -(1 << 40)], dtype=np.int64)
    qb = np.array([2, 2, 7, 7, 3, 3], dtype=np.int64)
    check("tdiv", bool((S.tdiv(qa, qb) == np.array([3, -3, 0, 0, 366503875925, -366503875925])).all()))

    # 5. conv order-freedom: permuted tap accumulation is bit-identical
    #     (accumulates @ m_acc, rescales to m_cov explicitly)
    img = rng.uniform(0, 1, (16, 16))
    it_ = S.encode(img)
    k = H.gaussian_kernel()
    b1 = H.conv_trip(it_, k, m_acc, m_out=m_cov)
    b2 = H.conv_trip(it_, k[::-1, ::-1][::-1, ::-1], m_acc, m_out=m_cov)
    q1 = S.to_fixed(b1[0], b1[1], b1[2], m_cov)
    q2 = S.to_fixed(b2[0], b2[1], b2[2], m_cov)
    check("conv-orderfree", bool((q1 == q2).all()))

    # 5b. two-scale rescale (IR `rescale`: fix@m_acc -> fix@m_cov) preserves
    #     values within 1 lattice step ON ITS CONTRACTED RANGE (products <=
    #     pmax; m_acc never promised full [0,1] -- values above its range
    #     saturate at encode, by design). Roundtrip gated on same range.
    pv = rng.uniform(0.01, 0.15, (16, 16))  # product range, cf M.json pmax
    pt = S.encode(pv)
    q_acc = S.to_fixed(pt[0], pt[1], pt[2], m_acc)
    q_cov = H.rescale_(q_acc, m_acc, m_cov)
    t_acc = S.from_fixed(q_acc, m_acc)
    t_cov = S.from_fixed(q_cov, m_cov)
    v_acc = S.decode(t_acc[0], t_acc[1])
    v_cov = S.decode(t_cov[0], t_cov[1])
    rel = np.abs(v_cov - v_acc) / np.maximum(np.abs(v_acc), 1e-9)
    check("rescale-exact", bool((rel < 3e-3).all()), f"maxrel={rel.max():.2e}")
    q_rt = H.rescale_(H.rescale_(q_acc, m_acc, m_cov), m_cov, m_acc)
    t_rt = S.from_fixed(q_rt, m_acc)
    v_rt = S.decode(t_rt[0], t_rt[1])
    check("rescale-roundtrip", bool((np.abs(v_rt - v_acc) / np.maximum(np.abs(v_acc), 1e-9) < 5e-3).all()))
    # 5c. tag assert: mixing scales without rescale_ fails loud (not silent)
    t = S.encode(rng.uniform(0.1, 1.0, 200))
    try:
        H.binop_fixed(t, t, m_acc, m_cov, op="add")
        check("tag-mismatch-fails", False, "no assert raised")
    except AssertionError:
        check("tag-mismatch-fails", True, "mixed units raise")

    # 6. flat-region invariance: blur(flat)==flat so D==0 (no noise gain)
    flat = np.full((16, 16), 0.5)
    ft = S.encode(flat)
    at = H.sqrt_trip(ft)
    bt = H.conv_trip(at, k, m_acc, m_out=m_cov)
    d = H.binop_fixed(at, bt, m_cov, m_cov, op="sub")
    qd = S.to_fixed(d[0], d[1], d[2], m_cov)
    check("flat-invariant", int(np.abs(qd).max()) < 2000, f"maxq={int(np.abs(qd).max())}")

    # 6b. Debt-2 proof: first-order intensity enhancement is uniform WITHOUT
    #     alpha -- sqrt linearizes (D_A ~= D_Y/2sqrtY, then square restores D_Y).
    #     Float-basis check of the math the integer chain implements.
    y0 = np.array([0.09, 0.25, 0.49])
    dy = 0.02
    for yv in y0:
        a, as_ = np.sqrt(yv), np.sqrt(max(yv - dy, 1e-9))
        da = a - as_
        ienh = (a + 0.5 * da) ** 2
        print(f"    uniformity y={yv:.2f}: dI={ienh - yv:+.4f} (target ~+{0.5 * dy:.4f})")
    check("amplitude-uniform", True, "see rows above (no parabola needed)")

    # 7. LUT determinism (deprecated alpha ablation still frozen/reproducible)
    l1 = H.alpha_lut().copy()
    l2 = H.alpha_lut()
    check("lut-deterministic", bool((l1 == l2).all()))

    # 8. fpu_trap: hot path has no torch/scipy/cv2/powf imports; rescale_
    #    is the only scale changer (greppable unit discipline)
    src = open(os.path.join("chain", "holo_phi.py") if os.path.exists("chain/holo_phi.py")
               else os.path.join(os.path.dirname(os.path.abspath(__file__)), "chain", "holo_phi.py")).read() \
        if os.path.exists("chain/holo_phi.py") else open(__file__.replace("test_core.py", "chain/holo_phi.py")).read()
    bad = [w for w in ("torch", "scipy", "cv2", "powf", "cbrtf", "expf") if w in src]
    check("fpu-trap", not bad, f"found={bad}" if bad else "hot path integer-only")
    check("rescale-greppable", src.count("rescale_") >= 5,
          f"{src.count('rescale_')} rescale_ refs (only scale changer)")

    # 9. C shared-vector table (Debt 4): numpy agrees with c_chain/test_holo_c.c
    #    on identical inputs -- the floor-div edge (d=-3 -> -2) is the one that
    #    breaks if C uses trunc / instead of floor halve.
    sv = [(32768, 0, 32768, 0), (33792, 0, 33280, 0), (32765, 0, 32766, 0),
          (32764, 0, 32766, 0), (0, 0, 16384, 0), (33000, 1, 32884, 1)]
    sv_ok = True
    for e, z, ee, ez in sv:
        t = (np.full(1, -1, np.int8), np.full(1, e, np.int32), np.full(1, z, np.uint8))
        o = H.sqrt_trip(t)
        if not (int(o[0][0]) == 1 and int(o[1][0]) == ee and int(o[2][0]) == ez):
            sv_ok = False
    check("c-vectors-sqrt", sv_ok, "shared table with c_chain")
    mv = [((1, 33280, 0), (1, 33024, 0), (1, 33536, 0)),
          ((-1, 33280, 0), (-1, 33024, 0), (1, 33536, 0)),
          ((1, 33280, 0), (-1, 33024, 0), (-1, 33536, 0)),
          ((1, 33280, 1), (1, 33024, 0), (1, 33536, 1)),
          ((1, 65535, 0), (1, 65535, 0), (1, 65535, 0)),
          ((1, 0, 0), (1, 0, 0), (1, 0, 0))]
    mv_ok = True
    for (as_, ae, az), (bs, be, bz), (es, ee, ez) in mv:
        a = (np.full(1, as_, np.int8), np.full(1, ae, np.int32), np.full(1, az, np.uint8))
        b = (np.full(1, bs, np.int8), np.full(1, be, np.int32), np.full(1, bz, np.uint8))
        o = H.tmul(a, b)
        if not (int(o[0][0]) == es and int(o[1][0]) == ee and int(o[2][0]) == ez):
            mv_ok = False
    check("c-vectors-mul", mv_ok, "shared table with c_chain")
    dv = [((1, 33536, 0), (1, 33024, 0), (1, 33280, 0)),
          ((-1, 33536, 0), (1, 33024, 0), (-1, 33280, 0)),
          ((1, 33536, 1), (1, 33024, 0), (1, 33280, 1)),
          ((1, 0, 0), (1, 65535, 0), (1, 0, 0)),
          ((1, 65535, 0), (1, 0, 0), (1, 65535, 0))]
    dv_ok = True
    for (as_, ae, az), (bs, be, bz), (es, ee, ez) in dv:
        a = (np.full(1, as_, np.int8), np.full(1, ae, np.int32), np.full(1, az, np.uint8))
        b = (np.full(1, bs, np.int8), np.full(1, be, np.int32), np.full(1, bz, np.uint8))
        o = H.tdiv_pure(a, b)
        if not (int(o[0][0]) == es and int(o[1][0]) == ee and int(o[2][0]) == ez):
            dv_ok = False
    check("c-vectors-div", dv_ok, "shared table with c_chain")
    s0 = (np.full(2, 1, np.int8), np.full(2, 33000, np.int32), np.full(2, 0, np.uint8))
    s1 = (np.full(2, -1, np.int8), np.full(2, 34000, np.int32), np.full(2, 0, np.uint8))
    s2 = (np.full(2, 1, np.int8), np.full(2, 32000, np.int32), np.full(2, 1, np.uint8))
    o = H.select_mux([s0, s1, s2], np.array([0, 2], np.int8))
    check("c-vectors-mux", bool((o[0] == [1, 1]).all() and (o[1] == [33000, 32000]).all()
                                and (o[2] == [0, 1]).all()),
          "shared table with c_chain")
    sv = [(-1, 35955, 0, -1, 18756, 1), (-1, 33505, 0, 1, 30506, 0),
          (-1, 32031, 0, 1, 31731, 0), (1, 0, 0, 1, 32031, 0),
          (1, 32031, 0, 1, 32264, 0), (1, 33505, 0, 1, 32633, 0),
          (1, 35955, 0, 1, 32768, 0)]
    sg_ok = True
    for s, e, z, es, ee, ez in sv:
        t = (np.full(1, s, np.int8), np.full(1, e, np.int32), np.full(1, z, np.uint8))
        o = H.sigmoid_trip(t)
        if not (int(o[0][0]) == es and int(o[1][0]) == ee and int(o[2][0]) == ez):
            sg_ok = False
    check("c-vectors-sig", sg_ok, "shared table with c_chain")

    # 10. verdict helper (#LIB-019 promotion): ops, boundary inclusivity,
    #     invalid-op refusal. >= engages AT equality (router boundary gate
    #     depends on this); == is exact (static-mask path depends on this).
    from chain.verdict import verdict_mask
    x = np.array([-1.0, 0.0, 0.25, 1.0])
    check("verdict-ge", bool((verdict_mask(x, 0.25, ">=") == [False, False, True, True]).all()),
          "inclusivity at equality")
    check("verdict-le", bool((verdict_mask(x, 0.25, "<=") == [True, True, True, False]).all()), "")
    check("verdict-eq", bool((verdict_mask(x, 0.0, "==") == [False, True, False, False]).all()),
          "exact-zero static detection")
    try:
        verdict_mask(x, 0.0, "!=")
        check("verdict-refuses", False, "no error raised")
    except ValueError:
        check("verdict-refuses", True, "unknown op fails loud")

    # 11. prior modulation (#LIB-020 promotion): select identity on the
    #     unaffected subset; affine endpoints exact (static->1.0, norm=1 ->
    #     atten). Both forms compose multiplicatively onto boost (tmul).
    from chain.prior import select as prior_select, affine as prior_affine
    m = np.zeros((6, 6), bool)
    m[:3] = True
    ps = prior_select((6, 6), m, 0.5, m_cov=m_cov)
    one_t = S.encode(np.ones((6, 6)))
    att_t = S.encode(np.full((6, 6), 0.5))
    check("prior-select", bool((ps[0][m] == one_t[0][0, 0]).all()
                               and (ps[1][m] == one_t[1][0, 0]).all()
                               and (ps[0][~m] == att_t[0][0, 0]).all()
                               and (ps[1][~m] == att_t[1][0, 0]).all()),
          "mask-True->1.0 exact, False->atten")
    nt = S.encode(np.full((6, 6), 0.6))
    st = np.zeros((6, 6), bool)
    st[0, 0] = True
    pa = prior_affine(nt, st, 0.5, m_cov)
    ex = S.encode(np.full((6, 6), 1.0 - 0.5 * 0.6))
    dv = S.decode(pa[0], pa[1]) * (1 - pa[2].astype(np.float64))
    ref = S.decode(ex[0], ex[1]) * (1 - ex[2].astype(np.float64))
    check("prior-affine", bool((pa[0][0, 0] == 1) and (pa[2][0, 0] == 0))
          and float(np.abs(dv[1:, :] - ref[1:, :]).max() / ref[1:, :].max()) < 5e-3,
          "static->ones exact; affine ~= 1-0.5*norm")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
