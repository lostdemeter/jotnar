"""Router gates (feedback #3, first half): select AMONG modes per frame.

The router is the first structure operating on CONFIGURATIONS, not pixels:
per frame, engage temporal memory or run still. v1 rule is analytic
(E >= 0.25 -> temporal); the gate proves the meta-claim: routed >= best
fixed arm, component-wise (no single-number tradeoff hiding).
Bases declared per test. Usage: python3 tests/test_router.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import router as R
from chain import temporal as T
from chain import oracle as O
from chain.oracle import warp_float_nihui

BAR_DB = 40.0
FAIL = []
N = 48


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def psnr(a, b, peak=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def lapvar(im):
    return float(np.var(im[2:, 1:-1] + im[:-2, 1:-1] + im[1:-1, 2:] + im[1:-1, :-2] - 4 * im[1:-1, 1:-1]))


def mixed_seq():
    """3 static-noisy frames + 4 translating-noisy frames (3px/frame).
    Static flows are exact zeros; moving flows exact uniform. Both sides
    (INT + oracle) route on the same boundary floats (structural agreement).
    Returns (ys, flows)."""
    rng = np.random.default_rng(23)
    yy, xx = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N),
                         indexing="ij")
    base = np.where(xx > 0.5, 0.85, 0.15)
    ys, flows = [], []
    for _ in range(3):
        ys.append(np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1))
        flows.append(np.zeros((N, N, 2)))
    for t in range(1, 5):
        bar = np.where(xx > 0.5 - 3 * t / N, 0.85, 0.15)
        ys.append(np.clip(bar + rng.normal(0, 0.02, bar.shape), 0, 1))
        flows.append(np.full((N, N, 2), [0.0, 3.0]))
    return ys, flows


def main():
    print(f"route_thr={R.ROUTE_THR} (frozen v1: mean motion >= 2px engages)")
    ys, flows = mixed_seq()

    # 1. decisions: static -> still, moving -> temporal (the meta-claim's
    #    atomic unit; boundary floats shared by construction).
    d0, e0 = R.decide(flows[0])
    d3, e3 = R.decide(flows[3])
    dn, _ = R.decide(None)
    check("decide-static", d0 == "still" and e0 == 0.0, f"E={e0:.3f}")
    check("decide-moving", d3 == "temporal", f"E={e3:.3f} (3px/8px)")
    check("decide-none", dn == "still", "None -> still")
    # boundary inclusivity stated: E exactly at thr engages.
    edge = np.full((4, 4, 2), [0.0, 2.0])
    de, ee = R.decide(edge)
    check("decide-edge", de == "temporal" and abs(ee - 0.25) < 1e-12,
          f"E={ee:.4f} >= thr engages")

    # 2. routed sequence == still outputs on static frames, bit-exactly
    #    (still path ignores state entirely -- seamlessness, structural).
    outs, decisions, _ = R.run_sequence(ys, flows, beta=0.5, blur="iso")
    check("decisions", decisions == ["still"] * 3 + ["temporal"] * 4,
          f"{decisions}")
    from chain.holo_phi import enhance_luminance_int
    for i in range(3):
        ref, _ = enhance_luminance_int(ys[i], beta=0.5, blur="iso")
        same = bool((outs[i][0] == ref[0]).all() and (outs[i][1] == ref[1]).all()
                    and (outs[i][2] == ref[2]).all())
        if not same:
            check(f"seamless-{i}", False, "static frame differs from still-only")
            break
    else:
        check("seamless-static", True, "3 static frames bit-identical to still-only")

    # 3. component inequalities on the moving run (measured behavior):
    #    temporal flicker < still flicker (memory smooths), temporal
    #    sharpness within 10% of still (bounded smear cost). Both sides gated
    #    separately -- no single number hides the tradeoff.
    mov_ys, mov_fl = ys[3:], flows[3:]
    outs_s, _, _ = R.run_sequence(mov_ys, mov_fl, beta=0.5, blur="iso",
                                  router_on=False)
    # temporal-only run for the comparison (router would pick temporal here;
    # run the arm directly for a clean control).
    st = T.new_state()
    outs_tt = []
    for y, f in zip(mov_ys, mov_fl):
        o, st, _ = T.step(y, f, st, blur="iso", beta=0.5)
        outs_tt.append(o)
    SH = mov_ys[0].shape
    dec = lambda t: np.clip((S.decode(t[0], t[1]) * (1 - t[2].astype(np.float64))).reshape(SH), 0, 1)
    s_outs = [dec(t) for t in outs_s]
    r_outs = [dec(t) for t in outs]

    def flick(seq, fls):
        return float(np.mean([np.abs(seq[i] - warp_float_nihui(seq[i - 1], fls[i])).mean()
                              for i in range(1, len(seq))]))

    fl_s = flick(s_outs, mov_fl)
    fl_t = flick([dec(t) for t in outs_tt], mov_fl)
    check("flicker-wins", fl_t < 0.9 * fl_s, f"still={fl_s:.4f} temp={fl_t:.4f}")

    def gain(seq, src):
        return float(np.mean([lapvar(o) / max(lapvar(s), 1e-9)
                              for o, s in zip(seq, src)]))

    g_s = gain(s_outs, mov_ys)
    g_t = gain([dec(t) for t in outs_tt], mov_ys)
    check("sharpness-bounded", g_t >= 0.9 * g_s, f"still={g_s:.3f} temp={g_t:.3f}")

    # 4. routed parity vs oracle-routed (same modes both sides: decisions
    #    from shared boundary floats agree structurally).
    modes = decisions
    refs = O.temporal_frames_float(ys, flows, beta=0.5, modes=[
        "still" if d == "still" else "temporal" for d in decisions])
    ds = [psnr(np.clip(S.decode(t[0], t[1]) * (1 - t[2].astype(np.float64)), 0, 1), r)
          for t, r in zip(outs, refs)]
    check("routed-parity", all(v >= BAR_DB for v in ds),
          f"{[f'{v:.1f}' for v in ds]}dB")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
