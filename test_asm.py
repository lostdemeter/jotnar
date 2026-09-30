"""Assembly fidelity gates (stage 1): the listing IS the program.

test_asm proves the assembly level FAITHFUL: running programs/holo_flagship.asm
produces bit-exact output vs the hand-written chain path (same functions, but
the TEXT fully specifies the computation -- no hidden Python). Plus assembler
discipline gates (unknown mnemonic / arity / use-before-def fail at assemble
time, never mid-run). Stage 2 (generative proof) gets its own gate file.
Usage: python3 test_asm.py
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.holo_phi import enhance_image_int

FAIL = []
CAND = "/home/thorin/Documents/OpenCode/rife_reverse/samples/f_012.png"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "programs", "holo_flagship.asm")) as fh:
        text = fh.read()
    rgb = np.asarray(Image.open(CAND).convert("RGB"))
    feeds = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS)
    got = feeds["OUT"]

    # hand-written flagship path (demo.py --blur splat_soft --ctrl v5)
    lin = np.power(rgb.astype(np.float64) / 255.0, 2.2).astype(np.float32)
    ref, _, _ = enhance_image_int(lin, beta=0.5, blur="splat_soft", ctrl="v5")
    ref8 = np.clip(np.power(np.clip(ref, 0, 1), 1.0 / 2.2) * 255.0,
                   0, 255).astype(np.uint8)
    check("asm-faithful", bool((got == ref8).all()),
          "listing output bit-exact vs hand-written chain")

    # assembler discipline: all failures at assemble time
    for bad, tag in [
        ("IN x\nOUT = NOPE(x)\n", "unknown-mnemonic"),
        ("IN x\nY = LUMA(x, y)\n", "arity"),
        ("IN x\nY = LUMA(nope)\n", "use-before-def"),
        ("OUT = LUMA(x)\n", "missing-in"),
    ]:
        try:
            ASM.run_text(bad, REGISTRY, np.zeros((2, 2, 3), np.uint8), sigs=SIGS)
            check(f"asm-{tag}", False, "assembled without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", True, f"fails loud ({str(e)[:60]})")

    # Batch 1 exposure wiring (GAPS.md): wrappers add NOTHING over phi-core
    # fns (0-diff with identical args); softmax normalization gated vs float
    # within its contract (inputs <= 1.0 abs -- to_fixed saturates above it,
    # the T-transformation doctrine; out-of-contract behavior pinned, not barred).
    import phi_core.lattice as S
    from phi_core import numpy_ops as N
    from phi_core.calibrate import m_of
    import chain.holo_phi as H
    rng = np.random.default_rng(0)
    ma, _ = H._load_scales()
    A = S.encode((rng.random((2, 4, 8)) - 0.5) * 6)
    B = S.encode((rng.random((8, 6)) - 0.5) * 2)
    check("asm-matmul", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["MATMUL"][0]([A, B], {}, {}), N.matmul_int(A, B, ma))),
        "0-diff vs phi-core (same m_acc)")
    t = S.encode(rng.uniform(-4, 4, (5, 16)))
    check("asm-silu", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["SILU"][0]([t], {}, {}), N.silu_int(t))), "0-diff")
    x = S.encode((rng.random((4, 32)) - 0.5) * 6)
    w = S.encode((rng.random(32) - 0.5) * 2 + 0.5)
    _, mc = H._load_scales()
    check("asm-rmsnorm", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["RMSNORM"][0]([x, w], {"eps_rms_c": "4514"}, {}),
        N.rmsnorm_int(*x, w, mc, 4514))), "0-diff at same (m, eps)")
    rows = [[-0.5, 0., 0.5], [0.2, 0.2, 0.2], [0.9, -0.9, 0.], [0., 0., 0.]]
    s = S.encode(np.array(rows))
    g = REGISTRY["SOFTMAX"][0]([s], {}, {})
    v = S.decode(g[0], g[1]) * (1 - g[2].astype(np.float64))
    ref = np.stack([(lambda r: np.exp(r - r.max()) / np.exp(r - r.max()).sum())(
        np.asarray(r)) for r in rows])
    check("asm-softmax", float(np.abs(v - ref).max()) < 1e-3,
          f"maxabsdiff={float(np.abs(v - ref).max()):.2e} (in-contract)")
    big = S.encode(np.array([[3., -3., 0.]]))
    gb = REGISTRY["SOFTMAX"][0]([big], {}, {})
    vb = S.decode(gb[0], gb[1]) * (1 - gb[2].astype(np.float64))
    clip = np.exp(np.array([1., -1., 0.]) - 1.)
    clip = clip / clip.sum()
    check("asm-softmax-saturates", float(np.abs(vb[0] - clip).max()) < 1e-3,
          "out-of-contract inputs saturate to softmax([1,-1,0]) (pinned)")

    # Batch 2: ROTARY (first genuinely-new structure) + BATCH_MATMUL exposure.
    # ROTARY gates prove the MEANING, not just the math: vs float RoPE,
    # norm preservation (rotation is an isometry -- geometric invariant),
    # relative-position property (dot depends on p-q only). Fixtures stay
    # inside frozen m_cov coverage (values in [-1,1]: holo calibrated
    # amax=1.0; transformer magnitudes need RECALIBRATED scales via the
    # defined procedure, not a code change -- the gate below proved the
    # current data doesn't cover transformers, which is scales doctrine
    # working, not an op bug).
    from chain.asm_ops import rope_tables
    rng2 = np.random.default_rng(1)
    D, NP = 16, 24
    xr = S.encode((rng2.random((NP, D)) - 0.5) * 2.0)
    pos = np.arange(NP, dtype=np.int64)
    got = REGISTRY["ROTARY"][0]([xr, pos], {}, {})
    gv = S.decode(got[0], got[1]) * (1 - got[2].astype(np.float64))
    xv = S.decode(xr[0], xr[1]) * (1 - xr[2].astype(np.float64))
    cos_t, sin_t = rope_tables(NP, D)
    ref = np.empty_like(xv)
    ref[:, 0::2] = xv[:, 0::2] * cos_t - xv[:, 1::2] * sin_t
    ref[:, 1::2] = xv[:, 0::2] * sin_t + xv[:, 1::2] * cos_t
    check("asm-rotary", float(np.abs(gv - ref).max() / max(np.abs(ref).max(), 1e-9)) < 5e-3,
          f"maxrelerr={float(np.abs(gv - ref).max() / max(np.abs(ref).max(), 1e-9)):.2e} vs float RoPE")
    n0 = np.sqrt((xv ** 2).sum(-1))
    n1 = np.sqrt((gv ** 2).sum(-1))
    check("asm-rotary-norm", float(np.abs(n1 - n0).max() / max(n0.max(), 1e-9)) < 5e-3,
          "rotation preserves norms (isometry)")
    d1 = float((gv[5] * gv[9]).sum())
    d2 = float((ref[5] * ref[9]).sum())
    check("asm-rotary-relpos", abs(d1 - d2) / max(abs(d2), 1e-9) < 5e-3,
          "cross-position dots match (relative geometry)")
    A2 = S.encode((rng2.random((2, 4, 8)) - 0.5) * 6)
    B2 = S.encode((rng2.random((8, 6)) - 0.5) * 2)
    check("asm-batch-matmul", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["BATCH_MATMUL"][0]([A2, B2], {}, {}), N.matmul_int(A2, B2, ma))),
        "0-diff batched+broadcast (same m_acc)")
    tp = S.encode((rng2.random((2, 4, 8)) - 0.5) * 2)
    gtp = REGISTRY["TRANSPOSE"][0]([tp], {}, {})
    check("asm-transpose", bool((gtp[0] == np.swapaxes(tp[0], -1, -2)).all()
                                and gtp[0].shape == (2, 8, 4)),
          "exact last-two-axes swap")
    gtt = REGISTRY["TRANSPOSE"][0]([gtp], {}, {})
    check("asm-transpose-involution", all(bool((a == b).all()) for a, b in zip(gtt, tp)),
          "transpose twice == identity (exact move)")
    # SPLIT family v1: reshape/permute exact moves (N-way split deferred: no
    # demand -- heads need reshape+permute only, stated).
    import numpy as _np
    tr = S.encode((rng2.random((2, 12)) - 0.5) * 2)
    gr = REGISTRY["RESHAPE3"][0]([tr, 2.0, 3.0, 4.0], {}, {})
    check("asm-reshape3", gr[0].shape == (2, 3, 4) and bool(
        (gr[0].reshape(2, 12) == tr[0]).all()), "exact unflatten")
    gb = REGISTRY["RESHAPE2"][0]([gr, 6.0, 4.0], {}, {})
    check("asm-reshape2", gb[0].shape == (6, 4) and bool(
        (gb[0].reshape(2, 12) == tr[0]).all()), "roundtrip exact")
    gp = REGISTRY["PERMUTE3"][0]([gr, 1.0, 0.0, 2.0], {}, {})
    check("asm-permute3", gp[0].shape == (3, 2, 4) and bool(
        (gp[0] == _np.transpose(gr[0], (1, 0, 2))).all()), "exact reorder")
    for badop, badargs, tag in [
        ("RESHAPE3", [tr, 2.0, 2.0, 2.0], "reshape-count"),
        ("RESHAPE2", [tr, 2.0, 1.5], "reshape-frac"),
        ("PERMUTE3", [gr, 1.0, 1.0, 2.0], "permute-dup"),
    ]:
        try:
            REGISTRY[badop][0](badargs, {}, {})
            check(f"asm-{tag}", False, "accepted bad shape")
        except (ValueError, AssertionError) as e:
            check(f"asm-{tag}", True, f"fails loud ({str(e)[:50]})")
    # BRANCH: general verdict-gated select (MIXDYAD's hardcoded pattern,
    # generalized). Exact; shape mismatch fails loud.
    ma2 = S.encode(np.full((4, 4), 0.7))
    mb2 = S.encode(np.full((4, 4), 0.2))
    mm = np.zeros((4, 4), bool)
    mm[:2] = True
    gs = REGISTRY["SELECT"][0]([mm, ma2, mb2], {}, {})
    va = S.decode(ma2[0], ma2[1])
    vb = S.decode(mb2[0], mb2[1])
    vg = S.decode(gs[0], gs[1]) * (1 - gs[2].astype(np.float64))
    check("asm-select", bool((vg[mm] == va[mm]).all() and (vg[~mm] == vb[~mm]).all()),
          "picks A/B by mask, exact")
    try:
        REGISTRY["SELECT"][0]([mm, ma2, S.encode(np.full((4, 5), 0.2))], {}, {})
        check("asm-select-shape", False, "accepted mismatched branches")
    except ValueError as e:
        check("asm-select-shape", True, f"fails loud ({str(e)[:50]})")

    # Batch 3 open: TYPED STREAMS v1. Concrete-vs-concrete mismatches fail
    # naming op+line+stream; UNKNOWN (unannotated) unifies silently
    # (gradual typing -- backward compat proven by every suite above passing
    # with partial annotations). The motivating incidents: float-vs-triples
    # confusion (SUB on floats would silently numpy-subtract!) and the
    # (16,16)-vs-(16,3) broadcast class.
    for bad, tag, want in [
        ("IN x AS F:HW\nIN y AS T:HW\nOUT = SUB(x, y)\n",
         "layout-conflict", "SUB"),
        ("IN x AS F:HW\nOUT = SQRT(x)\nOUT2 = GAIN(x, x, x)\n",
         "layout-gain", "GAIN"),
    ]:
        try:
            ASM.run_text(bad, REGISTRY, {"x": np.zeros((4, 4)),
                                         "y": (np.zeros((4, 4), np.int8),
                                               np.zeros((4, 4), np.int32),
                                               np.zeros((4, 4), np.uint8))},
                         sigs=SIGS)
            check(f"asm-{tag}", False, "ran without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", "line" in str(e) and want in str(e),
                  f"fails loud ({str(e)[:80]})")
    # gradual: fully-unannotated listings behave exactly as v0.1 (no checks).
    try:
        ASM.run_text("IN x\nOUT = ADD(x, x)\n", REGISTRY,
                     {"x": np.zeros(3)}, sigs=SIGS)
        check("asm-gradual", True, "unknown layouts never fail")
    except ASM.AsmError as e:
        check("asm-gradual", False, str(e)[:60])
    # LOOP: repeat() threads STATE; equals manual unroll bit-exactly.
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "programs", "stabilize_mgd.asm")) as fh:
        stext = fh.read()
    rng3 = np.random.default_rng(2)
    fr4 = [(np.clip(rng3.uniform(0, 1, (12, 12, 3)), 0, 1) * 255).astype(np.uint8)
           for _ in range(3)]
    fl4 = [np.zeros((12, 12, 2)) for _ in range(3)]
    hists, _, growth = ASM.repeat(stext, REGISTRY,
                                  {"rgb": fr4, "dprev": None, "flow": fl4},
                                  3, sigs=SIGS)
    man, dp = [], None
    for rgb, fl in zip(fr4, fl4):
        f = ASM.run_text(stext, REGISTRY,
                         {"rgb": rgb, "dprev": dp, "flow": fl}, sigs=SIGS)
        man.append(f)
        dp = f["dprev"]
    same = all(bool((hists[i]["OUT"] == man[i]["OUT"]).all()) for i in range(3))
    check("asm-repeat-unroll", same, "repeat == manual threading, exact")
    check("asm-repeat-growthlog", growth.get("dprev", []) == [],
          "fixed STATE logs no growth (append-only log stays empty)")
    # STATE discipline violations fail loud.
    for bad, tag in [
        ("IN x\nSTATE ghost\nOUT = ADD(x, x)\n", "state-undeclared"),
        ("IN x\nIN g\nSTATE g\nOUT = ADD(x, x)\n", "state-unassigned"),
    ]:
        try:
            ASM.assemble(bad, REGISTRY, SIGS)
            check(f"asm-{tag}", False, "assembled without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", True, f"fails loud ({str(e)[:60]})")
    # shape change across iterations refused (dynamic shapes out of scope).
    try:
        ASM.repeat(stext, REGISTRY,
                   {"rgb": [fr4[0], np.zeros((8, 8, 3), np.uint8), fr4[2]],
                    "dprev": None, "flow": fl4}, 3, sigs=SIGS)
        check("asm-state-shape", False, "accepted geometry change")
    except ASM.AsmError as e:
        check("asm-state-shape", True, f"fails loud ({str(e)[:60]})")
    # ITERATE v1: append-only growth (KV pattern). Cache accumulates rows
    # across iterations; contents exact vs manual concat; growth logged.
    kv_text = ("IN row\nIN cache\nSTATE cache\n"
               "cache = CONCAT(cache, row, 0)\n")
    import phi_core.lattice as S
    rng5 = np.random.default_rng(5)
    rows = [S.encode(rng5.uniform(-1, 1, (1, 6))) for _ in range(4)]
    seed = (np.empty((0, 6), np.int8), np.empty((0, 6), np.int32),
            np.empty((0, 6), np.uint8))
    hists, _, growth = ASM.repeat(kv_text, REGISTRY,
                                  {"row": rows, "cache": seed}, 4,
                                  sigs=SIGS, grow=["cache"])
    final = hists[-1]["cache"]
    ref_s = np.concatenate([r[0] for r in rows], axis=0)
    ref_e = np.concatenate([r[1] for r in rows], axis=0)
    ref_z = np.concatenate([r[2] for r in rows], axis=0)
    check("asm-kv-contents", final[0].shape == (4, 6) and bool(
        (final[0] == ref_s).all() and (final[1] == ref_e).all()
        and (final[2] == ref_z).all()), "cache rows exact vs manual concat")
    check("asm-kv-growth", growth.get("cache") == [(1, 6), (2, 6), (3, 6), (4, 6)],
          f"growth logged {growth.get('cache')}")
    # off-axis growth refused (append-only along axis 0). Here the op
    # itself fails first (non-axis dims differ) -- also loud, also accepted;
    # repeat()'s own off-axis check is defense for looser future ops.
    try:
        ASM.repeat("IN row\nIN cache\nSTATE cache\ncache = CONCAT(cache, row, 1)\n",
                   REGISTRY, {"row": rows, "cache": seed}, 4,
                   sigs=SIGS, grow=["cache"])
        check("asm-kv-offaxis", False, "accepted off-axis growth")
    except (ASM.AsmError, ValueError) as e:
        check("asm-kv-offaxis", True, f"fails loud ({str(e)[:60]})")

    # Batch 3 continued: static verifier v1 (verify() layouts above +
    # ranges.estimate below). Estimator scope, stated: hull intervals catch
    # SATURATION (above cap) and WHOLE-RANGE underflow soundly; they CANNOT
    # see precision loss inside a spanning range (the historical discriminant
    # case: hull [-0.13,0.13] spans the floor while real values sat at 9e-6).
    # That class stays with unit gates + the sum-at-m_acc rule -- the tensor
    # demo (programs/tensor_disc.asm) documents the boundary. Positive
    # controls prove the machinery isn't vacuous.
    from chain import ranges as RG
    f_flag, rep = RG.estimate(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "programs", "holo_flagship.asm")).read(),
        REGISTRY, SIGS)
    check("estimate-flagship-clean", f_flag == [], f"findings={f_flag}")
    sat_text = "IN a\nRANGE a 0.0 2.0\nB = MUL(a, a)\n"
    f_sat, _ = RG.estimate(sat_text, REGISTRY, SIGS)
    check("estimate-saturation", len(f_sat) == 1 and "exceeds" in f_sat[0],
          f"flags [0,4] over cap ({f_sat[0][:60] if f_sat else 'none'})")
    und_text = "IN z\nRANGE z 0.0 1e-7\nW = MUL(z, z)\n"
    f_und, _ = RG.estimate(und_text, REGISTRY, SIGS)
    check("estimate-underflow", len(f_und) == 1 and "underflows" in f_und[0],
          f"flags [0,1e-14] under floor")
    f_ten, _ = RG.estimate(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "programs", "tensor_disc.asm")).read(),
        REGISTRY, SIGS)
    check("estimate-tensor-blind", f_ten == [],
          "documents the hull limit (see tensor_disc.asm header)")
    # GAUSS mnemonic (added for the tensor demo): parity vs float oracle.
    from chain import oracle as _O
    rng4 = np.random.default_rng(4)
    gf = rng4.uniform(0, 1, (12, 12))
    gt = S.encode(gf)
    gg = REGISTRY["GAUSS"][0]([gt, 1.0, 0.8], {}, {})
    gv = S.decode(gg[0], gg[1]) * (1 - gg[2].astype(np.float64))
    ref = _O._corr_replicate(gf, _O.gauss_kernel(radius=1, sigma=0.8))
    check("asm-gauss", 10 * np.log10(1.0 / float(np.mean((gv - ref) ** 2))) >= 40.0,
          "gaussian blur parity vs float oracle")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
