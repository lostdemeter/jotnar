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

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from chain.holo_phi import enhance_image_int

FAIL = []
_CAND_EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rife_reverse", "samples", "f_012.png")
CAND = _CAND_EXT if os.path.isfile(_CAND_EXT) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "input_example.png")


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

    # Batch 1 exposure wiring (GAPS.md): wrappers add NOTHING over phi-core    # fns (0-diff with identical args); softmax normalization gated vs float
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
    # LAYERNORM exposure (v1.6 anatomy): 0-diff both eps paths; in-coverage
    # parity vs torch (66dB; the first fixture ran hot past U_mcov and read
    # 15.8dB -- coverage, not the op); eps_ln-read tripwire (same lesson
    # as eps_rms: no silently-ignored keys).
    _xl = S.encode((rng.uniform(-1, 1, (4, 32))))
    _wl = S.encode(np.ones(32) * 0.9 + (rng.uniform(-1, 1, (32,)) * 0.05))
    _bl = S.encode(rng.uniform(-0.1, 0.1, (32,)))
    _, _mc = H._load_scales()
    check("asm-layernorm-0diff", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["LAYERNORM"][0]([_xl, _wl, _bl], {"eps_ln_c": "4514"}, {}),
        N.int_layernorm_rows(_xl[0], _xl[1], _xl[2], _wl, _bl, _mc, 4514))),
        "0-diff legacy path")
    import torch as _torch2
    _gl = REGISTRY["LAYERNORM"][0]([_xl, _wl, _bl], {"eps_ln": "1e-5"}, {})
    _gv = S.decode(_gl[0], _gl[1]) * (1 - _gl[2].astype(np.float64))
    _xv = S.decode(_xl[0], _xl[1]) * (1 - _xl[2].astype(np.float64))
    _wv = S.decode(_wl[0], _wl[1]) * (1 - _wl[2].astype(np.float64))
    _bv = S.decode(_bl[0], _bl[1]) * (1 - _bl[2].astype(np.float64))
    _gref = _torch2.nn.functional.layer_norm(
        _torch2.tensor(_xv), (32,), weight=_torch2.tensor(_wv),
        bias=_torch2.tensor(_bv), eps=1e-5).numpy()
    _gmse = float(np.mean((_gv - _gref) ** 2))
    check("asm-layernorm-torch", 10 * np.log10(1.0 / _gmse) >= 40.0,
          f"{10 * np.log10(1.0 / _gmse):.1f}dB vs torch layernorm")
    _gl1 = REGISTRY["LAYERNORM"][0]([_xl, _wl, _bl], {"eps_ln": "1.0"}, {})
    check("asm-layernorm-epsread", any(bool((a != b).any()) for a, b in zip(_gl1, _gl)),
          "eps_ln=1.0 moves output (key is read)")
    x = S.encode((rng.random((4, 32)) - 0.5) * 6)
    w = S.encode((rng.random(32) - 0.5) * 2 + 0.5)
    _, mc = H._load_scales()
    check("asm-rmsnorm", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["RMSNORM"][0]([x, w], {"eps_rms_c": "4514"}, {}),
        N.rmsnorm_int(*x, w, mc, 4514))), "0-diff at same (m, eps)")
    # eps_rms (true float) must be READ, not silently ignored: eps=1.0
    # dominates any row energy, so it MUST differ from legacy (on normal
    # fixtures both epsilons are negligible and identical outputs prove
    # nothing -- the huge-eps probe is the tripwire).
    # (A CONFIG key silently ignored by its op cost us a full SmolLM2
    # misdiagnosis round -- this gate exists so it cannot recur.)
    _re = REGISTRY["RMSNORM"][0]([x, w], {"eps_rms": "1.0"}, {})
    _rl = REGISTRY["RMSNORM"][0]([x, w], {"eps_rms_c": "4514"}, {})
    check("asm-epsrms-read",
          any(bool((a != b).any()) for a, b in zip(_re, _rl)),
          "eps_rms=1.0 moves output (key is read)")
    _re6 = REGISTRY["RMSNORM"][0]([x, w], {"eps_rms": "1e-6"}, {})
    from phi_core import lattice as _S2
    _U = float(_S2.PHI ** ((mc - _S2.BIAS) / _S2.K))
    _ec = int(round(1e-6 * float(1 << 36) / (_U * _U)))
    check("asm-epsrms-match", all(bool((a == b).all()) for a, b in zip(
        _re6, N.rmsnorm_int(*x, w, mc, _ec))), "matches direct call")
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
    # F3 (v1.1 backlog): float masks are silent coercion (0.5->True) --
    # refused; bool/int stay (documented nonzero rule, LANGUAGE.md §4).
    try:
        REGISTRY["SELECT"][0]([np.full((4, 4), 0.5), ma2, mb2], {}, {})
        check("asm-select-floatmask", False, "accepted float mask")
    except ValueError as e:
        check("asm-select-floatmask", True, f"fails loud ({str(e)[:50]})")
    _mi = np.zeros((4, 4), np.int64)
    _mi[:2] = 1
    _gi = REGISTRY["SELECT"][0]([_mi, ma2, mb2], {}, {})
    check("asm-select-intmask", all(bool((a == b).all()) for a, b in zip(_gi, gs)),
          "int 0/1 mask == bool mask, exact (documented rule)")

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
    # gradual: fully-unannotated listings behave exactly as v0.1 (no layout
    # checks). NOTE: op-level kind assertions still apply (ADD on floats was
    # never legal -- v0.1 silently computed garbage there, now loud). So the
    # gradual probe uses a float-legit op (SQRT: float -> triples).
    try:
        ASM.run_text("IN x\nOUT = SQRT(x)\n", REGISTRY,
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

    # Debugger: trace mode records per-op summaries + timing without
    # perturbing values; pretty-printer renders them. The trace is how the
    # broadcast scramble / transposed flow / diagonal swap class of bug gets
    # caught in minutes (shapes + layouts + means per line) instead of probe
    # scripts. Honest limit, stated: summaries show symptoms, gates prove causes.
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "programs", "holo_flagship.asm")) as fh:
        ftext = fh.read()
    frgb = np.asarray(Image.open(CAND).convert("RGB"))
    log1, log2 = [], []
    f1 = ASM.run_text(ftext, REGISTRY, frgb, sigs=SIGS, trace=log1)
    f2 = ASM.run_text(ftext, REGISTRY, frgb, sigs=SIGS, trace=log2)
    check("asm-trace-count", len(log1) == 11 and len(log2) == 11,
          f"{len(log1)} records for 11 instructions")
    check("asm-trace-keys", all(set(r) >= {"line", "op", "in", "out", "sec"} for r in log1),
          "every record has line/op/in/out/sec")
    check("asm-trace-pure", bool((f1["OUT"] == ref8).all()),
          "trace mode bit-identical to untraced run (via fidelity ref)")
    s1 = [{k: (v if k != "sec" else 0) for k, v in r.items()} for r in log1]
    s2 = [{k: (v if k != "sec" else 0) for k, v in r.items()} for r in log2]
    check("asm-trace-deterministic", s1 == s2, "summaries stable across runs")
    lines = ASM.format_trace(log1)
    check("asm-trace-print", len(lines) == 12 and "SPLAT_BLUR" in lines[3] and lines[-1].startswith("total"),
          f"{len(lines)} lines incl. total")
    check("asm-trace-profile", all(r["sec"] >= 0 for r in log1) and
          max(log1, key=lambda r: r["sec"])["op"] == "SPLAT_BLUR",
          "timings non-negative; blur dominates (matches fusion pricing)")

    # Pile A exposure batch (GAPS.md): thin wrappers add NOTHING (0-diff vs
    # source fns with identical args); new-small-math (ARGMAX) gated vs
    # numpy semantics; dyadic enforcement + shape contracts fail loud.
    from phi_core import numpy_ops as _N
    ta = S.encode(np.array([[0.5, -1.0, 0.0, 2.0], [3.0, 3.0, 1.0, -2.0]]))
    check("asm-argmax", bool((REGISTRY["ARGMAX"][0]([ta, 1.0], {}, {})
                                   == np.array([3, 0])).all()),
          "exact lattice order incl. tie->first, negatives, zero")
    try:
        REGISTRY["ARGMAX"][0]([ta, 1.5], {}, {})
        check("asm-argmax-axis", False, "accepted fractional axis")
    except ValueError as e:
        check("asm-argmax-axis", True, f"fails loud ({str(e)[:40]})")
    ts = S.encode((rng2.random((4, 6)) - 0.5) * 2)
    gs = REGISTRY["SLICE"][0]([ts, 1.0, 1.0, 4.0], {}, {})
    check("asm-slice", gs[0].shape == (4, 3) and bool(
        (gs[0] == ts[0][:, 1:4]).all()), "exact window")
    try:
        REGISTRY["SLICE"][0]([ts, 1.0, 3.0, 9.0], {}, {})
        check("asm-slice-bounds", False, "accepted OOB window")
    except ValueError as e:
        check("asm-slice-bounds", True, f"fails loud ({str(e)[:40]})")
    tc = S.encode(rng2.uniform(-2, 2, (3, 5)))
    check("asm-clip", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["CLIP"][0]([tc, -1.0, 1.0], {}, {}),
        H.clip_fixed(tc, -1.0, 1.0, mc))), "0-diff vs holo fn")
    td = S.encode(rng2.uniform(0.1, 1.0, (10,)))
    te = S.encode(rng2.uniform(0.1, 2.0, (10,)))
    check("asm-div", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["DIV"][0]([td, te], {}, {}), H.tdiv_pure(td, te))),
        "0-diff (zero-or, no guards)")
    check("asm-sigmoid", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["SIGMOID"][0]([t], {}, {}), H.sigmoid_trip(t))),
        "0-diff (shared c-vectors-sig table covers values)")
    tr = S.encode(rng2.uniform(0.1, 1.0, (20,)))
    gr = REGISTRY["RESCALE"][0]([tr, float(mc)], {}, {})
    v0 = S.decode(tr[0], tr[1])
    v1 = S.decode(gr[0], gr[1]) * (1 - gr[2].astype(np.float64))
    check("asm-rescale-id", float(np.abs(v1 - v0).max() / v0.max()) < 3e-3,
          "same-scale value-preserving (lattice quantum)")
    try:
        REGISTRY["RESCALE"][0]([tr, 70000.0], {}, {})
        check("asm-rescale-range", False, "accepted wild scale")
    except ValueError as e:
        check("asm-rescale-range", True, f"fails loud ({str(e)[:40]})")
    tw = S.encode((rng2.random((9, 7)) - 0.5) * 2)
    ids = np.array([3, 0, 8])
    check("asm-gather", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["GATHER"][0]([tw, ids], {}, {}), _N.gather_int(tw, ids))),
        "0-diff (reindex family)")
    tp2 = S.encode(rng2.uniform(-2, 2, (6,)))
    sl = S.encode(np.full((6,), 0.25))
    check("asm-prelu", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["PRELU"][0]([tp2, sl], {}, {}), _N.prelu_int(tp2, sl))),
        "0-diff")
    tp3 = S.encode((rng2.random((4, 5, 3)) - 0.5) * 2)
    gp = REGISTRY["POOLAVG"][0]([tp3], {}, {})
    ref = tp3
    rv = S.decode(ref[0], ref[1]) * (1 - ref[2].astype(np.float64))
    gv = S.decode(gp[0], gp[1]) * (1 - gp[2].astype(np.float64))
    check("asm-poolavg", gp[0].shape == (3,) and
          float(np.abs(gv - rv.mean(axis=(0, 1))).max()) < 5e-3,
          f"shape (C,) documented; trunc-mean within 5e-3")
    try:
        REGISTRY["POOLAVG"][0]([S.encode(rng2.random((4, 32)))], {}, {})
        check("asm-poolavg-rank", False, "accepted non-HWC")
    except ValueError as e:
        check("asm-poolavg-rank", True, f"fails loud ({str(e)[:40]})")
    td4 = S.encode((rng2.random((5, 5, 2)) - 0.5) * 2)
    wd = S.encode((rng2.random((3, 2, 2, 2)) - 0.5) * 2)
    Wd = {"s": wd[0], "e": wd[1], "z": wd[2]}
    check("asm-deconv", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["DECONV"][0]([td4, wd, 2.0, 0.0], {}, {}),
        _N.deconv_int(td4[0], td4[1], td4[2], Wd, mc, stride=2, pad=0))),
        "0-diff, no bias v1 (stated)")
    tq = S.encode((rng2.random((6, 7, 2)) - 0.5) * 2)
    iq = REGISTRY["INTERP"][0]([tq, 1.0, 1.0], {}, {})
    iv0 = S.decode(tq[0], tq[1]) * (1 - tq[2].astype(np.float64))
    iv1 = S.decode(iq[0], iq[1]) * (1 - iq[2].astype(np.float64))
    check("asm-interp-id", iq[0].shape == (6, 7, 2) and
          float(np.abs(iv1 - iv0).max()) < 5e-3, "1.0 identity (dyadic)")
    try:
        REGISTRY["INTERP"][0]([tq, 1.5, 1.0], {}, {})
        check("asm-interp-nondyadic", False, "accepted non-dyadic scale")
    except ValueError as e:
        check("asm-interp-nondyadic", True, f"LOWERING ERROR, loud ({str(e)[:40]})")
    tk = (rng2.random((3, 3)) - 0.5) / 4
    tk /= abs(tk).sum()
    tc5 = S.encode(rng2.uniform(0, 1, (10, 10)))
    check("asm-conv", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["CONV"][0]([tc5, tk], {}, {}),
        H.conv_trip(tc5, tk, *H._load_scales()[:1], m_out=mc))),
        "0-diff vs conv_trip (proves wiring)")

    # Procedures: DEF/CALL/IMPORT as textual macros with per-call-site
    # namespacing (no runtime call overhead; recursion refused). Expansion
    # exactness: CALL result == manually inlined listing, bit-exact.
    # NOTE: arithmetic is triples-only (float arrays to ADD fail loud --
    # silently computing garbage was a real bug class, closed by op-level
    # assertions; gated below as asm-triples-only).
    def _run(t, payload):
        return ASM.run_text(t, REGISTRY, payload, sigs=SIGS)
    Sanchez = (  # noqa: odd name guards against copy-paste reuse below
        "IN a\nIN b\n"
        "DEF twice(x) -> (y)\n"
        "  T = ADD(x, x)\n"
        "  y = ADD(T, x)\n"
        "END\n")
    ta = S.encode(np.full(3, 0.25))
    tb = S.encode(np.full(3, 0.0))
    try:
        r1 = _run(Sanchez + "OUT = CALL twice(a)\n", {"a": ta, "b": tb})
        r2 = _run("IN a\nIN b\nT = ADD(a, a)\nOUT = ADD(T, a)\n",
                  {"a": ta, "b": tb})
        check("asm-call-exact", all(bool((r1["OUT"][k] == r2["OUT"][k]).all()) for k in (0, 1, 2)),
              "CALL == manually inlined, exact")
    except ASM.AsmError as e:
        check("asm-call-exact", False, str(e)[:80])
    # isolation: two CALLs of one DEF with different args don't collide.
    # Fixture values stay inside m_cov coverage (cap ~1.62: thrice-summed
    # 1.0 saturates BY DESIGN -- documented cap, not a bug; the suite caught
    # my first fixture exceeding it).
    def _dec(t):
        return S.decode(t[0], t[1]) * (1 - t[2].astype(np.float64))
    try:
        t1 = S.encode(np.full(3, 0.25))
        t10 = S.encode(np.full(3, 0.1))
        r3 = _run(Sanchez + "O1 = CALL twice(a)\nO2 = CALL twice(b)\n",
                  {"a": t1, "b": t10})
        v1, v2 = _dec(r3["O1"]), _dec(r3["O2"])
        check("asm-call-isolation",
              bool((np.abs(v1 - 0.75) < 0.02).all() and (np.abs(v2 - 0.3) < 0.02).all()),
              f"per-call-site namespacing ({v1[0]:.2f}, {v2[0]:.2f})")
    except ASM.AsmError as e:
        check("asm-call-isolation", False, str(e)[:80])
    try:
        _run("IN a\nIN b\nOUT = ADD(a, b)\n",
             {"a": np.full(3, 1.0), "b": np.full(3, 2.0)})
        check("asm-triples-only", False, "float arithmetic accepted")
    except (ValueError, ASM.AsmError) as e:
        check("asm-triples-only", True, f"fails loud ({str(e)[:60]})")
    for bad, tag in [
        (Sanchez + "OUT = CALL twice(a, b)\n", "call-arity"),
        (Sanchez + "OUT = CALL nosuch(a)\n", "call-unknown"),
        (Sanchez + "DEF twice(x) -> (y)\n  y = ADD(x, x)\nEND\nOUT = CALL twice(a)\n",
         "call-dupdef"),
        ("IN a\nDEF rec(x) -> (y)\n  y = CALL rec(x)\nEND\nOUT = CALL rec(a)\n",
         "call-recursive"),
        ("IN a\nDEF d(x) -> (y)\n  DEF e(z) -> (w)\n  w = ADD(z, z)\nEND\n  y = ADD(x, x)\nEND\nOUT = CALL d(a)\n",
         "call-nested-def"),
        ("IN a\nDEF d(x) -> (y)\n  CONFIG k v\n  y = ADD(x, x)\nEND\nOUT = CALL d(a)\n",
         "call-config-in-def"),
        ("IN a\nOUT = CALL twice(a)\n", "call-bare-def-missing"),
    ]:
        try:
            _run(bad, {"a": np.full(3, 1.0), "b": np.full(3, 1.0)})
            check(f"asm-{tag}", False, "assembled without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", True, f"fails loud ({str(e)[:70]})")
    # IMPORT: spliced file + cycle refusal (tmp files, cleaned).
    import tempfile as _tf
    import shutil as _sh
    d = _tf.mkdtemp()
    open(os.path.join(d, "lib.asm"), "w").write(
        "DEF inc(x) -> (y)\n  y = ADD(x, x)\nEND\n")
    open(os.path.join(d, "main.asm"), "w").write(
        'IMPORT "lib.asm"\nIN a\nOUT = CALL inc(a)\n')
    open(os.path.join(d, "cyc1.asm"), "w").write('IMPORT "cyc2.asm"\nIN a\nOUT = ADD(a, a)\n')
    open(os.path.join(d, "cyc2.asm"), "w").write('IMPORT "cyc1.asm"\n')
    try:
        r4 = ASM.run_text(open(os.path.join(d, "main.asm")).read(),
                          REGISTRY, {"a": S.encode(np.full(2, 0.5))}, sigs=SIGS,
                          basedir=d)
        check("asm-import", bool((r4["OUT"][0] == r4["OUT"][0]).all()) and
              abs(float(_dec(r4["OUT"])[0]) - 1.0) < 0.05,
              "spliced file executes (basedir-relative)")
        try:
            ASM.run_text(open(os.path.join(d, "cyc1.asm")).read(),
                         REGISTRY, {"a": S.encode(np.full(2, 0.5))}, sigs=SIGS,
                         basedir=d)
            check("asm-import-cycle", False, "accepted import cycle")
        except ASM.AsmError as e:
            check("asm-import-cycle", "cycle" in str(e),
                  f"fails loud ({str(e)[:60]})")
    finally:
        _sh.rmtree(d, ignore_errors=True)

    # shape rules (structural): silent numpy broadcasting across streams hid
    # real bugs (the (16,16)-vs-(16,3) class). Every rule below fails naming
    # op+line; the transpose hint on K-mismatch is deliberate (most common
    # cause, measured). Positives above (all suites green with checks active)
    # prove no legitimate broadcasting broke.
    _a44 = S.encode(np.zeros((4, 4)))
    _a43 = S.encode(np.zeros((4, 3)))
    _a36 = S.encode(np.zeros((3, 6)))
    for prog, tag, want in [
        ("IN a\nIN b\nOUT = ADD(a, b)\n", "shape-add", "mismatch"),
        ("IN a\nIN b\nOUT = SUB(a, b)\n", "shape-sub", "mismatch"),
        ("IN a\nIN b\nOUT = MUL(a, b)\n", "shape-mul", "mismatch"),
        ("IN a\nIN b\nOUT = DIV(a, b)\n", "shape-div", "mismatch"),
        ("IN m\nIN a\nIN b\nOUT = SELECT(m, a, b)\n", "shape-select-branch", "mismatch"),
        ("IN a\nIN b\nOUT = MATMUL(a, b)\n", "shape-matmul", "TRANSPOSE"),
        ("IN a\nIN b\nOUT = BATCH_MATMUL(a, b)\n", "shape-bmatmul", "TRANSPOSE"),
    ]:
        feeds = {"a": _a44, "b": _a36 if "matmul" in tag else _a43,
                 "m": np.zeros((4, 4), bool)}
        try:
            ASM.run_text(prog, REGISTRY, feeds, sigs=SIGS)
            check(f"asm-{tag}", False, "ran without error")
        except ASM.AsmError as e:
            check(f"asm-{tag}", want in str(e), f"fails loud ({str(e)[:70]})")
    try:
        ASM.run_text("IN a\nIN f\nOUT = WARP(a, f)\n", REGISTRY,
                     {"a": _a44, "f": np.zeros((5, 5, 2))}, sigs=SIGS)
        check("asm-shape-warp", False, "ran without error")
    except ASM.AsmError as e:
        check("asm-shape-warp", "spatial mismatch" in str(e),
              f"fails loud ({str(e)[:60]})")
    try:
        ASM.run_text("IN a\nIN k\nOUT = ARGMAX(a, k)\n", REGISTRY,
                     {"a": _a44, "k": 5.0}, sigs=SIGS)
        check("asm-shape-argmax-axis", False, "accepted bad axis")
    except (ASM.AsmError, ValueError) as e:
        check("asm-shape-argmax-axis", True, f"fails loud ({str(e)[:50]})")
    try:
        ASM.run_text("IN w\nIN i\nOUT = GATHER(w, i)\n", REGISTRY,
                     {"w": _a44, "i": np.array([0, 99])}, sigs=SIGS)
        check("asm-shape-gather", False, "accepted OOB ids")
    except ASM.AsmError as e:
        check("asm-shape-gather", "out of range" in str(e),
              f"fails loud ({str(e)[:60]})")

    # Composition discipline: DEF formal AS contracts checked at CALL sites
    # (both known + concrete + unequal fails naming CALL site AND DEF origin;
    # UNKNOWN either side defers to post-expansion verify + runtime).
    _t3 = S.encode(np.full(3, 0.25))
    _lib = ("DEF shl(x AS T:HW) -> (y AS T:HW)\n"
            "  y = ADD(x, x)\nEND\n")
    try:
        r5 = ASM.run_text(_lib + "IN a AS T:HW\nOUT = CALL shl(a)\n",
                          REGISTRY, {"a": _t3}, sigs=SIGS)
        check("asm-contract-match", True, "declared layouts agree, runs")
    except ASM.AsmError as e:
        check("asm-contract-match", False, str(e)[:70])
    try:
        ASM.run_text(_lib + "IN a AS F:HW\nOUT = CALL shl(a)\n",
                     REGISTRY, {"a": np.zeros((4, 4))}, sigs=SIGS)
        check("asm-contract-violation", False, "mismatched CALL accepted")
    except ASM.AsmError as e:
        check("asm-contract-violation",
              "wants T:HW" in str(e) and "DEF at" in str(e),
              f"names site+origin ({str(e)[:80]})")
    try:
        r6 = ASM.run_text(_lib + "IN a\nOUT = CALL shl(a)\n",
                          REGISTRY, {"a": _t3}, sigs=SIGS)
        check("asm-contract-defer", True, "UNKNOWN actual defers silently")
    except ASM.AsmError as e:
        check("asm-contract-defer", False, str(e)[:70])
    try:
        ASM.run_text("IN a AS T:HW\nDEF v(x AS $A) -> (y)\n  y = ADD(x, x)\nEND\n"
                     "OUT = CALL v(a)\n",
                     REGISTRY, {"a": _t3}, sigs=SIGS)
        check("asm-contract-varformal", False, "accepted $VAR formal")
    except ASM.AsmError as e:
        check("asm-contract-varformal", "must not use $VAR" in str(e),
              f"fails loud ({str(e)[:60]})")
    # IMPORT contracts: interface table + cross-file enforcement.
    import tempfile as _tf2
    import shutil as _sh2
    d2 = _tf2.mkdtemp()
    try:
        open(os.path.join(d2, "geom.asm"), "w").write(_lib)
        errs, rep = ASM.verify('IMPORT "geom.asm"\nIN a AS T:HW\nOUT = CALL shl(a)\n',
                               REGISTRY, SIGS, basedir=d2)
        check("asm-iface-table", "geom.asm" in rep["defs"].get("shl", {}).get("origin", ""),
              f"interface visible: {sorted(rep['defs'])}")
        check("asm-iface-verify", errs == [], f"compliant import verifies clean: {errs}")
        ASM.run_text('IMPORT "geom.asm"\nIN a AS F:HW\nOUT = CALL shl(a)\n',
                     REGISTRY, {"a": np.zeros((4, 4))}, sigs=SIGS, basedir=d2)
        check("asm-iface-violation", False, "cross-file mismatch accepted")
    except ASM.AsmError as e:
        check("asm-iface-violation", "wants T:HW" in str(e),
              f"enforced across files ({str(e)[:80]})")
    finally:
        _sh2.rmtree(d2, ignore_errors=True)

    # v1.0 Gate 5 drill: GELU (first stranger-supplied structure, via
    # phi-core ASM_HANDOFF.md). Thin wrapper: 0-diff vs phi-core fn;
    # parity vs independent torch gelu (peak=1.0 basis, bar 40dB);
    # in-listing use proves the mnemonic works in-language (not just as
    # a direct registry call). Fixture spans the activation shape:
    # saturation (large neg -> 0), dip (~-1), zero, linear (large pos).
    import torch as _torch
    tg = S.encode(rng2.uniform(-4, 4, (4, 32)))
    check("asm-gelu-0diff", all(bool((a == b).all()) for a, b in zip(
        REGISTRY["GELU"][0]([tg], {}, {}), _N.gelu_erf_int(tg))),
        "0-diff vs phi-core (wrapper adds nothing)")
    gg = REGISTRY["GELU"][0]([tg], {}, {})
    gv = S.decode(gg[0], gg[1]) * (1 - gg[2].astype(np.float64))
    gref = _torch.nn.functional.gelu(
        _torch.tensor(S.decode(tg[0], tg[1])
                      * (1 - tg[2].astype(np.float64)))).numpy()
    gmse = float(np.mean((gv - gref) ** 2))
    check("asm-gelu-torch", 10 * np.log10(1.0 / gmse) >= 40.0,
          f"{10 * np.log10(1.0 / gmse):.1f}dB vs torch gelu (peak=1.0)")
    ge = S.encode(np.array([-10.0, -1.0, 0.0, 10.0]))
    ge2 = REGISTRY["GELU"][0]([ge], {}, {})
    gev = S.decode(ge2[0], ge2[1]) * (1 - ge2[2].astype(np.float64))
    check("asm-gelu-edges", abs(gev[0]) < 1e-6 and abs(gev[2]) < 1e-6
          and abs(gev[3] - 10.0) < 0.05 and gev[1] < 0,
          f"saturate/dip/zero/linear {np.round(gev, 4)}")
    try:
        gl = ASM.run_text("IN x AS T:SEQ\nOUT = GELU(x)\n", REGISTRY,
                          {"x": tg}, sigs=SIGS)
        check("asm-gelu-listing", all(
            bool((gl["OUT"][k] == gg[k]).all()) for k in (0, 1, 2)),
            "in-listing GELU == direct call, exact")
    except ASM.AsmError as e:
        check("asm-gelu-listing", False, str(e)[:70])

    # v1.0 Gates 3+4: dogfooded stdlib + search order. Bare IMPORT names
    # resolve stdlib-first (shared listings addressable by name from any
    # basedir); explicit paths (`./x`, `sub/x`) stay relative-only.
    import tempfile as _tf3
    import shutil as _sh3
    d3 = _tf3.mkdtemp()
    try:
        # shadow: tmpdir/mlp.asm defines something ELSE -- stdlib must win.
        open(os.path.join(d3, "mlp.asm"), "w").write(
            "DEF otherdef(x) -> (y)\n  y = ADD(x, x)\nEND\n")
        errs, rep = ASM.verify('IMPORT "mlp.asm"\nIN a\nOUT = CALL swiglu_block(a, a, a, a)\n',
                               REGISTRY, SIGS, basedir=d3)
        check("asm-import-stdlib-first",
              "swiglu_block" in rep["defs"]
              and "stdlib" in rep["defs"]["swiglu_block"]["origin"],
              f"stdlib wins over basedir shadow ({rep['defs'].get('swiglu_block', {}).get('origin', 'MISSING')})")
        # explicit relative path bypasses stdlib (no shadowing surprises).
        open(os.path.join(d3, "local.asm"), "w").write(
            "DEF localinc(x) -> (y)\n  y = ADD(x, x)\nEND\n")
        r7 = ASM.run_text('IMPORT "./local.asm"\nIN a\nOUT = CALL localinc(a)\n',
                          REGISTRY, {"a": S.encode(np.full(2, 0.25))},
                          sigs=SIGS, basedir=d3)
        check("asm-import-relative", True, "explicit ./ path stays relative")
    except ASM.AsmError as e:
        check("asm-import-stdlib-first", False, str(e)[:80])
    finally:
        _sh3.rmtree(d3, ignore_errors=True)
    # dogfood: programs/xf_block.asm (IMPORTs + CALLs onto stdlib) is
    # bit-exact vs the original spelled-out listing (embedded here, the
    # pre-Gate-3 body). Refactoring shared prologues must not move values.
    _inline_xf = """CONFIG heads 8
IN x AS T:SEQ
IN pos AS I:SEQ
IN wq AS T:SEQ
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2
XN = RMSNORM(x, rms_w1)
Q = MATMUL(XN, wq)
K = MATMUL(XN, wk)
V = MATMUL(XN, wv)
QR = ROTARY(Q, pos)
KR = ROTARY(K, pos)
KT = TRANSPOSE(KR)
SCORES = BATCH_MATMUL(QR, KT)
P = SOFTMAX(SCORES)
CTX = BATCH_MATMUL(P, V)
O = MATMUL(CTX, wo)
H = ADD(x, O)
HN = RMSNORM(H, rms_w2)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = MATMUL(MID, wdown)
OUT = ADD(H, DOWN)
"""
    _xdog = np.random.default_rng(7)
    _Sq, _D, _Df = 4, 8, 16
    _xd = S.encode((_xdog.random((_Sq, _D)) - 0.5) * 0.3)
    _posd = np.arange(_Sq, dtype=np.int64)

    def _wd(sh):
        return S.encode((_xdog.random(sh) - 0.5) * 0.3)
    _payd = {"x": _xd, "pos": _posd, "wq": _wd((_D, _D)),
             "wk": _wd((_D, _D)), "wv": _wd((_D, _D)),
             "wo": _wd((_D, _D)), "wup": _wd((_D, _Df)),
             "wgate": _wd((_D, _Df)), "wdown": _wd((_Df, _D)),
             "rms_w1": _wd((_D,)), "rms_w2": _wd((_D,))}
    try:
        _fprod = ASM.run_text(
            open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "programs", "xf_block.asm")).read(),
            REGISTRY, _payd, sigs=SIGS,
            basedir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "programs"))
        _finline = ASM.run_text(_inline_xf, REGISTRY, _payd, sigs=SIGS)
        check("asm-dogfood-xf", all(
            bool((_fprod["OUT"][k] == _finline["OUT"][k]).all())
            for k in (0, 1, 2)), "IMPORT-refactored == spelled-out, exact")
    except ASM.AsmError as e:
        check("asm-dogfood-xf", False, str(e)[:80])

    # v1.0 Gate 2: LANGUAGE.md covers every mnemonic (drift gate -- the
    # reference must not silently fall behind the registry).
    _lang = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "docs", "LANGUAGE.md")).read()
    _missing = sorted(m for m in REGISTRY if m not in _lang)
    check("asm-lang-coverage", _missing == [],
          f"{len(REGISTRY) - len(_missing)}/{len(REGISTRY)} mnemonics documented"
          + ("" if not _missing else f" (missing {_missing})"))

    # v1.1 per-block spike (prototype): MATMUL honors CONFIG `m_acc`.
    # Large-magnitude fixture rails at the frozen scale (-10dB) and holds
    # 55dB one scale up -- the 66dB swing is the per-block win, measured.
    # Default path stays 0-diff (every suite below passing unchanged is
    # the proof); bad overrides fail loud.
    _sc = np.random.default_rng(3)
    _sA = S.encode((_sc.random((4, 8)) - 0.5) * 4)
    _sB = S.encode((_sc.random((8, 6)) - 0.5) * 4)
    _sAv = S.decode(_sA[0], _sA[1]) * (1 - _sA[2].astype(np.float64))
    _sBv = S.decode(_sB[0], _sB[1]) * (1 - _sB[2].astype(np.float64))
    _sref = _sAv @ _sBv
    _ma, _ = H._load_scales()
    _same = REGISTRY["MATMUL"][0]([_sA, _sB], {"m_acc": str(_ma)}, {})
    _dflt = REGISTRY["MATMUL"][0]([_sA, _sB], {}, {})
    check("asm-scale-default", all(bool((a == b).all()) for a, b in zip(_same, _dflt)),
          "explicit frozen value == default, exact (default untouched)")
    # same-line family: BATCH_MATMUL (m_acc) + ADD (m_cov) defaults 0-diff.
    _bB = S.encode((_sc.random((2, 4, 8)) - 0.5) * 2)
    _bA = S.encode((_sc.random((2, 4, 8)) - 0.5) * 2)
    _bb0 = REGISTRY["BATCH_MATMUL"][0]([_sA, _sB], {"m_acc": str(_ma)}, {})
    _bb1 = REGISTRY["BATCH_MATMUL"][0]([_sA, _sB], {}, {})
    check("asm-scale-default-bmm", all(bool((a == b).all()) for a, b in zip(_bb0, _bb1)),
          "BATCH_MATMUL default untouched")
    _, _mc = H._load_scales()
    _ad0 = REGISTRY["ADD"][0]([_bA, _bB], {"m_cov": str(_mc)}, {})
    _ad1 = REGISTRY["ADD"][0]([_bA, _bB], {}, {})
    check("asm-scale-default-add", all(bool((a == b).all()) for a, b in zip(_ad0, _ad1)),
          "ADD default untouched")
    _big = REGISTRY["MATMUL"][0]([_sA, _sB], {"m_acc": "35492"}, {})
    _bigv = S.decode(_big[0], _big[1]) * (1 - _big[2].astype(np.float64))
    _dfltv = S.decode(_dflt[0], _dflt[1]) * (1 - _dflt[2].astype(np.float64))
    _mse_big = float(np.mean((_bigv - _sref) ** 2))
    _mse_dflt = float(np.mean((_dfltv - _sref) ** 2))
    _db_big = 10 * np.log10(1.0 / _mse_big)
    _db_dflt = 10 * np.log10(1.0 / _mse_dflt) if _mse_dflt > 0 else 999
    check("asm-scale-override", _db_big >= 40.0 and _db_dflt < 40.0,
          f"override {_db_big:.1f}dB vs default {_db_dflt:.1f}dB (row non-vacuous)")
    try:
        _inlist = ASM.run_text("CONFIG m_acc 35492\nIN a\nIN b\nOUT = MATMUL(a, b)\n",
                               REGISTRY, {"a": _sA, "b": _sB}, sigs=SIGS)
        check("asm-scale-listing", all(
            bool((_inlist["OUT"][k] == _big[k]).all()) for k in (0, 1, 2)),
            "in-listing CONFIG override == direct call, exact")
    except ASM.AsmError as e:
        check("asm-scale-listing", False, str(e)[:70])
    for _bad, _tag in [("1.5", "frac"), ("70000", "range"), ("abc", "nonnum")]:
        try:
            REGISTRY["MATMUL"][0]([_sA, _sB], {"m_acc": _bad}, {})
            check(f"asm-scale-bad-{_tag}", False, "accepted bad scale")
        except ValueError as e:
            check(f"asm-scale-bad-{_tag}", True, f"fails loud ({str(e)[:40]})")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
