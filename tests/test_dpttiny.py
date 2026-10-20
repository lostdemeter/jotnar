"""DPT-tiny build gate: listing executes with parity + determinism.

Builder-constructed listing (programs/dpt_tiny.py, fiber doctrine —
no reshape anywhere; 3x3 convs in GATHER-tap form, nearest-GATHER
upsampling, LayerNorm, unfused H4 attention, GELU-form MLPs) vs a
torch twin with IDENTICAL values (kaiming weights, seed 0; twin
mirrors listing dataflow exactly, incl. RCU-fuse-RCU order and
Wq pre-scaling). Proof bar (pre-run): lattice-vs-torch >=40dB
(drill convention 10log10(1/mse)), CUDA-served-vs-torch >=40dB,
determinism bit-exact reruns, shapes exact every stream incl. all
four reassemble scales, fusion interior parity, P1-equivalence
probe (GATHER-form conv == emitted-C P1 binary, near-exact).
CUDA legs SKIP without nvcc/GPU. No training anywhere.
Usage: python3 tests/test_dpttiny.py (lattice ~minutes, CPU).
"""
import glob
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "programs"))

FAIL = []

SEED = 0
GR, GC, T = 42, 64, 2688
WD, NH, NB = 128, 4, 6
SCALES = {"s32": (21, 32), "s16": (42, 64), "s8": (84, 128), "s4": (168, 256)}


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def db(ref, got):
    import math
    import numpy as np
    mse = float(np.mean((np.asarray(ref, dtype=np.float64).ravel()
                         - np.asarray(got, dtype=np.float64).ravel()) ** 2))
    return 10 * math.log10(1.0 / max(mse, 1e-300))


def gen_weights():
    import numpy as np
    import torch
    torch.manual_seed(SEED)
    Wd = {}
    # Proof setup (documented): global 0.04x scale keeps EVERY stage
    # within fixed-point ±1.0 coverage with margin (measured: E-matmul
    # 16.0dB@1.0 -> 28.0dB@0.25 -> 41.9dB@0.05; fusion is norm-free
    # (pure accumulation quantum improves monotonically with smaller
    # s); s=0.04 for fusion headroom while trunk keeps margin.
    # The twin is arbitrary proof constants; the listing is
    # scale-agnostic. Drill convention needs small magnitudes
    # for 40dB-grade parity (768-wide accumulation quantum).
    # (kaiming fan note: torch sizes (in,out) tensors with fan=size(1);
    #  our (768,128) tap matrices get bound sqrt(6/128) — measured,
    #  accounted, harmless.)
    SC = 0.04

    def kai(*shape):
        t = torch.empty(*shape)
        torch.nn.init.kaiming_uniform_(t, nonlinearity="relu")
        return (t.numpy() * SC).astype(np.float64)

    Wd["pemb"] = kai(768, WD)
    Wd["pos"] = kai(T, WD) * 0.5
    Wd["lnw"] = np.ones(WD)
    Wd["lnb"] = np.zeros(WD)
    for bi in range(NB):
        Wd[f"b{bi}wq"] = kai(WD, WD) / np.sqrt(WD // NH)
        Wd[f"b{bi}wk"] = kai(WD, WD)
        Wd[f"b{bi}wv"] = kai(WD, WD)
        Wd[f"b{bi}wo"] = kai(WD, WD)
        Wd[f"b{bi}w1"] = kai(WD, 4 * WD)
        Wd[f"b{bi}w2"] = kai(4 * WD, WD)
    for nm in ("r32", "r16", "r8", "r4"):
        Wd[nm] = kai(WD, WD)
    for li in range(4):
        for ri in range(2):
            for ci in range(2):
                for ti in range(9):
                    Wd[f"f{li}r{ri}c{ci}t{ti}"] = kai(WD, WD)
    Wd["hp"] = kai(WD, 32)
    Wd["hl"] = kai(32, 1)
    return Wd


def twin_forward(img, Wd, ids):
    import numpy as np
    import torch

    def geluform(x):
        return torch.nn.functional.silu(x * 1.702) / 1.702

    x = torch.tensor(img, dtype=torch.float32)
    Hh, Ww, _ = x.shape
    toks = torch.stack([x.reshape(-1, 3)[torch.tensor(ids["patch"][ti])]
                        for ti in range(256)], -1).reshape(T, 768)
    h = toks @ torch.tensor(Wd["pemb"], dtype=torch.float32)
    h = h + torch.tensor(Wd["pos"], dtype=torch.float32)
    tap_out = {}
    for bi in range(NB):
        hn = torch.nn.functional.layer_norm(h, (WD,), None, None, 1e-5)
        q = hn @ torch.tensor(Wd[f"b{bi}wq"], dtype=torch.float32)
        k = hn @ torch.tensor(Wd[f"b{bi}wk"], dtype=torch.float32)
        v = hn @ torch.tensor(Wd[f"b{bi}wv"], dtype=torch.float32)
        q4 = q.view(T, NH, WD // NH).transpose(0, 1)
        k4 = k.view(T, NH, WD // NH).transpose(0, 1)
        v4 = v.view(T, NH, WD // NH).transpose(0, 1)
        a = torch.softmax(q4 @ k4.transpose(-1, -2), -1)
        o = (a @ v4).transpose(0, 1).reshape(T, WD)
        h = h + o @ torch.tensor(Wd[f"b{bi}wo"], dtype=torch.float32)
        hn = torch.nn.functional.layer_norm(h, (WD,), None, None, 1e-5)
        h = h + (geluform(hn @ torch.tensor(Wd[f"b{bi}w1"], dtype=torch.float32))
                 @ torch.tensor(Wd[f"b{bi}w2"], dtype=torch.float32))
        if bi in (1, 3, 4, 5):
            tap_out[bi] = h
    outs = {}
    r32 = tap_out[1][torch.tensor(ids["dn16_32"])]
    m32 = r32 @ torch.tensor(Wd["r32"], dtype=torch.float32)
    m16 = tap_out[3] @ torch.tensor(Wd["r16"], dtype=torch.float32)
    g8 = tap_out[4][torch.tensor(ids["up16_8"])]
    m8 = g8 @ torch.tensor(Wd["r8"], dtype=torch.float32)
    g4 = tap_out[5][torch.tensor(ids["up16_4"])]
    m4 = g4 @ torch.tensor(Wd["r4"], dtype=torch.float32)
    outs.update({"m32": m32, "m16": m16, "m8": m8, "m4": m4})

    def conv_block(f4d, li, ri):
        import torch.nn.functional as F
        y = f4d
        for ci in range(2):
            # Wt[ti] is (in,out) listing layout -> conv (out,in,ky,kx)
            Wc = torch.empty(WD, WD, 3, 3)
            for ti in range(9):
                Wc[:, :, ti // 3, ti % 3] = torch.tensor(
                    Wd[f"f{li}r{ri}c{ci}t{ti}"], dtype=torch.float32).t()
            yp = F.pad(y, (1, 1, 1, 1), mode="replicate")
            y = F.conv2d(yp, Wc)
            y = torch.nn.functional.silu(y)
        return y

    def to4(f2d, gh, gw):
        return f2d.view(gh, gw, WD).permute(2, 0, 1).unsqueeze(0)

    def from4(f4d):
        return f4d[0].permute(1, 2, 0).reshape(-1, WD)

    f32 = m32
    f32 = from4(conv_block(to4(f32, 21, 32), 0, 0))
    f32 = from4(conv_block(to4(f32, 21, 32), 0, 1))
    outs["F0b"] = f32
    up = f32[torch.tensor(ids["up32_16"])]
    f16 = from4(conv_block(to4(m16, 42, 64), 1, 0))
    f16 = f16 + up
    outs["F1f"] = f16
    f16 = from4(conv_block(to4(f16, 42, 64), 1, 1))
    up = f16[torch.tensor(ids["up16_8"])]
    f8 = from4(conv_block(to4(m8, 84, 128), 2, 0))
    f8 = f8 + up
    outs["F2f"] = f8
    f8 = from4(conv_block(to4(f8, 84, 128), 2, 1))
    up = f8[torch.tensor(ids["up8_4"])]
    f4 = from4(conv_block(to4(m4, 168, 256), 3, 0))
    f4 = f4 + up
    outs["F3f"] = f4
    f4 = from4(conv_block(to4(f4, 168, 256), 3, 1))
    outs["F3b"] = f4
    hp = f4 @ torch.tensor(Wd["hp"], dtype=torch.float32)
    big = hp[torch.tensor(ids["up4_full"])]
    out = (big @ torch.tensor(Wd["hl"], dtype=torch.float32)).reshape(-1)
    outs["OUT"] = out
    return outs


def main():
    import numpy as np
    import phi_core.lattice as S
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    from chain.builder import Prog
    import dpt_tiny as DT
    from PIL import Image

    im = np.asarray(Image.open(
        "samples/marigold/inputs/dogs.png").convert("RGB").resize((1024, 672))
        ).astype(np.float64) / 255.0
    Wd = gen_weights()
    print(f"weight-scale liveness: pemb maxabs "
          f"{float(np.abs(Wd['pemb']).max()):.4f} (expect ~0.011)",
          flush=True)
    twin = twin_forward(im, Wd, {
        "patch": DT._ids_patch(), "dn16_32": DT._ids_down((21, 32), (42, 64)),
        "up16_8": DT._ids_stride((84, 128), (42, 64)),
        "up16_4": DT._ids_stride((168, 256), (42, 64)),
        "up8_4": DT._ids_stride((168, 256), (84, 128)),
        "up32_16": DT._ids_stride((42, 64), (21, 32)),
        "up4_full": DT._ids_stride((672, 1024), (168, 256))})

    prog, ids = DT.build_dpttiny()
    text = prog.text()
    feeds = {"RGB": S.encode(np.ascontiguousarray(im.reshape(-1, 3)))}
    for ti in range(256):
        feeds[f"ptap{ti}"] = np.ascontiguousarray(ids["patch"][ti])
    feeds["pos"] = S.encode(np.ascontiguousarray(Wd["pos"]))
    feeds["lnw"] = S.encode(np.ascontiguousarray(Wd["lnw"]))
    feeds["lnb"] = S.encode(np.ascontiguousarray(Wd["lnb"]))
    feeds["pemb"] = S.encode(np.ascontiguousarray(Wd["pemb"]))
    for bi in range(NB):
        for nm in ("wq", "wk", "wv", "wo", "w1", "w2"):
            feeds[f"b{bi}{nm}"] = S.encode(
                np.ascontiguousarray(Wd[f"b{bi}{nm}"]))
    for nm in ("r32", "r16", "r8", "r4", "hp", "hl"):
        feeds[nm] = S.encode(np.ascontiguousarray(Wd[nm]))
    for li in range(4):
        for ri in range(2):
            for ci in range(2):
                for ti in range(9):
                    feeds[f"f{li}r{ri}c{ci}t{ti}"] = S.encode(
                        np.ascontiguousarray(Wd[f"f{li}r{ri}c{ci}t{ti}"]))
    for nm in ("up16_8", "up16_4", "dn16_32", "up8_4", "up4_full", "up32_16"):
        feeds[nm] = np.ascontiguousarray(ids[nm])
    for nm, (gh, gw) in (("conv32", (21, 32)), ("conv16", (42, 64)),
                         ("conv8", (84, 128)), ("conv4", (168, 256))):
        for ti in range(9):
            feeds[f"{nm}t{ti}"] = np.ascontiguousarray(
                DT._ids_conv_taps(gh, gw)[ti])

    trace = []
    got = ASM.run_text(text, REGISTRY, feeds, sigs=SIGS, trace=trace)
    check("dpt-assembled", True, f"{len(trace)} ops bound")

    def dec(name):
        t = got[name]
        return (S.decode(np.ascontiguousarray(t[0]), np.ascontiguousarray(t[1]))
                * (1 - np.ascontiguousarray(t[2]).astype(np.float64)))

    # -- shape-exact every stream incl. 4 scales ------------------------------
    exp_shapes = {"m32": (672, WD), "m16": (2688, WD), "m8": (10752, WD),
                  "m4": (43008, WD), "F0b": (672, WD), "F1f": (2688, WD),
                  "F2f": (10752, WD), "F3f": (43008, WD), "F3b": (43008, WD),
                  "OUT": (688128, 1)}
    shape_ok = True
    for nm, sh in exp_shapes.items():
        gotsh = np.shape(dec(nm))
        if tuple(gotsh) != sh:
            shape_ok = False
            print(f"  shape {nm}: got {tuple(gotsh)} want {sh}", flush=True)
    check("dpt-shapes", bool(shape_ok), "4 scales + fusion + OUT exact")
    fin_ok = all(bool(np.isfinite(np.asarray(dec(nm))).all())
                 for nm in exp_shapes)
    check("dpt-finite", bool(fin_ok), "all checked streams finite")

    # -- parity vs torch twin ---------------------------------------------------
    ref = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in twin.items()}
    gotd = {k: np.asarray(dec(k), dtype=np.float64).reshape(-1)
            for k in ("m32", "m16", "m8", "m4", "F0b", "F1f", "F2f", "F3f",
                      "F3b", "OUT")}
    for k in ("OUT",):
        check("dpt-parity-out", db(ref[k], gotd[k]) >= 40,
              f"{db(ref[k], gotd[k]):.1f}dB (lattice vs torch)")
    for k in ("F1f", "F2f", "F3f", "F3b"):
        check(f"dpt-fusion-{k}", db(ref[k], gotd[k]) >= 40,
              f"{db(ref[k], gotd[k]):.1f}dB interior parity")
    for k in ("m32", "m16", "m8", "m4", "F0b"):
        check(f"dpt-scale-{k}", db(ref[k], gotd[k]) >= 40,
              f"{db(ref[k], gotd[k]):.1f}dB")

    # -- determinism rerun --------------------------------------------------------
    got2 = ASM.run_text(text, REGISTRY, feeds, sigs=SIGS)
    same = True
    for k in ("OUT", "F3b", "m16"):
        a, b = got[k], got2[k]
        same &= all(bool((np.ascontiguousarray(x)
                          == np.ascontiguousarray(y)).all())
                    for x, y in zip(a, b))
    check("dpt-deterministic", bool(same), "lattice rerun bit-exact")

    # -- P1-equivalence probe (GATHER-form == emitted-C P1) ------------------------
    rng = np.random.default_rng(4)
    xp = (rng.random((6, 8)) - 0.5)
    kp = (rng.random((3, 3)) - 0.5) / 4
    from chain.emit_c import compile_program, build as _bc
    pt = "IN X\nIN K\nO = CONV(X, K)\n"
    samp = {"X": np.ascontiguousarray(xp), "K": np.ascontiguousarray(kp)}
    artp = compile_program(pt, "c", sample=samp, outputs=["O"],
                           basedir=os.path.join(ROOT, "programs"))
    workp = "/tmp/dpt_p1probe"
    os.makedirs(workp, exist_ok=True)
    exep = _bc(artp["source"], workp, name="p1probe")
    xp.astype(np.float64).tofile(os.path.join(workp, "x.bin"))
    kp.astype(np.float64).tofile(os.path.join(workp, "k.bin"))
    rp = subprocess.run([exep, os.path.join(workp, "x.bin"),
                         os.path.join(workp, "k.bin"),
                         os.path.join(workp, "o.bin")],
                        capture_output=True, text=True)
    check("dpt-p1-run", rp.returncode == 0, f"rc={rp.returncode}")
    if rp.returncode == 0:
        p1o = np.fromfile(os.path.join(workp, "o.bin"),
                          dtype=np.float64).reshape(6, 8)
        import dpt_tiny as _DT
        taps = _DT._ids_conv_taps(6, 8)
        gf = sum(xp.ravel()[taps[ti]].reshape(6, 8) * kp.ravel()[ti]
                 for ti in range(9))
        check("dpt-p1-equivalence",
              float(np.abs(p1o - gf).max()) < 1e-9,
              f"maxabs={float(np.abs(p1o - gf).max()):.1e} (GATHER-form==P1)")

    # -- CUDA served legs -------------------------------------------------------------
    import torch as _t
    if shutil.which("nvcc") is None or not _t.cuda.is_available():
        print("SKIP cuda legs (needs nvcc + GPU)")
    else:
        from chain.emit_cuda import build_cu
        sample = {"RGB": np.ascontiguousarray(im.reshape(-1, 3).astype(np.float32))}
        for k, v in Wd.items():
            sample[k] = np.ascontiguousarray(v.astype(np.float32))
        for nm in ("up16_8", "up16_4", "dn16_32", "up8_4", "up4_full", "up32_16"):
            sample[nm] = np.ascontiguousarray(ids[nm].astype(np.int64))
        for nm, (gh, gw) in (("conv32", (21, 32)), ("conv16", (42, 64)),
                             ("conv8", (84, 128)), ("conv4", (168, 256))):
            for ti in range(9):
                sample[f"{nm}t{ti}"] = np.ascontiguousarray(
                    DT._ids_conv_taps(gh, gw)[ti].astype(np.int64))
        for ti in range(256):
            sample[f"ptap{ti}"] = np.ascontiguousarray(
                ids["patch"][ti].astype(np.int64))
        artu = compile_program(text, "cuda", sample=sample, outputs=["OUT"],
                               basedir=os.path.join(ROOT, "programs"))
        worku = "/tmp/dpttiny_cu"
        os.makedirs(worku, exist_ok=True)
        exeu = build_cu(artu["source"], worku, name="dpttiny",
                        units=artu.get("units"))
        argv = [exeu]
        in_order = [ln.split()[1] for ln in text.splitlines()
                    if ln.startswith("IN ")]
        for nm in in_order:
            fp = os.path.join(worku, f"f_{nm}.bin")
            arr = sample[nm]
            (arr.astype(np.float32) if arr.dtype != np.int64
             else arr).tofile(fp)
            argv.append(fp)
        argv.append(os.path.join(worku, "o_OUT.bin"))
        outs = []
        for i in range(2):
            rr = subprocess.run(argv, capture_output=True, text=True)
            if rr.returncode != 0:
                check("dpt-cu-run", False, rr.stderr[:300])
                break
            outs.append(np.fromfile(os.path.join(worku, "o_OUT.bin"),
                                    dtype=np.float32))
        else:
            check("dpt-cu-run", True, "")
            check("dpt-cu-deterministic",
                  bool((outs[0] == outs[1]).all()), "bit-exact re-run")
            check("dpt-cu-parity",
                  db(ref["OUT"].astype(np.float32), outs[0]) >= 40,
                  f"{db(ref['OUT'].astype(np.float32), outs[0]):.1f}dB (CUDA vs torch)")

    # -- demo-pipeline integrity (no camera; stills fallback end-to-end) ---
    # The live loop itself is demo, never gate. This leg runs the
    # stills-fallback path: capture + preprocess + served step +
    # banner + out-of-tree default.
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import demo_dpt_webcam as DEMO
    check("dpt-demo-outdir", DEMO.OUTDIR.startswith("/tmp"),
          f"default out-of-tree: {DEMO.OUTDIR}")
    check("dpt-demo-banner", "UNTRAINED" in DEMO.BANNER,
          "untrained banner present")
    dtmp = "/tmp/dpttiny_demo_leg"
    shutil.rmtree(dtmp, ignore_errors=True)
    dpaths, dorigin = DEMO.capture("stills", 2, dtmp)
    check("dpt-demo-capture",
          len(dpaths) == 2 and dorigin.startswith("stills")
          and all(os.path.isfile(p) for p in dpaths),
          f"stills fallback: {dorigin}")
    from PIL import Image as _Image
    rgb0 = np.asarray(_Image.open(dpaths[0]).convert("RGB"))
    depth0 = DEMO.served_step(
        os.path.join("/tmp/dpttiny_cu", "dpttiny"),
        [ln.split()[1] for ln in text.splitlines()
         if ln.startswith("IN ")],
        "/tmp/dpttiny_cu", rgb0.astype(np.float64) / 255.0,
        {k: v for k, v in Wd.items()},
        {"patch": ids["patch"], "up16_8": ids["up16_8"],
         "up16_4": ids["up16_4"], "dn16_32": ids["dn16_32"],
         "up8_4": ids["up8_4"], "up4_full": ids["up4_full"],
         "up32_16": ids["up32_16"]}) \
        if (shutil.which("nvcc") is not None and _t.cuda.is_available()
            and os.path.isfile("/tmp/dpttiny_cu/dpttiny")) else None
    if depth0 is None:
        print("SKIP dpt-demo-served (needs built served binary + GPU)")
    else:
        check("dpt-demo-served",
              depth0.shape == (672, 1024) and bool(np.isfinite(depth0).all()),
              "served step finite (672,1024)")
        viz = DEMO.visualize(depth0)
        check("dpt-demo-viz", viz.size == (1024, 672), f"viz {viz.size}")
    repo_leak = [p for p in
                 glob.glob(os.path.join(ROOT, "samples", "**", "*strip*.png"),
                           recursive=True)
                 + glob.glob(os.path.join(ROOT, "samples", "**", "cam*.png"),
                             recursive=True)]
    check("dpt-demo-privacy", repo_leak == [], "no webcam frames in-repo")
    shutil.rmtree(dtmp, ignore_errors=True)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
