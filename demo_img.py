"""Image demo: one geometric program, three backends, screenshots.

Pipeline (2D planes, covered ops only): luma grayscale, invert,
brightness, contrast, 3x3 box blur (valid, nine SLICEs), unsharp mask,
before/after strip. build_img_prog(H, W) generates the listing for any
size (specialized literals -- recompilation on shape change, stated).

Backends: SAME text runs on c/CUDA (float planes) and nonfpu
(triple-encoded U8 planes + triple consts). Kinds come from the sample,
never the text -- the shader promise, demonstrated.
Gallery: python3 demo_img.py (needs PIL + samples/input_example.png).
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain.builder import Prog

CONSTS = ("cr", "cg", "cb", "boost", "csc", "cmid", "amount", "invone",
          "ninth")
IMG_OUTS = ["gray", "inv", "br", "ct", "blur", "us", "strip"]


def build_img_prog(H, W):
    """Geometric image pipeline for HxW planes. Operating point: m=41200
    covers 9-pixel sums (~2300) in ambient (U_m ~= 2700, quantum ~0.01
    -- U8-invisible). Smaller m folds magnitudes (silent saturation);
    the scale IS the exposure, chosen deliberately, not defaulted."""
    p = Prog("img-demo")
    p.config("m_acc", 41200).config("m_cov", 41200)
    p.inp("r", "g", "b", *CONSTS)
    p.op("MUL", "r", "cr", out="t0")
    p.op("MUL", "g", "cg", out="t1")
    p.op("ADD", "t0", "t1", out="t2")
    p.op("MUL", "b", "cb", out="t3")
    p.op("ADD", "t2", "t3", out="gray")
    p.op("SUB", "invone", "gray", out="inv")
    p.op("ADD", "gray", "boost", out="br")
    p.op("SUB", "gray", "cmid", out="t4")
    p.op("MUL", "t4", "csc", out="t5")
    p.op("ADD", "t5", "cmid", out="ct")
    cells = []
    for i, (r0, r1) in enumerate([(0, H - 2), (1, H - 1), (2, H)]):
        for j, (c0, c1) in enumerate([(0, W - 2), (1, W - 1), (2, W)]):
            rs = p.op("SLICE", "gray", 0, r0, r1, out=f"r{i}{j}")
            cells.append(p.op("SLICE", rs, 1, c0, c1, out=f"b{i}{j}"))
    acc = cells[0]
    for k, c in enumerate(cells[1:]):
        acc = p.op("ADD", acc, c, out=f"s{k}")
    p.op("MUL", acc, "ninth", out="blur")
    crop = p.op("SLICE", p.op("SLICE", "gray", 0, 1, H - 1, out="cr0"),
                1, 1, W - 1, out="crop")
    p.op("SUB", "crop", "blur", out="detail")
    p.op("MUL", "amount", "detail", out="scaled")
    p.op("ADD", "crop", "scaled", out="us")
    p.op("CONCAT", "crop", "blur", 1, out="strip")
    return p


def const_vals():
    return {"cr": 0.299, "cg": 0.587, "cb": 0.114, "boost": 30.0,
            "csc": 1.4, "cmid": 128.0, "amount": 1.5, "invone": 255.0,
            "ninth": 1.0 / 9.0}


def tenc(a):
    """Exact-ish triple encode of an integer-valued array."""
    a = np.ascontiguousarray(a)
    ss = np.zeros_like(a, np.int8)
    ee = np.zeros_like(a, np.int32)
    zz = np.zeros_like(a, np.uint8)
    for v in np.unique(a):
        ws, we, wz = S.encode(np.array([float(v)]))
        m = a == v
        ss[m], ee[m], zz[m] = ws[0], we[0], wz[0]
    return (np.ascontiguousarray(ss), np.ascontiguousarray(ee),
            np.ascontiguousarray(zz))


def tdec(d):
    return (S.decode(np.ascontiguousarray(d["s"]), np.ascontiguousarray(d["e"]))
            * (1 - np.ascontiguousarray(d["z"]).astype(np.float64)))


def run_backend(text, target, payload, work, name, libs_sentinel="dflt"):
    """Compile + run one backend. Returns ({stream: float array}, art).
    T outputs are decoded to float here (bit-exactness is gated in tests,
    not in the demo driver)."""
    from chain.emit_c import compile_program, build
    art = compile_program(text, target, sample=payload, outputs=IMG_OUTS,
                          basedir="programs")
    if target == "cuda":
        from chain.emit_cuda import build_cu
        exe = build_cu(art["source"], work, name=name)
    else:
        kw = {} if libs_sentinel == "dflt" else {"libs": libs_sentinel}
        exe = build(art["source"], work, name=name, **kw)
    argv = [exe]
    fdt = np.float32 if target == "cuda" else np.float64
    for n in art["inputs"]:
        k, _, _ = art["streams"][n]
        v = payload[n]
        if k == "T":
            for comp, arr in zip("sez", v):
                fn = os.path.join(work, f"in_{n}_{comp}.bin")
                np.ascontiguousarray(arr).tofile(fn)
                argv.append(fn)
        elif k == "F":
            fn = os.path.join(work, f"in_{n}.bin")
            np.ascontiguousarray(v, dtype=fdt).tofile(fn)
            argv.append(fn)
        else:
            fn = os.path.join(work, f"in_{n}.bin")
            np.ascontiguousarray(v, dtype=np.int64).tofile(fn)
            argv.append(fn)
    ofns = []
    for o in art["outputs"]:
        ok = art["streams"][o][0]
        if ok == "T":
            cur = []
            for c in "sez":
                fn = os.path.join(work, f"out_{o}_{c}.bin")
                cur.append(fn)
                argv.append(fn)
            ofns.append((o, cur))
        else:
            fn = os.path.join(work, f"out_{o}.bin")
            ofns.append((o, fn))
            argv.append(fn)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{target} run failed rc={r.returncode}: "
                           f"{r.stderr[:300]}")
    res = {}
    odt = np.float32 if target == "cuda" else np.float64
    for o, fn in ofns:
        if isinstance(fn, list):
            dt = {"s": np.int8, "e": np.int32, "z": np.uint8}
            res[o] = tdec({c: np.fromfile(f, dtype=dt[c])
                           for c, f in zip("sez", fn)})
        else:
            res[o] = np.fromfile(fn, dtype=odt).astype(np.float64)
    return res, art


def main():
    from PIL import Image
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "samples", "input_example.png")
    if not os.path.isfile(src):
        print("SKIP (no samples/input_example.png)")
        sys.exit(0)
    im = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
    H, W, _ = im.shape
    print(f"input: {W}x{H} RGB")
    prog = build_img_prog(H, W)
    text = prog.text()
    open(os.path.join(root, "programs", "img_demo.asm"), "w").write(
        "# Generated by demo_img.build_img_prog -- do not hand-edit.\n"
        + text)
    nops = sum(1 for ln in text.splitlines()
               if "=" in ln and not ln.strip().startswith(("IN", "CONFIG")))
    print(f"listing: {nops} ops")
    shapes = {k: (H, W) for k in IMG_OUTS}
    shapes["strip"] = (H - 2, 2 * (W - 2))
    shapes["blur"] = (H - 2, W - 2)
    shapes["us"] = (H - 2, W - 2)
    cv = const_vals()
    fpayload = {"r": np.ascontiguousarray(im[:, :, 0]),
                "g": np.ascontiguousarray(im[:, :, 1]),
                "b": np.ascontiguousarray(im[:, :, 2])}
    fpayload.update({k: np.full((H, W), v) for k, v in cv.items()
                     if k not in ("ninth", "amount")})
    fpayload["ninth"] = np.full((H - 2, W - 2), cv["ninth"])
    fpayload["amount"] = np.full((H - 2, W - 2), cv["amount"])
    outs = {}
    for target, tag in (("c", "c"), ("cuda", "cu")):
        work = f"/tmp/img_{tag}"
        os.makedirs(work, exist_ok=True)
        res, _ = run_backend(text, target, dict(fpayload), work, tag)
        outs[target] = {k: v.reshape(shapes[k]) for k, v in res.items()}
        print(f"{target}: ok")
    tpayload = {k: tenc(np.rint(fpayload[k]).astype(np.int64))
                for k in ("r", "g", "b")}
    tpayload.update({k: tenc(np.full((H, W), v)) for k, v in cv.items()
                     if k not in ("ninth", "amount")})
    tpayload["ninth"] = tenc(np.full((H - 2, W - 2), cv["ninth"]))
    tpayload["amount"] = tenc(np.full((H - 2, W - 2), cv["amount"]))
    # Fractional consts encode to nearest-triple (lattice quantum ~1e-4
    # relative -- invisible in U8, measured as PSNR below).
    work = "/tmp/img_nf"
    os.makedirs(work, exist_ok=True)
    from chain.emit_nonfpu import float_trap
    from chain.emit_c import compile_program
    _art_probe = compile_program(text, "nonfpu", sample=tpayload,
                                 outputs=list(IMG_OUTS), basedir="programs")
    float_trap(_art_probe["source"])
    res_nf, _ = run_backend(text, "nonfpu", tpayload, work, "nf",
                            libs_sentinel=[])
    outs["nonfpu"] = {k: v.reshape(shapes[k]) for k, v in res_nf.items()}
    print("nonfpu: ok (trap-clean, no libm)")
    for o in IMG_OUTS:
        a = outs["c"][o]
        dc = float(np.abs(outs["cuda"][o].reshape(a.shape) - a).max())
        dn = float(np.abs(outs["nonfpu"][o].reshape(a.shape) - a).max())
        mse = float(((outs["nonfpu"][o].reshape(a.shape) - a) ** 2).mean())
        psnr = 10 * np.log10(255 * 255 / max(mse, 1e-12))
        print(f"{o:6s}: cuda-vs-c={dc:.2e} nonfpu-vs-c={dn:.3f} "
              f"PSNR={psnr:.1f}dB")
    gal = os.path.join(root, "gallery")
    os.makedirs(gal, exist_ok=True)

    def save(name, arr):
        Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8)).save(
            os.path.join(gal, name))

    save("input.png", im)
    for o in IMG_OUTS:
        save(f"c_{o}.png", outs["c"][o])
    save("cu_us.png", outs["cuda"]["us"])
    save("nf_us.png", outs["nonfpu"]["us"])
    save("cu_strip.png", outs["cuda"]["strip"])
    print(f"gallery: {gal} ({len(os.listdir(gal))} files)")


if __name__ == "__main__":
    main()
