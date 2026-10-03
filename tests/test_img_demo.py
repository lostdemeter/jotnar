"""Image-demo gate: one program, three backends, synthetic 16x12 image.

Gates: C-vs-numpy bit-exact (float path is exact arithmetic here);
CUDA-vs-C tight eps (FP32); nonfpu-vs-lattice BIT-EXACT (same
fixed-point math -- the strong claim); trap-clean + no-libm on the
nonfpu source. Fast: 16x12 gradient+square, no files needed.
Usage: python3 tests/test_img_demo.py (needs cc + nvcc + GPU).
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain import asm as ASM
from chain.asm_ops import REGISTRY, SIGS
from demo_img import IMG_OUTS, build_img_prog, const_vals, tenc

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    H, W = 12, 16
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    r = (xx * 16) % 256
    g = (yy * 21) % 256
    b = ((xx + yy) * 8) % 256
    r[3:7, 4:9] = 200.0
    prog = build_img_prog(H, W)
    text = prog.text()
    cv = const_vals()
    fp = {"r": np.ascontiguousarray(r), "g": np.ascontiguousarray(g),
          "b": np.ascontiguousarray(b)}
    for k, v in cv.items():
        sh = (H - 2, W - 2) if k in ("ninth", "amount") else (H, W)
        fp[k] = np.full(sh, v)
    shapes = {k: (H, W) for k in IMG_OUTS}
    shapes["strip"] = (H - 2, 2 * (W - 2))
    shapes["blur"] = (H - 2, W - 2)
    shapes["us"] = (H - 2, W - 2)
    # numpy reference for the pointwise head (luma/invert/bright/contrast)
    ref_gray = 0.299 * r + 0.587 * g + 0.114 * b
    tp = {k: tenc(np.rint(fp[k]).astype(np.int64)) for k in ("r", "g", "b")}
    tp.update({k: tenc(v) for k, v in fp.items() if k not in ("r", "g", "b")})
    fref = ASM.run_text(text, REGISTRY, tp, sigs=SIGS, basedir="programs")

    from chain.emit_c import compile_program, build
    from chain.emit_nonfpu import float_trap

    # -- C exact vs numpy -------------------------------------------
    art = compile_program(text, "c", sample=dict(fp), outputs=list(IMG_OUTS),
                          basedir="programs")
    work = "/tmp/img_test_c"
    os.makedirs(work, exist_ok=True)
    exe = build(art["source"], work, name="t")
    argv = [exe]
    for n in art["inputs"]:
        fn = os.path.join(work, f"in_{n}.bin")
        fp[n].astype(np.float64).tofile(fn)
        argv.append(fn)
    ofns = []
    for o in art["outputs"]:
        fn = os.path.join(work, f"out_{o}.bin")
        ofns.append(fn)
        argv.append(fn)
    r_ = subprocess.run(argv, capture_output=True, text=True)
    check("img-c-run", r_.returncode == 0, f"rc={r_.returncode}")
    got_c = {}
    for o, fn in zip(art["outputs"], ofns):
        got_c[o] = np.fromfile(fn, dtype=np.float64).reshape(shapes[o])
    check("img-c-luma-exact",
          bool((got_c["gray"] == ref_gray).all()), "float path exact")

    # -- CUDA tight vs C ----------------------------------------------
    if shutil.which("nvcc") is None:
        print("SKIP cuda (no nvcc)")
    else:
        from chain.emit_cuda import build_cu
        art2 = compile_program(text, "cuda", sample=dict(fp),
                               outputs=list(IMG_OUTS), basedir="programs")
        exe2 = build_cu(art2["source"], work, name="tu")
        argv = [exe2]
        for n in art2["inputs"]:
            fn = os.path.join(work, f"uin_{n}.bin")
            fp[n].astype(np.float32).tofile(fn)
            argv.append(fn)
        ofns = []
        for o in art2["outputs"]:
            fn = os.path.join(work, f"uout_{o}.bin")
            ofns.append(fn)
            argv.append(fn)
        r_ = subprocess.run(argv, capture_output=True, text=True)
        check("img-cu-run", r_.returncode == 0, f"rc={r_.returncode}")
        if r_.returncode == 0:
            dmax = 0.0
            for o, fn in zip(art2["outputs"], ofns):
                g = np.fromfile(fn, dtype=np.float32).reshape(shapes[o])
                dmax = max(dmax, float(np.abs(g - got_c[o]).max()))
            check("img-cu-tight", dmax < 1e-3, f"maxabs={dmax:.1e}")

    # -- nonfpu bit-exact vs lattice ------------------------------------
    art3 = compile_program(text, "nonfpu", sample=tp, outputs=list(IMG_OUTS),
                           basedir="programs")
    try:
        float_trap(art3["source"])
        check("img-nf-trap", True, "")
    except Exception as e:  # noqa: BLE001
        check("img-nf-trap", False, str(e)[:120])
    exe3 = build(art3["source"], work, name="tn", libs=[])
    argv = [exe3]
    for n in art3["inputs"]:
        for comp, arr in zip("sez", tp[n]):
            fn = os.path.join(work, f"nin_{n}_{comp}.bin")
            np.ascontiguousarray(arr).tofile(fn)
            argv.append(fn)
    ofns = []
    for o in art3["outputs"]:
        cur = []
        for c in "sez":
            fn = os.path.join(work, f"nout_{o}_{c}.bin")
            cur.append(fn)
            argv.append(fn)
        ofns.append((o, cur))
    r_ = subprocess.run(argv, capture_output=True, text=True)
    check("img-nf-run", r_.returncode == 0, f"rc={r_.returncode}")
    if r_.returncode == 0:
        ok = True
        for o, fns in ofns:
            ref = fref[o]
            for c, dt, fn in zip("sez", (np.int8, np.int32, np.uint8), fns):
                gotp = np.fromfile(fn, dtype=dt).reshape(shapes[o])
                same = bool((gotp == np.ascontiguousarray(
                    ref["sez".index(c)])).all())
                ok = ok and same
        check("img-nf-exact", ok, "all outs, all planes vs lattice")

    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
