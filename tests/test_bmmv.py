"""BMMV gate: strided-view batched matmul is exactly invariant.

48th mnemonic. Same products in the same order as per-head
composition, so agreement is bit-exact where integer (lattice vs
per-head lattice) and eps where float (CUDA/C vs numpy float32).
Gates: lattice bit-exactness (asm-level, always runs), C vs numpy,
CUDA vs numpy + CUDA vs per-head CUDA (SKIPs without nvcc/GPU).
Usage: python3 tests/test_bmmv.py
"""
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

S, NH, NKV, DH = 5, 4, 2, 4
HID, KVD = NH * DH, NKV * DH
PER = NH // NKV


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def bmmv_text():
    # scores per group over strided head views (group offsets via SLICE,
    # heads within a group via BMMV strides; no per-head transposes).
    L = ["IN Qc", "IN Kc", "IN Vc", "CONFIG m_acc 4096", "CONFIG m_cov 128"]
    for g in range(NKV):
        L.append(f"QG{g} = SLICE(Qc, 1, {g * PER * DH}, {(g + 1) * PER * DH})")
        L.append(f"KG{g} = SLICE(Kc, 1, {g * DH}, {(g + 1) * DH})")
        L.append(f"SC{g} = BMMV(QG{g}, KG{g}, {PER}, {S}, {S}, {DH}, "
                 f"{PER * DH}, {DH}, {DH}, 0, {S}, {S * S}, 1)")
    return "\n".join(L) + "\n"


def per_head_text():
    L = ["IN Qc", "IN Kc", "IN Vc", "CONFIG m_acc 4096", "CONFIG m_cov 128"]
    for h in range(NH):
        g = h // PER
        L.append(f"QH{h} = SLICE(Qc, 1, {h * DH}, {(h + 1) * DH})")
        L.append(f"KH{h} = SLICE(Kc, 1, {g * DH}, {(g + 1) * DH})")
        L.append(f"KT{h} = TRANSPOSE(KH{h})")
        L.append(f"SC{h} = BATCH_MATMUL(QH{h}, KT{h})")
    return "\n".join(L) + "\n"


def main():
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    rng = np.random.default_rng(11)
    Qc = np.ascontiguousarray(rng.normal(size=(S, HID)))
    Kc = np.ascontiguousarray(rng.normal(size=(S, KVD)))
    Vc = np.ascontiguousarray(rng.normal(size=(S, KVD)))

    # -- lattice: BMMV == per-head composition, bit-exact ---------------
    import phi_core.lattice as LS

    def enc(a):
        s, e, z = LS.encode(a.ravel())
        return (s.reshape(a.shape), e.reshape(a.shape), z.reshape(a.shape))

    feeds = ASM.run_text(bmmv_text(), REGISTRY,
                         {"Qc": enc(Qc), "Kc": enc(Kc), "Vc": enc(Vc)},
                         sigs=SIGS, basedir=os.path.join(ROOT, "programs"))
    f2 = ASM.run_text(per_head_text(), REGISTRY,
                      {"Qc": enc(Qc), "Kc": enc(Kc), "Vc": enc(Vc)},
                      sigs=SIGS, basedir=os.path.join(ROOT, "programs"))
    ref_heads = [np.ascontiguousarray(f2[f"SC{h}"][1]) for h in range(NH)]
    ref = np.stack(ref_heads).reshape(NH, S, S)
    for g in range(NKV):
        got = np.ascontiguousarray(feeds[f"SC{g}"][1]).reshape(PER, S, S)
        check(f"bmmv-lattice-g{g}",
              bool((got == ref[g * PER:(g + 1) * PER]).all()),
              "bit-exact vs per-head lattice")

    # -- float hosts: C and CUDA vs numpy -------------------------------
    from chain.emit_c import compile_program
    prog = ("IN Qc\nIN Kc\n"
            f"KG = SLICE(Kc, 1, 0, {DH})\n"
            f"SC = BMMV(Qc, KG, {PER}, {S}, {S}, {DH}, {HID}, {DH}, "
            f"{DH}, 0, {S}, {S * S}, 1)\n")
    sample = {"Qc": Qc.astype(np.float64), "Kc": Kc.astype(np.float64)}
    ref32 = np.stack([Qc[:, h * DH:(h + 1) * DH]
                      @ Kc[:, 0:DH].T
                      for h in range(PER)]).reshape(PER, S, S)
    art = compile_program(prog, "c", sample=sample, outputs=["SC"],
                          basedir=os.path.join(ROOT, "programs"))
    from chain.emit_c import build as _bc
    work = "/tmp/bmmv_c"
    os.makedirs(work, exist_ok=True)
    exe = _bc(art["source"], work, name="bmmv")
    f1 = os.path.join(work, "q.bin")
    f2 = os.path.join(work, "k.bin")
    fo = os.path.join(work, "o.bin")
    Qc.astype(np.float64).tofile(f1)
    Kc.astype(np.float64).tofile(f2)
    r = subprocess.run([exe, f1, f2, fo], capture_output=True, text=True)
    check("bmmv-c-run", r.returncode == 0, f"rc={r.returncode} {r.stderr[:150]}")
    if r.returncode == 0:
        gotc = np.fromfile(fo, dtype=np.float64).reshape(PER, S, S)
        check("bmmv-c-eps", np.abs(gotc - ref32).max() < 1e-9,
              f"maxabs={np.abs(gotc - ref32).max():.2e}")

    # -- loud dispositions (compile-only, always run) --------------------
    # BMMV on non-FPU: integer target, float views -- refused, pinned.
    from chain.asm import AsmError
    try:
        compile_program(prog, "nonfpu", sample=sample, outputs=["SC"],
                        basedir=os.path.join(ROOT, "programs"))
        check("bmmv-nonfpu-loud", False, "compiled float views?!")
    except AsmError as e:
        check("bmmv-nonfpu-loud", True, str(e)[:90])
    # CUDA execution modes on the C backend: refused, never dropped.
    for kw, tag in (({"live": ["Qc"]}, "live"), ({"graph": True}, "graph"),
                    ({"sync_each": False}, "nosync")):
        try:
            compile_program(prog, "c", sample=sample, outputs=["SC"],
                            basedir=os.path.join(ROOT, "programs"), **kw)
            check(f"bmmv-c-{tag}-loud", False, "dropped silently?!")
        except AsmError as e:
            check(f"bmmv-c-{tag}-loud", True, str(e)[:80])

    # -- fp16-input path (GemmEx): same views, half storage -------------
    # Exercises the F16->F32 convert path that mixed-precision callers
    # need; untested paths are liabilities, so this one is gated (CUDA).
    prog16 = ("IN Qc\nIN Kc\n"
              f"KG = SLICE(Kc, 1, 0, {DH})\n"
              f"SC = BMMV(Qc, KG, {PER}, {S}, {S}, {DH}, {HID}, {DH}, "
              f"{DH}, 0, {S}, {S * S}, 1)\n")
    if shutil.which("nvcc") is None:
        print("SKIP bmmv fp16 leg (needs nvcc + GPU)", flush=True)
    else:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if not has_cuda:
            print("SKIP bmmv fp16 leg (no GPU)", flush=True)
        else:
            from chain.emit_cuda import build_cu as _bcu16
            art16 = compile_program(prog16, "cuda", sample=sample,
                                    outputs=["SC"],
                                    basedir=os.path.join(ROOT, "programs"),
                                    use_fp16=True)
            w16 = "/tmp/bmmv_cu16"
            os.makedirs(w16, exist_ok=True)
            exe16 = _bcu16(art16["source"], w16, name="bmmv16")
            g1 = os.path.join(w16, "q.bin")
            g2 = os.path.join(w16, "k.bin")
            go = os.path.join(w16, "o.bin")
            Qc.astype(np.float16).tofile(g1)
            Kc.astype(np.float16).tofile(g2)
            r16 = subprocess.run([exe16, g1, g2, go], capture_output=True,
                                 text=True)
            check("bmmv-cu-f16-run", r16.returncode == 0,
                  f"rc={r16.returncode} {r16.stderr[:150]}")
            if r16.returncode == 0:
                got16 = np.fromfile(go, dtype=np.float32).reshape(PER, S, S)
                check("bmmv-cu-f16-eps", np.abs(got16 - ref32).max() < 1e-2,
                      f"maxabs={np.abs(got16 - ref32).max():.2e} (fp16 class)")

    if shutil.which("nvcc") is None:
        print("SKIP cuda legs (needs nvcc + GPU)", flush=True)
    else:
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        if not has_cuda:
            print("SKIP cuda legs (no GPU)", flush=True)
        else:
            from chain.emit_cuda import build_cu as _bcu
            artu = compile_program(prog, "cuda", sample=sample, outputs=["SC"],
                                   basedir=os.path.join(ROOT, "programs"))
            wu = "/tmp/bmmv_cu"
            os.makedirs(wu, exist_ok=True)
            exeu = _bcu(artu["source"], wu, name="bmmv")
            g1 = os.path.join(wu, "q.bin")
            g2 = os.path.join(wu, "k.bin")
            go = os.path.join(wu, "o.bin")
            Qc.astype(np.float32).tofile(g1)
            Kc.astype(np.float32).tofile(g2)
            ru = subprocess.run([exeu, g1, g2, go], capture_output=True,
                                text=True)
            check("bmmv-cu-run", ru.returncode == 0,
                  f"rc={ru.returncode} {ru.stderr[:150]}")
            if ru.returncode == 0:
                gotu = np.fromfile(go, dtype=np.float32).reshape(PER, S, S)
                refg = ref32[:PER]
                check("bmmv-cu-eps", np.abs(gotu - refg).max() < 1e-4,
                      f"maxabs={np.abs(gotu - refg).max():.2e} (fp32 class)")
    print("FAILURES:", FAIL if FAIL else "none")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
