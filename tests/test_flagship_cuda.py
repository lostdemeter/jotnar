"""Flagship CUDA parity: full splat_soft+v5 pipeline A->YENH on device.

Host (float, boundary by doctrine) does decode/luma/encode-A with the SAME
numpy calls the int path uses; the binary runs the entire triple pipeline
(structure tensor, coherence+buckets, soft bank blend, v5 gate, finish) in
~50 kernel launches. Gate is BIT-EXACT on YENH triples vs
enhance_luminance_int (composition of unit-gated kernels + driver-side
composition with no new math). Rows: synthetic, real frame, non-default
m_acc (scale plumbing end-to-end).
Build: `make -C c_chain cuda` (needs nvcc + GPU); without nvcc SKIPs.
Usage: python3 tests/test_flagship_cuda.py
"""
import os
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from chain import holo_phi as H
from chain import splat as SP
from chain import control as C

FAIL = []
FXCHG = "/tmp/holo_flagship_cuda_exchange"
BETA = 0.5


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def const_plane(v, shape):
    return S.encode(np.full(shape, float(v), dtype=np.float64))


def write_in(path, y, m_acc, m_cov):
    from chain.substrate import build_flagship_inbin
    with open(path, "wb") as f:
        f.write(build_flagship_inbin(y, m_acc, m_cov, BETA))


def read_yenh(path, shape):
    n = int(np.prod(shape))
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    return s, e, z


def one_case(tag, y, m_acc, m_cov):
    y = np.ascontiguousarray(y, dtype=np.float64)
    ref, _ = H.enhance_luminance_int(y, beta=BETA, m_acc=m_acc, m_cov=m_cov,
                                     blur="splat_soft", ctrl="v5")
    pin, pout = "/tmp/holo_flagship_in.bin", "/tmp/holo_flagship_out.bin"
    write_in(pin, y, m_acc, m_cov)
    r = subprocess.run([FXCHG, pin, pout], capture_output=True, text=True)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    got = read_yenh(pout, y.shape)
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    if not same:
        ns = int((got[0] != ref[0]).sum())
        ne = int((got[1] != ref[1]).sum())
        nz = int((got[2] != ref[2]).sum())
        gv = S.decode(got[0], got[1]) * (1 - got[2].astype(np.float64))
        rv = S.decode(ref[0], ref[1]) * (1 - ref[2].astype(np.float64))
        mse = float(np.mean((gv - rv) ** 2))
        db = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
        check(tag, False, f"ds={ns} de={ne} dz={nz} ({db:.1f}dB fallback)")
    else:
        check(tag, True, f"{y.shape} bit-exact YENH")


def main():
    cc = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "c_chain")
    if not os.path.exists(FXCHG):
        r = subprocess.run(["make", "-C", cc, "cuda"],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"SKIP (needs nvcc + GPU: {r.stderr[-300:]})")
            sys.exit(0)
    ma, mc = H._load_scales()
    gx, gy = np.meshgrid(np.linspace(0.05, 1.0, 16), np.linspace(0.05, 1.0, 16))
    synth = (((gx + gy) / 2)).astype(np.float64)
    synth[8, :] = 0.95
    synth[0, 0] = 1.0
    one_case("flagship-cuda-synth", synth, ma, mc)
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ext = os.path.join(root, "..", "rife_reverse", "samples", "f_012.png")
    cand = ext if os.path.isfile(ext) else os.path.join(root, "samples", "input_example.png")
    rgb = np.asarray(Image.open(cand).convert("RGB"), dtype=np.float64) / 255.0
    y = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
         + 0.0722 * rgb[:, :, 2]).astype(np.float64)
    one_case("flagship-cuda-real", y, ma, mc)
    one_case("flagship-cuda-bigm", synth, 35492, mc)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
