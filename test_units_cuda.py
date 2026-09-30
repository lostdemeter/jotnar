"""CUDA units parity: k_conv_rep + k_mux vs proven references, BIT-EXACT.

Each kernel mirrors its proven C twin line-for-line (holo_conv.c /
holo_ops.c holo_mux, both file-exchange-gated vs numpy), so the gate
isolates CUDA parallelism slips: conv rows reuse test_c_conv fixtures +
numpy ref; mux rows compare vs the promoted select_mux helper (in-range)
and the documented clamp contract (out-of-range buckets).
Driver: units_cuda_exchange (task 0=conv, 1=mux).
Build: `make -C c_chain cuda` (needs nvcc + GPU); without nvcc SKIPs.
Usage: python3 test_units_cuda.py
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phi-core")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import holo_phi as H
import test_c_conv as TC

FAIL = []
UXCHG = "/tmp/holo_units_cuda_exchange"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def write_conv(path, a_trip, k_trip, kh, m_acc, m_cov):
    Hh, Ww = a_trip[0].shape
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 0))
        f.write(struct.pack("<5i", Hh, Ww, kh, m_acc, m_cov))
        for t in (a_trip, (k_trip[0].reshape(-1), k_trip[1].reshape(-1),
                            k_trip[2].reshape(-1))):
            for plane, dt in ((t[0], np.int8), (t[1], np.int32),
                              (t[2], np.uint8)):
                f.write(np.ascontiguousarray(plane, dt).tobytes())


def write_mux(path, streams, bucket):
    n = streams[0][0].size
    ns = len(streams)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<2i", n, ns))
        # planar across streams (matches the driver: all-s, all-e, all-z)
        for pi, dt in ((0, np.int8), (1, np.int32), (2, np.uint8)):
            for t in streams:
                f.write(np.ascontiguousarray(t[pi].reshape(-1), dt).tobytes())
        f.write(np.ascontiguousarray(bucket.reshape(-1), np.int8).tobytes())


def read_planar(path, shape):
    n = int(np.prod(shape))
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    return s, e, z


def run_bin(path_in, path_out):
    r = subprocess.run([UXCHG, path_in, path_out],
                       capture_output=True, text=True)
    return r


def one_conv(tag, img, kernel):
    m_acc, m_cov = H._load_scales()
    kh = kernel.shape[0]
    a_t = H.sqrt_trip(S.encode(np.ascontiguousarray(img, np.float64)))
    k_t = TC.encode_kernel(kernel)
    ref = TC.conv_numpy_with_triples(a_t, k_t, kh, m_acc, m_cov)
    pin, pout = "/tmp/holo_units_conv_in.bin", "/tmp/holo_units_conv_out.bin"
    write_conv(pin, a_t, k_t, kh, m_acc, m_cov)
    r = run_bin(pin, pout)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    got = read_planar(pout, img.shape)
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    check(tag, same, f"{img.shape} KH={kh} bit-exact" if same else "MISMATCH")


def one_mux(tag, streams, bucket, ref):
    pin, pout = "/tmp/holo_units_mux_in.bin", "/tmp/holo_units_mux_out.bin"
    write_mux(pin, streams, bucket)
    r = run_bin(pin, pout)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    got = read_planar(pout, streams[0][0].shape)
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    check(tag, same, "bit-exact" if same else "MISMATCH")


def main():
    cc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_chain")
    if not os.path.exists(UXCHG):
        r = subprocess.run(["make", "-C", cc, "cuda"],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"SKIP (needs nvcc + GPU: {r.stderr[-300:]})")
            sys.exit(0)
    rng = np.random.default_rng(7)
    k5 = H.gaussian_kernel(radius=2, sigma=1.0)
    Hh, Ww = 16, 16
    gx, gy = np.meshgrid(np.linspace(0.05, 1.0, Ww), np.linspace(0.05, 1.0, Hh))
    img = ((gx + gy) / 2).astype(np.float64)
    img[8, :] = 0.95
    img[0, 0] = 1.0
    img[15, 15] = 0.02
    one_conv("units-cuda-conv-fig", img, k5)
    one_conv("units-cuda-conv-flat", np.full((16, 16), 0.5), k5)
    corner = np.full((12, 10), 0.3)
    corner[0, 0] = 1.0
    one_conv("units-cuda-conv-corner", corner,
             H.gaussian_kernel(radius=1, sigma=1.0))
    one_conv("units-cuda-conv-noise",
             np.clip(rng.uniform(0, 1, (16, 16)), 0, 1), k5)
    # mux: 5 streams, in-range buckets vs promoted helper ...
    rng2 = np.random.default_rng(9)
    streams = [S.encode(rng2.uniform(-1, 1, (6, 6))) for _ in range(5)]
    b_in = rng2.integers(0, 5, (6, 6)).astype(np.int8)
    one_mux("units-cuda-mux", streams, b_in, H.select_mux(streams, b_in))
    # ... out-of-range buckets vs the documented clamp contract.
    b_oob = np.array([[-1, 5, 7, 0, 2, 99]], np.int8)
    streams2 = [S.encode(rng2.uniform(-1, 1, (1, 6))) for _ in range(5)]
    bc = np.clip(b_oob, 0, 4)
    ref = (np.stack([streams2[bc[0, i]][0][0, i] for i in range(6)]).reshape(1, 6).astype(np.int8),
           np.stack([streams2[bc[0, i]][1][0, i] for i in range(6)]).reshape(1, 6).astype(np.int32),
           np.stack([streams2[bc[0, i]][2][0, i] for i in range(6)]).reshape(1, 6).astype(np.uint8))
    one_mux("units-cuda-mux-clamp", streams2, b_oob, ref)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
