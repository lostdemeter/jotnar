"""File-exchange parity: numpy splat_blur vs C splat_exchange, BIT-EXACT.

Covers the whole intelligent path (gradients, tensor, coherence, buckets,
bank, mux) -- not dB, exact equality on out triples + bucket map.
Weight/scalar triples are pre-encoded identically both sides (kernel_triples
/ const_trip values), so this gates the lowering, not the encoder.
Cases: vertical/horizontal bars (oriented), flat (fallback), noise (mixed).
Usage: python3 test_splat_c.py
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
from chain import splat as SP
from chain.control import load_ctrl

FAIL = []
SXCHG = "/tmp/holo_splat_exchange"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def wplanes(f, s, e, z):
    f.write(np.ascontiguousarray(s, np.int8).tobytes())
    f.write(np.ascontiguousarray(e, np.int32).tobytes())
    f.write(np.ascontiguousarray(z, np.uint8).tobytes())


def wkernel(f, kernel):
    ks, ke, kz = H.kernel_triples(kernel)
    f.write(struct.pack("<i", kernel.shape[0]))
    wplanes(f, ks.reshape(-1), ke.reshape(-1), kz.reshape(-1))


def write_in(path, a_trip, kernels, m_acc, m_cov, tq, qlo, qhi, four,
             fused=0):
    Hh, Ww = a_trip[0].shape
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", Hh, Ww, m_acc, m_cov, fused))
        f.write(struct.pack("<3q", tq, qlo, qhi))
        f.write(struct.pack("<b", int(four[0])) + struct.pack("<i", int(four[1]))
                + struct.pack("<B", int(four[2])))
        wplanes(f, a_trip[0], a_trip[1], a_trip[2])
        for k in kernels:
            wkernel(f, k)


def read_out(path, shape):
    n = shape[0] * shape[1]
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    b = np.frombuffer(buf[n + 5 * n:n + 6 * n], np.int8).reshape(shape).copy()
    return s, e, z, b


def one_case(tag, img, fused=0):
    m_acc, m_cov = H._load_scales()
    P = load_ctrl()
    a_t = H.sqrt_trip(S.encode(np.ascontiguousarray(img, np.float64)))
    bank = SP.bank()
    kernels = [SP.SOBEL_X, SP.SOBEL_Y, SP.tensor_smooth_kernel()] + bank
    four = SP.const_trip(4.0, (1,))
    tq = int(S.to_fixed(*SP.const_trip(P["coh_thr"], (1,)), m_cov)[0])
    qlo = int(S.to_fixed(*SP.const_trip(0.0, (1,)), m_cov)[0])
    qhi = int(S.to_fixed(*SP.const_trip(1.0, (1,)), m_cov)[0])
    ref, diag = SP.splat_blur(a_t, m_acc, m_cov)
    pin, pout = "/tmp/holo_splat_in.bin", "/tmp/holo_splat_out.bin"
    write_in(pin, a_t, kernels, m_acc, m_cov, tq, qlo, qhi,
             (four[0][0], four[1][0], four[2][0]), fused=fused)
    r = subprocess.run([SXCHG, pin, pout], capture_output=True, text=True)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    gs, ge, gz, gb = read_out(pout, img.shape)
    ok = bool((gs == ref[0]).all() and (ge == ref[1]).all()
              and (gz == ref[2]).all() and (gb == diag["bucket"]).all())
    if not ok:
        de = int((ge != ref[1]).sum())
        db = int((gb != diag["bucket"]).sum())
        print(f"    out-e diffs={de} bucket diffs={db} n={img.size}")
    check(tag, ok, f"{img.shape} bit-exact" if ok else "MISMATCH (see counts)")


def main():
    if not os.path.exists(SXCHG):
        subprocess.run(["make", "-C", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c_chain"), "check"],
            check=False)
    v = np.zeros((20, 20))
    v[:, 10:] = 0.8
    one_case("splat-x-vert-bar", v)
    h = np.zeros((20, 20))
    h[10:, :] = 0.8
    one_case("splat-x-horiz-bar", h)
    yy, xx = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    one_case("splat-x-diag-slash", np.where(xx + yy > 1.0, 0.8, 0.0))
    one_case("splat-x-diag-backslash", np.where(xx - yy > 0.0, 0.8, 0.0))
    one_case("splat-x-flat", np.full((16, 16), 0.5))
    rng = np.random.default_rng(9)
    one_case("splat-x-noise", np.clip(rng.uniform(0, 1, (20, 20)), 0, 1))
    # fused lowering vs the same compositional reference (bit-exact both)
    one_case("splat-fused-bars", v, fused=1)
    one_case("splat-fused-noise", np.clip(rng.uniform(0, 1, (20, 20)), 0, 1),
             fused=1)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
