"""File-exchange parity: numpy conv_trip vs C holo_conv, BIT-EXACT.

Not dB -- exact equality on (s,e,z). The C binary reads triples written by
numpy (identical weight triples, no float encode divergence), runs the
lowering, writes triples back. Any mismatch = lowering damage.
Cases: gradient+step+impulse (edge/replicate stress), flat, corner impulse.
Usage: python3 test_c_conv.py
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/home/thorin/Documents/OpenCode/phi-core")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phi_core.lattice as S
from chain import holo_phi as H

FAIL = []
XCHG = "/tmp/holo_conv_exchange"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def write_in(path, a_trip, k_trip, kh, m_acc, m_cov):
    Hh, Ww = a_trip[0].shape
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", Hh, Ww, kh, m_acc, m_cov))
        f.write(np.ascontiguousarray(a_trip[0], np.int8).tobytes())
        f.write(np.ascontiguousarray(a_trip[1], np.int32).tobytes())
        f.write(np.ascontiguousarray(a_trip[2], np.uint8).tobytes())
        f.write(np.ascontiguousarray(k_trip[0].reshape(-1), np.int8).tobytes())
        f.write(np.ascontiguousarray(k_trip[1].reshape(-1), np.int32).tobytes())
        f.write(np.ascontiguousarray(k_trip[2].reshape(-1), np.uint8).tobytes())


def read_out(path, shape):
    n = shape[0] * shape[1]
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    return s, e, z


def encode_kernel(kernel):
    kh = kernel.shape[0]
    ks = np.empty((kh, kh), np.int8)
    ke = np.empty((kh, kh), np.int32)
    kz = np.empty((kh, kh), np.uint8)
    for dy in range(kh):
        for dx in range(kh):
            # identical per-tap encode as conv_trip (same boundary values)
            ws, we, wz = S.encode(np.array([float(kernel[dy, dx])]))
            ks[dy, dx], ke[dy, dx], kz[dy, dx] = ws[0], we[0], wz[0]
    ktap = (ks, ke, kz)
    # rebuild the float kernel conv_trip would see is unnecessary: conv_trip
    # re-encodes from floats, but we pass pre-encoded triples via a shim below.
    return ktap


def conv_numpy_with_triples(a_trip, k_trip, kh, m_acc, m_cov):
    """conv_trip but with pre-encoded kernel triples (avoids float re-encode
    divergence -- tests the lowering, not the encoder)."""
    r = kh // 2
    Hh, Ww = a_trip[0].shape

    def pad(x):
        return np.pad(x, ((r, r), (r, r)), mode="edge")
    ps, pe, pz = pad(a_trip[0]), pad(a_trip[1]), pad(a_trip[2])
    acc = None
    for dy in range(kh):
        for dx in range(kh):
            tap = (ps[dy:dy + Hh, dx:dx + Ww].astype(np.int8),
                   pe[dy:dy + Hh, dx:dx + Ww], pz[dy:dy + Hh, dx:dx + Ww])
            prod = H.tmul(tap, (np.full((Hh, Ww), k_trip[0][dy, dx], np.int8),
                                np.full((Hh, Ww), k_trip[1][dy, dx], np.int32),
                                np.full((Hh, Ww), k_trip[2][dy, dx], np.uint8)))
            q = S.to_fixed(prod[0], prod[1], prod[2], m_acc)
            acc = q if acc is None else acc + q
    return S.from_fixed(H.rescale_(acc, m_acc, m_cov), m_cov)


def one_case(tag, img, kernel):
    m_acc, m_cov = H._load_scales()
    kh = kernel.shape[0]
    a_t = H.sqrt_trip(S.encode(np.ascontiguousarray(img, np.float64)))
    k_t = encode_kernel(kernel)
    ref = conv_numpy_with_triples(a_t, k_t, kh, m_acc, m_cov)
    pin, pout = "/tmp/holo_conv_in.bin", "/tmp/holo_conv_out.bin"
    write_in(pin, a_t, k_t, kh, m_acc, m_cov)
    r = subprocess.run([XCHG, pin, pout], capture_output=True, text=True)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    got = read_out(pout, img.shape)
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    if not same:
        ns = int((got[0] != ref[0]).sum())
        ne = int((got[1] != ref[1]).sum())
        nz = int((got[2] != ref[2]).sum())
        de = np.abs(got[1].astype(np.int64) - ref[1].astype(np.int64))
        check(tag, False, f"ds={ns} de={ne} dz={nz} maxede={int(de.max())}")
    else:
        check(tag, True, f"{img.shape} KH={kh} bit-exact")


def main():
    if not os.path.exists(XCHG):
        r = subprocess.run(["make", "-C", os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "c_chain"), "check"],
            capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-1500:], r.stderr[-1500:])
    rng = np.random.default_rng(7)
    k5 = H.gaussian_kernel(radius=2, sigma=1.0)
    # gradient + step + impulse (replicate-edge + interior stress)
    Hh, Ww = 16, 16
    gx, gy = np.meshgrid(np.linspace(0.05, 1.0, Ww), np.linspace(0.05, 1.0, Hh))
    img = ((gx + gy) / 2).astype(np.float64)
    img[8, :] = 0.95
    img[0, 0] = 1.0
    img[15, 15] = 0.02
    one_case("conv-exchange-fig", img, k5)
    one_case("conv-exchange-flat", np.full((16, 16), 0.5), k5)
    corner = np.full((12, 10), 0.3)
    corner[0, 0] = 1.0  # replicate corner stress, non-square shape
    one_case("conv-exchange-corner", corner, H.gaussian_kernel(radius=1, sigma=1.0))
    noisy = np.clip(rng.uniform(0, 1, (16, 16)), 0, 1)
    one_case("conv-exchange-noise", noisy, k5)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
