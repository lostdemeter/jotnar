"""File-exchange parity: numpy matmul_int vs C holo_matmul, BIT-EXACT.

Not dB -- exact equality on (s,e,z). The C binary reads triples written by
numpy (identical input triples, no float encode divergence), runs the
lowering at the given m_acc, writes triples back. Any mismatch = lowering
damage. Cases: small 2D, batched, B-broadcast, big-m regime (v1.1 scale
path works in C too), zeros+negatives edge (zero-or + sign paths).
Usage: python3 tests/test_matmul_c.py (needs gcc; builds via c_chain Makefile).
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import phi_core.lattice as S
from phi_core import numpy_ops as N
from chain.holo_phi import _load_scales

FAIL = []
XCHG = "/tmp/holo_matmul_exchange"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def write_in(path, a_trip, b_trip, m_acc):
    nb = a_trip[0].shape[0]
    n, k = a_trip[0].shape[1], a_trip[0].shape[2]
    nbb = b_trip[0].shape[0]
    m = b_trip[0].shape[2]
    assert b_trip[0].shape[1] == k and nbb in (1, nb)
    with open(path, "wb") as f:
        f.write(struct.pack("<6i", nb, n, k, m, nbb, m_acc))
        for t in (a_trip, b_trip):
            for plane, dt in ((t[0], np.int8), (t[1], np.int32),
                              (t[2], np.uint8)):
                f.write(np.ascontiguousarray(plane, dt).tobytes())


def read_out(path, shape):
    n = int(np.prod(shape))
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    return s, e, z


def one_case(tag, a_f, b_f, m_acc, binary=XCHG):
    a_t = S.encode(np.ascontiguousarray(a_f, np.float64))
    b_t = S.encode(np.ascontiguousarray(b_f, np.float64))
    ref = N.matmul_int(a_t, b_t, m_acc)
    pin, pout = "/tmp/holo_matmul_in.bin", "/tmp/holo_matmul_out.bin"
    write_in(pin, a_t, b_t, m_acc)
    r = subprocess.run([binary, pin, pout], capture_output=True, text=True)
    if r.returncode != 0:
        check(tag, False, f"binary exit={r.returncode} {r.stderr[-500:]}")
        return
    got = read_out(pout, ref[0].shape)
    same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all()
                and (got[2] == ref[2]).all())
    if not same:
        ns = int((got[0] != ref[0]).sum())
        ne = int((got[1] != ref[1]).sum())
        nz = int((got[2] != ref[2]).sum())
        check(tag, False, f"ds={ns} de={ne} dz={nz}")
    else:
        check(tag, True, f"{a_t[0].shape}x{b_t[0].shape} m_acc={m_acc} bit-exact")


def main():
    if not os.path.exists(XCHG):
        r = subprocess.run(["make", "-C", os.path.join(
            os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "c_chain"), "check"],
            capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-1500:], r.stderr[-1500:])
    ma, _ = _load_scales()
    rng = np.random.default_rng(11)
    one_case("matmul-exchange-small",
             (rng.random((1, 4, 8)) - 0.5) * 2, (rng.random((1, 8, 6)) - 0.5) * 2, ma)
    one_case("matmul-exchange-batched",
             (rng.random((2, 3, 5)) - 0.5) * 2, (rng.random((2, 5, 4)) - 0.5) * 2, ma)
    one_case("matmul-exchange-broadcast",
             (rng.random((2, 3, 5)) - 0.5) * 2, (rng.random((1, 5, 4)) - 0.5) * 2, ma)
    one_case("matmul-exchange-bigm",
             (rng.random((1, 4, 8)) - 0.5) * 8, (rng.random((1, 8, 6)) - 0.5) * 8, 35492)
    z = np.array([[[0.0, -1.0, 1.0, -0.5], [0.5, 0.0, -1.5, 2.0]]])
    w = np.array([[[1.0, 0.0], [-1.0, 0.5], [0.0, -2.0], [0.25, 0.0]]])
    one_case("matmul-exchange-zeros", z, w, ma)
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
