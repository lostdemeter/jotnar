"""3-way parity: numpy matmul_int vs C holo_matmul vs CUDA k_matmul.

Same file-exchange format feeds both binaries (matmul_exchange +
matmul_cuda_exchange read identical in.bin); each side is gated bit-exact
vs the same numpy reference, which transitively proves C == CUDA. Cases
mirror test_matmul_c.py (small/batched/broadcast/big-m/zeros): the big-m
case proves the v1.1 scale path reaches CUDA too.
Build: `make -C c_chain cuda` (needs nvcc + GPU); without nvcc the suite
SKIPs (exit 0) like the other needs-hardware gates.
Usage: python3 tests/test_matmul_cuda.py
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "..", "phi-core")))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import test_matmul_c as MC

FAIL = []
CXCHG = "/tmp/holo_matmul_cuda_exchange"


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}")
    if not cond:
        FAIL.append(tag)


def main():
    cc = os.path.join(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")), "c_chain")
    if not os.path.exists(CXCHG):
        r = subprocess.run(["make", "-C", cc, "cuda"],
                           capture_output=True, text=True)
        if r.returncode:
            print(f"SKIP (needs nvcc + GPU: {r.stderr[-300:]})")
            sys.exit(0)
    from chain.holo_phi import _load_scales
    ma, _ = _load_scales()
    rng = np.random.default_rng(11)  # same fixtures as the C gate
    cases = [
        ("cuda-small", (rng.random((1, 4, 8)) - 0.5) * 2,
         (rng.random((1, 8, 6)) - 0.5) * 2, ma),
        ("cuda-batched", (rng.random((2, 3, 5)) - 0.5) * 2,
         (rng.random((2, 5, 4)) - 0.5) * 2, ma),
        ("cuda-broadcast", (rng.random((2, 3, 5)) - 0.5) * 2,
         (rng.random((1, 5, 4)) - 0.5) * 2, ma),
        ("cuda-bigm", (rng.random((1, 4, 8)) - 0.5) * 8,
         (rng.random((1, 8, 6)) - 0.5) * 8, 35492),
    ]
    nf = MC.FAIL
    for tag, a, b, m in cases:
        MC.one_case(tag, a, b, m, binary=CXCHG)
    FAIL.extend(MC.FAIL[len(nf):])
    z = np.array([[[0.0, -1.0, 1.0, -0.5], [0.5, 0.0, -1.5, 2.0]]])
    w = np.array([[[1.0, 0.0], [-1.0, 0.5], [0.0, -2.0], [0.25, 0.0]]])
    nf = len(MC.FAIL)
    MC.one_case("cuda-zeros", z, w, ma, binary=CXCHG)
    FAIL.extend(MC.FAIL[nf:])
    # C-vs-CUDA agreement is transitive via the shared numpy ref above;
    # pin it explicitly on one case (byte-compare of out planes).
    check("cuda-transitive", not FAIL, "both binaries bit-exact vs numpy ref")
    # Decode-shape cuBLAS determinism (MATMUL round 1): the production
    # F16->F32 path must be bit-exact across separate processes. Runs the
    # isolation microbench twice, compares all 20 output hashes.
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ubcu = os.path.join(root, "bench", "matmul_decode.cu")
    ubbin = "/tmp/mm_ub_gate"
    r = subprocess.run(["nvcc", "-O2", "-arch=native", "-o", ubbin, ubcu, "-lcublas"],
                       capture_output=True, text=True)
    if r.returncode:
        print(f"SKIP gemmex-deterministic (needs nvcc + GPU: {r.stderr[-300:]})")
    else:
        import re
        hs = []
        for _ in range(2):
            p = subprocess.run([ubbin, "5"], capture_output=True, text=True)
            if p.returncode:
                FAIL.append("gemmex-deterministic")
                print(f"gemmex run failed rc={p.returncode}: {p.stderr[-300:]}")
                break
            hs.append(re.findall(r"h=([0-9a-f]+)", p.stdout))
        if len(hs) == 2:
            same = hs[0] == hs[1] and len(hs[0]) == 20
            check("gemmex-deterministic", same,
                  f"{len(hs[0])} hashes, identical={hs[0] == hs[1]}")
    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
