"""Substrate emission (v1.2 gate 3): run a listing on numpy/C/CUDA.

Whole-program emission per (program, substrate): numpy executes the
listing text; C/CUDA run file-exchange binaries fed by builders using the
SAME numpy calls the int path uses (same values, conversion location
documented in each test file). Substrate is a RUN concern (machine), not
a listing concern -- deliberately NOT a CONFIG key (CONFIG travels with
the program text; listings stay machine-independent).

Cells with no lowering fail loud naming program+substrate (EmissionError)
-- never silent fallback to numpy. substrate.run_log records every
dispatch (binary path + returncode) so matrix gates prove the binary ran
(agreement alone would also match a silent fallback -- green wallpaper).
"""
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))

import phi_core.lattice as S

C_CHAIN = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "c_chain"))

PROGRAMS = ("flagship", "xf_block")
SUBSTRATES = ("numpy", "c", "cuda")

# (program, substrate) -> binary filename in c_chain (None = numpy path).
# Missing entries are unimplemented cells (loud refusal, never fallback).
BINARIES = {
    ("flagship", "c"): "flagship_c_exchange",
    ("flagship", "cuda"): "flagship_cuda_exchange",
}

run_log = []


class EmissionError(Exception):
    pass


def build_flagship_inbin(y, m_acc, m_cov, beta=0.5):
    """in.bin bytes for the flagship C/CUDA drivers (shared format -- one
    builder gates both sides, no format fork). A + pre-encoded kernels +
    pre-encoded scalar planes, all via the SAME builders the int path uses
    (splat.bank, kernel_triples, control levels, encode)."""
    from chain import holo_phi as H
    from chain import splat as SP
    from chain import control as C
    P = C.load_ctrl()
    Hh, Ww = y.shape
    a_t = H.sqrt_trip(S.encode(np.ascontiguousarray(y, dtype=np.float64)))
    thr = P["coh_thr"]
    tq = int(S.to_fixed(*S.encode(np.array([thr])), m_cov)[0])
    qs1 = S.encode(np.array([1.0]))
    qhi1 = int(S.to_fixed(qs1[0], qs1[1], qs1[2], m_cov)[0])
    att = P.get("iso_atten", 0.5)
    md = P.get("mid_atten", 0.6)
    st = P.get("strong_atten", 1.0)
    hi = P.get("coh_hi", 0.4)

    def const(v):
        return S.encode(np.full((Hh, Ww), float(v), dtype=np.float64))
    consts = [const(beta), const(att), const(md - att), const(st - md),
              const(thr), const(hi), const(30.0),
              const(P.get("v5_w0", 0.0)), const(P.get("v5_w1", 0.0)),
              const(P.get("v5_w2", 0.0)), const(4.0), const(0.5),
              (np.zeros((Hh, Ww), np.int8), np.zeros((Hh, Ww), np.int32),
               np.ones((Hh, Ww), np.uint8))]
    kernels = [H.kernel_triples(SP.SOBEL_X), H.kernel_triples(SP.SOBEL_Y),
               H.kernel_triples(SP.tensor_smooth_kernel())]
    kernels += [H.kernel_triples(k) for k in SP.bank()]
    buf = struct.pack("<4i", Hh, Ww, m_acc, m_cov)
    buf += struct.pack("<q", tq)
    buf += struct.pack("<q", qhi1)

    def flat(t):
        return (np.ascontiguousarray(t[0].reshape(-1), np.int8).tobytes()
                + np.ascontiguousarray(t[1].reshape(-1), np.int32).tobytes()
                + np.ascontiguousarray(t[2].reshape(-1), np.uint8).tobytes())
    for t in [a_t] + kernels + consts:
        buf += flat(t)
    return buf


def read_yenh_trips(path, shape):
    n = int(np.prod(shape))
    with open(path, "rb") as f:
        buf = f.read()
    s = np.frombuffer(buf[0:n], np.int8).reshape(shape).copy()
    e = np.frombuffer(buf[n:n + 4 * n], np.int32).reshape(shape).copy()
    z = np.frombuffer(buf[n + 4 * n:n + 5 * n], np.uint8).reshape(shape).copy()
    return s, e, z


def run(program, payload, substrate="numpy", m_acc=None, m_cov=None,
        beta=0.5, basedir="."):
    """Run one matrix cell. flagship payload: Y float HxW (linear luma).
    Returns YENH triples. Logs every dispatch; unknown cells raise."""
    if program not in PROGRAMS:
        raise EmissionError(f"unknown program: {program} (known: {list(PROGRAMS)})")
    if substrate not in SUBSTRATES:
        raise EmissionError(
            f"unknown substrate: {substrate} (known: {list(SUBSTRATES)})")
    if m_acc is None or m_cov is None:
        from chain.holo_phi import _load_scales
        ma, mc = _load_scales()
        m_acc = ma if m_acc is None else m_acc
        m_cov = mc if m_cov is None else m_cov
    if substrate == "numpy":
        from chain import holo_phi as H
        if program != "flagship":
            raise EmissionError("numpy cell only serves flagship here; "
                                "xf_block runs through its own listing runner")
        yenh_t, _ = H.enhance_luminance_int(
            np.ascontiguousarray(payload, dtype=np.float64), beta=beta,
            m_acc=m_acc, m_cov=m_cov, blur="splat_soft", ctrl="v5")
        run_log.append({"program": program, "substrate": "numpy",
                        "binary": None, "returncode": 0})
        return yenh_t
    key = (program, substrate)
    if key not in BINARIES:
        raise EmissionError(
            f"no {substrate} lowering for {program} (backlog, not fallback)")
    binary = os.path.join("/tmp", "holo_" + BINARIES[key])
    if not os.path.isfile(binary):
        build = "make -C c_chain check" if substrate == "c" else "make -C c_chain cuda"
        raise EmissionError(
            f"no {substrate} binary at {binary} (build: {build})")
    pin, pout = "/tmp/holo_emit_in.bin", "/tmp/holo_emit_out.bin"
    with open(pin, "wb") as f:
        f.write(build_flagship_inbin(payload, m_acc, m_cov, beta))
    r = subprocess.run([binary, pin, pout], capture_output=True, text=True)
    run_log.append({"program": program, "substrate": substrate,
                    "binary": binary, "returncode": r.returncode})
    if r.returncode != 0:
        raise EmissionError(f"{binary} exit={r.returncode}: {r.stderr[-500:]}")
    return read_yenh_trips(pout, np.shape(payload)[:2])
