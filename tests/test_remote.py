"""Remote-node gate (protocol Milestone 4): gimli is the calculator.

Drives a lattice-only TcpServer on a second machine over the real
network with the unmodified PipeController (TcpClient already takes
a host -- the node runs unchanged; that no-change fact is the proof).
Remote host comes from JOTNAR_REMOTE_HOST (default gimli); the gate
SKIPs cleanly when it is unset, so the gold suite stays green with
no second machine. Port 41201 (out-of-band node setup, see
docs/PROTOCOL.md M4 section).
Usage: JOTNAR_REMOTE_HOST=gimli python3 tests/test_remote.py
  (CPU only both ends, seconds; gimli needs ~/jotnar-m4 payload +
  a lattice node on 41201 -- see spec for setup).
"""
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

REMOTE_PORT = 41201

CHAIN = """CONFIG m_acc 31344
CONFIG m_cov 31344
IN a AS T:SEQ
IN b AS T:SEQ
IN c AS T:SEQ
MID = MATMUL(a, b)
SUM = ADD(MID, c)
OUT = SILU(SUM)
"""


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def norm(v):
    return np.stack(v) if isinstance(v, tuple) else np.asarray(v)


def resolve(host):
    """gimli is an SSH alias, not DNS: resolve via ssh config when the
    name does not resolve directly. Returns (display, connect_host)."""
    import socket as _s
    try:
        _s.gethostbyname(host)
        return host, host
    except OSError:
        pass
    import subprocess as _sp
    r = _sp.run(["ssh", "-G", host], capture_output=True, text=True,
                timeout=15)
    for ln in r.stdout.splitlines():
        if ln.startswith("hostname "):
            return host, ln.split(None, 1)[1]
    return host, host


def main():
    host = os.environ.get("JOTNAR_REMOTE_HOST", "")
    if not host:
        print("SKIP (set JOTNAR_REMOTE_HOST=gimli to run the remote leg)")
        return
    from protocol.tcp import TcpClient, TransportError
    from protocol.pipeline import PipeController, RouteError
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S

    print(f"remote: {host}:{REMOTE_PORT} (set JOTNAR_REMOTE_HOST to change host)",
          flush=True)
    disp, conn = resolve(host)
    if conn != disp:
        print(f"remote: ssh alias {disp} -> {conn}", flush=True)
    try:
        ctl = PipeController()
        caps = ctl.add_node("gimli", TcpClient(conn, REMOTE_PORT, timeout=60))
    except TransportError as e:
        check("remote-reachable", False, f"{e}"[:120])
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    # -- PING advertisement across the real network -----------------------
    check("remote-advertise", caps == ["lattice"], caps)

    # -- lattice EXEC bit-exact vs local run_text --------------------------
    rng = np.random.default_rng(3)
    A = S.encode(np.ascontiguousarray(rng.normal(size=(2, 4)) * .5))
    B = S.encode(np.ascontiguousarray(rng.normal(size=(4, 3)) * .5))
    C = S.encode(np.ascontiguousarray(rng.normal(size=(2, 3)) * .5))
    ref = ASM.run_text(CHAIN, REGISTRY, {"a": A, "b": B, "c": C}, sigs=SIGS)
    ctl.program_auto("ichain", CHAIN, backend="lattice")
    check("remote-placed-lattice", ctl.placement("ichain") == ("gimli", "lattice"),
          ctl.log)
    h, out = ctl.exec("ichain", {"a": A, "b": B, "c": C}, ["OUT"])
    check("remote-exec-exact",
          h.get("ok") and np.array_equal(norm(out["OUT"]), norm(ref["OUT"])),
          h.get("error"))
    _, out2 = ctl.exec("ichain", {"a": A, "b": B, "c": C}, ["OUT"])
    check("remote-deterministic",
          np.array_equal(norm(out["OUT"]), norm(out2["OUT"])),
          "bit-exact re-run over the network")

    # -- served PROGRAM refused loud (CPU-only node) ------------------------
    try:
        ctl.program("gimli", "nope", CHAIN, backend="served")
        check("remote-served-refused", False, "accepted")
    except RouteError as e:
        check("remote-served-refused", True, str(e)[:80])
    rh, _ = ctl.nodes["gimli"].program(
        "nope2", CHAIN, backend="served",
        config={"exe": "/nonexistent/qdec"})
    check("remote-served-wire",
          (not rh.get("ok")) and rh["error"]["code"] == "backend-unavailable",
          rh.get("error"))

    # -- dead stage fails loud, no hang -------------------------------------
    t0 = time.time()
    try:
        TcpClient(conn, 1, timeout=5)
        check("remote-dead-stage", False, "connected to nothing")
    except TransportError as e:
        check("remote-dead-stage", time.time() - t0 < 15, f"{e}"[:80])

    # -- Leg 2: GPU stage on gimli SKIPPED BY HARDWARE -----------------------
    # Probe over ssh (JOTNAR_REMOTE_SSH defaults to the test host; set it
    # to the ssh alias when connecting through a local tunnel).
    import subprocess as _sp
    _ssh_target = os.environ.get("JOTNAR_REMOTE_SSH", "") or host
    r = _sp.run(["ssh", _ssh_target, "nvidia-smi"],
                capture_output=True, text=True, timeout=30)
    _out = (r.stdout + r.stderr).strip().splitlines() or ["?"]
    no_gpu = r.returncode != 0 and "couldn't communicate" in (r.stdout + r.stderr)
    check("remote-no-gpu", no_gpu, _out[0][:90])
    print("SKIP leg2-gpu-stage (gimli has no GPU -- documented, not forced)")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
