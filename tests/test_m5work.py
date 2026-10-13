"""Remote-work gate (protocol Milestone 5): gimli does real work.

Serves the flagship image pipeline (programs/holo_flagship.asm) on a
CPU-only lattice node over the real network: full-program output
bytes bit-identical vs local run_text every iteration, gimli-side
CPU jiffies for its server PID across the run (the needle must
move), honest wall-vs-compute split (offload is placement, never a
latency claim). SKIP-clean without JOTNAR_REMOTE_HOST (gold suite
stays green machineless); gimli needs the M4 node setup (payload +
lattice node on 41201 -- see docs/PROTOCOL.md M4 section).
Usage: JOTNAR_REMOTE_HOST=gimli python3 tests/test_m5work.py
  (tunnel form: JOTNAR_REMOTE_HOST=127.0.0.1 JOTNAR_REMOTE_SSH=gimli)
"""
import os
import subprocess as _sp
import sys
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

REMOTE_PORT = 41201
ITERS = 10


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


def resolve(host):
    """gimli is an SSH alias, not DNS: resolve via ssh config when the
    name does not resolve directly. Returns (display, connect_host)."""
    import socket as _s
    try:
        _s.gethostbyname(host)
        return host, host
    except OSError:
        pass
    r = _sp.run(["ssh", "-G", host], capture_output=True, text=True,
                timeout=15)
    for ln in r.stdout.splitlines():
        if ln.startswith("hostname "):
            return host, ln.split(None, 1)[1]
    return host, host


def ssh(target, cmd, timeout=30):
    r = _sp.run(["ssh", target] + cmd, capture_output=True, text=True,
                timeout=timeout)
    return r


def server_pid(target):
    """gimli-side python node PID. pgrep -f matches the wrapper bash too
    (its cmdline embeds the python command), so list full command lines
    and skip `bash -c` -- sampling the wrapper reads 0 jiffies forever."""
    r = ssh(target, ["pgrep", "-af", "jotnar-node/bin/python"])
    for ln in r.stdout.splitlines():
        if "bash -c" in ln:
            continue
        pid = ln.strip().split(None, 1)[0]
        if pid.isdigit():
            return pid
    return ""


def jiffies(target, pid):
    r = ssh(target, ["cat", f"/proc/{pid}/stat"])
    try:
        f = r.stdout.split()
        return int(f[13]) + int(f[14])
    except (IndexError, ValueError):
        return -1


def main():
    host = os.environ.get("JOTNAR_REMOTE_HOST", "")
    if not host:
        print("SKIP (set JOTNAR_REMOTE_HOST=gimli to run the remote-work leg)")
        return
    from protocol.tcp import TcpClient, TransportError
    from protocol.pipeline import PipeController
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    from PIL import Image

    disp, conn = resolve(host)
    if conn != disp:
        print(f"remote: ssh alias {disp} -> {conn}", flush=True)
    ssh_target = os.environ.get("JOTNAR_REMOTE_SSH", "") or host
    print(f"remote: {conn}:{REMOTE_PORT} (ssh via {ssh_target})", flush=True)

    rgb = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    ref = np.asarray(ASM.run_text(text, REGISTRY, {"rgb": rgb}, sigs=SIGS)["OUT"])
    print(f"local ref: {ref.shape} {ref.dtype}", flush=True)

    try:
        ctl = PipeController()
        caps = ctl.add_node("gimli", TcpClient(conn, REMOTE_PORT, timeout=300))
    except TransportError as e:
        check("remote-reachable", False, f"{e}"[:120])
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)
    check("remote-advertise", caps == ["lattice"], caps)
    ctl.program_auto("flag", text, backend="lattice")
    check("remote-placed-lattice", ctl.placement("flag") == ("gimli", "lattice"),
          ctl.log)

    pid = server_pid(ssh_target)
    check("remote-pid-found", bool(pid), f"pid={pid!r}")
    if not pid:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)
    j0 = jiffies(ssh_target, pid)
    check("remote-jiffies-read", j0 >= 0, f"j0={j0}")

    walls, exact = [], True
    t_all = time.time()
    for i in range(ITERS):
        t0 = time.time()
        h, out = ctl.exec("flag", {"rgb": rgb}, ["OUT"])
        walls.append(time.time() - t0)
        got = np.asarray(out["OUT"])
        ok = h.get("ok") and got.shape == ref.shape and np.array_equal(got, ref)
        exact = exact and ok
        print(f"iter {i}: ok={h.get('ok')} exact={ok} "
              f"wall={walls[-1]:.2f}s", flush=True)
    wall_all = time.time() - t_all
    j1 = jiffies(ssh_target, pid)
    dj = j1 - j0
    check("remote-exec-exact", exact, f"{ITERS}/{ITERS} bit-identical")
    check("remote-jiffies-moved", dj > 100,
          f"{j0} -> {j1} (+{dj} = {dj / 100:.1f} CPU-s on gimli)")
    compute = dj / 100
    print(f"split: wall {wall_all:.1f}s for {ITERS} iters "
          f"({wall_all / ITERS * 1000:.0f}ms/iter); server CPU {compute:.1f}s "
          f"({compute / ITERS * 1000:.0f}ms/iter); remainder is network + "
          f"framing + local driver (offload is placement, not latency)",
          flush=True)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
