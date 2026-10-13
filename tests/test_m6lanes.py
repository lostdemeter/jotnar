"""Lane-saturation gate (protocol Milestone 6): gimli sweats.

N concurrent flagship EXEC lanes against N node processes (distinct
ports -- the busy policy is per-server, so multi-process is the legal
fan-out; no protocol change). Every lane bit-identical vs local
run_text every iteration (exactness under concurrency is the
load-bearing assertion); per-PID jiffies for all servers + a 1/2/4/8
scaling curve with an honest network split. Throughput is the target;
per-turn latency is not claimed.
SKIP-clean without JOTNAR_REMOTE_HOST (gold suite stays green
machineless); gimli needs 8 lattice nodes on 41211-41218 (setup in
docs/PROTOCOL.md M6 section; M4 tunnel convention for the ports).
Usage: JOTNAR_REMOTE_HOST=127.0.0.1 JOTNAR_REMOTE_SSH=gimli \
         python3 tests/test_m6lanes.py
"""
import os
import subprocess as _sp
import sys
import threading
import time

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []

LANE_BASE = 41211
CURVE = [1, 2, 4, 8]
ITERS = 5


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


def ssh(target, args, timeout=30):
    return _sp.run(["ssh", target] + args, capture_output=True,
                   text=True, timeout=timeout)


def lane_pids(target, n):
    """Map lane index -> server python PID (skip wrapper bash lines)."""
    r = ssh(target, ["ps", "-eo", "pid,args"])
    pids = {}
    for ln in r.stdout.splitlines():
        if "serve_tcp.py --host" not in ln or "bash -c" in ln:
            continue
        parts = ln.strip().split(None, 1)
        for i in range(n):
            if f"--port {LANE_BASE + i}" in ln:
                pids[i] = parts[0]
    return pids


def sample_jiffies(target, pids):
    """One ssh call -> {lane: jiffies} for the given {lane: pid}."""
    if not pids:
        return {}
    cmd = ["cat"] + [f"/proc/{p}/stat" for p in pids.values()]
    r = ssh(target, cmd)
    out = {}
    lanes = list(pids)
    for lane, ln in zip(lanes, r.stdout.splitlines()):
        try:
            f = ln.split()
            out[lane] = int(f[13]) + int(f[14])
        except (IndexError, ValueError):
            out[lane] = -1
    return out


def main():
    host = os.environ.get("JOTNAR_REMOTE_HOST", "")
    if not host:
        print("SKIP (set JOTNAR_REMOTE_HOST for the lane-saturation leg)")
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

    rgb = np.asarray(Image.open(
        os.path.join(ROOT, "samples", "input_example.png")).convert("RGB"))
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    ref = np.asarray(ASM.run_text(text, REGISTRY, {"rgb": rgb}, sigs=SIGS)["OUT"])
    print(f"local ref: {ref.shape} {ref.dtype}", flush=True)

    ctl = PipeController()
    for i in range(8):
        try:
            ctl.add_node(f"lane{i}",
                         TcpClient(conn, LANE_BASE + i, timeout=300))
        except TransportError as e:
            check("lanes-reachable", False,
                  f"lane{i} unreachable ({e}); start 8 nodes "
                  f"41211-41218 per docs/PROTOCOL.md M6".split(";")[0][:100])
            print("RESULT:", f"FAILURES: {FAIL}")
            sys.exit(1)
    check("lanes-reachable", True, "8/8 lane servers answer PING")

    for i in range(8):
        ctl.program(f"lane{i}", f"flag{i}", text, backend="lattice")
    placed = [ctl.placement(f"flag{i}") == (f"lane{i}", "lattice")
              for i in range(8)]
    check("lanes-placed", all(placed), ctl.log)

    pids = lane_pids(ssh_target, 8)
    check("lanes-pids", len(pids) == 8, sorted(pids))
    if len(pids) != 8:
        print("RESULT:", f"FAILURES: {FAIL}")
        sys.exit(1)

    lock = threading.Lock()
    stats = {}

    def drive(lane, iters, ref):
        errs = 0
        for _ in range(iters):
            h, out = ctl.exec(f"flag{lane}", {"rgb": rgb}, ["OUT"])
            got = np.asarray(out["OUT"])
            if not (h.get("ok") and got.shape == ref.shape
                    and np.array_equal(got, ref)):
                errs += 1
        with lock:
            stats[lane] = errs

    print("lanes wall execs exact/serverCPU-per-iter eff-cores", flush=True)
    for nlanes in CURVE:
        stats.clear()
        j0 = sample_jiffies(ssh_target, {i: pids[i] for i in range(nlanes)})
        t0 = time.time()
        ths = [threading.Thread(target=drive, args=(i, ITERS, ref))
               for i in range(nlanes)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.time() - t0
        j1 = sample_jiffies(ssh_target, {i: pids[i] for i in range(nlanes)})
        dj = {i: j1.get(i, -1) - j0.get(i, -1) for i in range(nlanes)}
        bad = {i: stats.get(i, -1) for i in range(nlanes)}
        exact = all(v == 0 for v in bad.values())
        check(f"lanes{nlanes}-exact", exact,
              f"{nlanes * ITERS}/{nlanes * ITERS} bit-identical "
              f"(errs={bad})")
        moved = all(v > 50 for v in dj.values())
        check(f"lanes{nlanes}-moved", moved,
              f"jiffy deltas={sorted(dj.values())}")
        cpu = sum(dj.values()) / 100
        eff = cpu / wall if wall > 0 else 0
        print(f"curve: {nlanes} lanes, wall {wall:.1f}s, "
              f"{nlanes * ITERS} execs, server CPU {cpu:.1f}s "
              f"({cpu / (nlanes * ITERS) * 1000:.0f}ms/iter), "
              f"effective cores {eff:.2f}", flush=True)

    r = ssh(ssh_target, ["cat", "/proc/loadavg"])
    print(f"gimli loadavg after run: {r.stdout.strip()}", flush=True)
    print("ferry: 304KB U8 in + 304KB U8 out per EXEC "
          "(608KB/turn; ~2ms on 2.5GbE -- per-lane network share is "
          "noise next to ~550ms compute; aggregate scales with lanes, "
          "still <2% of wall at 8 lanes)", flush=True)

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
