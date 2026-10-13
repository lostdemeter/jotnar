"""Protocol loopback gate (Milestone 1): spec'd messages, bit-exact wire.

In-process client+server through the real wire bytes (u32BE length +
JSON header + payload -- no object smuggling): MATMUL+ADD+SILU chain
executes bit-exactly vs direct run_text; T/F/I codecs roundtrip;
malformed frames, unknown types/opcodes/handles all fail loud with
stable error codes; the SERVER section stays model-blind (grep).
Usage: python3 tests/test_protocol.py (CPU only, seconds).
"""
import os
import re
import shutil
import struct
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

FAIL = []


def check(tag, cond, extra=""):
    print(f"{tag}: {'OK' if cond else 'FAIL'} {extra}", flush=True)
    if not cond:
        FAIL.append(tag)


CHAIN = """CONFIG m_acc 31344
CONFIG m_cov 31344
IN a AS T:SEQ
IN b AS T:SEQ
IN c AS T:SEQ
MID = MATMUL(a, b)
SUM = ADD(MID, c)
OUT = SILU(SUM)
"""


def norm(v):
    return np.stack(v) if isinstance(v, tuple) else np.asarray(v)


def _read_frame(sock):
    """Read one framed reply from a raw socket. Raises ConnectionError
    on short reads (peer already gone)."""
    import struct as _st
    from protocol.loopback import loads as _loads
    buf = b""
    while len(buf) < 4:
        chunk = sock.recv(4 - len(buf))
        if not chunk:
            raise ConnectionError("peer closed before length")
        buf += chunk
    (n,) = _st.unpack(">I", buf)
    body = b""
    while len(body) < n:
        chunk = sock.recv(n - len(body))
        if not chunk:
            raise ConnectionError("peer closed mid-reply")
        body += chunk
    return _loads(buf + body)


def _await_served_raw(port, timeout=10, first_id=900):
    """Acquire a connection the server actually served (busy-race guard).

    Each attempt opens a probe, PINGs, and closes it unless the reply
    is served -- so no accept can race a preceding close: the returned
    socket was accepted with a free slot by construction, and every
    failed probe is closed before the next opens. Returns (sock,
    header) or (None, None) on bounded-timeout expiry (caller fails
    loud). Polling, never a bare sleep; strictly stronger than
    connect-once.
    """
    import socket as _s
    import time as _t
    from protocol.loopback import dumps as _dumps
    deadline = _t.time() + timeout
    attempt = 0
    while _t.time() < deadline:
        try:
            s = _s.create_connection(("127.0.0.1", port), timeout=5)
            s.settimeout(5)
        except OSError:
            attempt += 1
            continue
        try:
            s.sendall(_dumps("ping", {"id": first_id + attempt}))
            header, _ = _read_frame(s)
        except (ConnectionError, OSError):
            try:
                s.close()
            except OSError:
                pass
            attempt += 1
            continue
        if header.get("ok"):
            return s, header
        try:
            s.close()
        except OSError:
            pass
        attempt += 1
    return None, None


def _await_served_client(port, timeout=10):
    """Same guard for TcpClient users: connect, PING, keep iff served."""
    import time as _t
    from protocol.tcp import TcpClient, TransportError
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        try:
            c = TcpClient("127.0.0.1", port, timeout=30)
        except (OSError, TransportError):
            continue
        try:
            h, _ = c.ping()
        except (OSError, TransportError):
            try:
                c.close()
            except OSError:
                pass
            continue
        if h.get("ok"):
            return c
        try:
            c.close()
        except OSError:
            pass
    return None


def main():
    from protocol.loopback import Server, Client, dumps, loads, FrameError
    from chain import asm as ASM
    from chain.asm_ops import REGISTRY, SIGS
    import phi_core.lattice as S
    rng = np.random.default_rng(3)
    A = S.encode(np.ascontiguousarray(rng.normal(size=(2, 4)) * .5))
    B = S.encode(np.ascontiguousarray(rng.normal(size=(4, 3)) * .5))
    C = S.encode(np.ascontiguousarray(rng.normal(size=(2, 3)) * .5))
    ref = ASM.run_text(CHAIN, REGISTRY, {"a": A, "b": B, "c": C}, sigs=SIGS)

    sv, cl = Server(), None
    sv = Server()
    cl = Client(sv)

    # -- ping + program -------------------------------------------------
    h, _ = cl.ping()
    check("ping-ok", h.get("ok") and h["pong"]["backends"] == ["lattice", "served"],
          h.get("pong"))
    h, _ = cl.program("m1", CHAIN)
    check("program-ok", h.get("ok") and h.get("handle") == "m1", h)
    h, _ = cl.program("m1", CHAIN)
    check("program-replace", h.get("ok") and h.get("replaced") is True, h)

    # -- exec bit-exact vs direct ----------------------------------------
    h, out = cl.exec("m1", {"a": A, "b": B, "c": C}, ["MID", "SUM", "OUT"])
    check("exec-ok", h.get("ok"), h.get("error"))
    exact = h.get("ok") and all(
        np.array_equal(norm(out[k]), norm(ref[k])) for k in ["MID", "SUM", "OUT"])
    check("exec-exact", exact, "wire == run_text bit-exact")

    # -- determinism: same bytes twice -----------------------------------
    _, out2 = cl.exec("m1", {"a": A, "b": B, "c": C}, ["OUT"])
    check("exec-deterministic",
          np.array_equal(norm(out["OUT"]), norm(out2["OUT"])),
          "bit-exact re-run")

    # -- store/list/drop --------------------------------------------------
    h, _ = cl.store("w", np.ascontiguousarray(rng.normal(size=(3, 3))))
    check("store-ok", h.get("ok") and h.get("bytes") == 72, h)
    h, _ = cl.list()
    check("list-ok", h.get("ok") and "m1" in h["programs"] and "w" in h["stores"], h)
    h, _ = cl.drop("w")
    check("drop-ok", h.get("ok"), h)
    h, _ = cl.drop("nope")
    check("drop-missing-ok", h.get("ok") and h.get("missing") is True, h)

    # -- fail loud: opcodes/handles/stores --------------------------------
    h, _ = cl.program("bad", "OUT = BOGUS(a)")
    check("program-npattern", (not h.get("ok")) and h["error"]["code"] == "npattern",
          h.get("error"))
    h, _ = cl.exec("ghost", {"a": A}, ["OUT"])
    check("exec-unknown-handle",
          (not h.get("ok")) and h["error"]["code"] == "unknown-handle", h.get("error"))
    rh, _ = loads(sv.handle(dumps("exec", {"id": 99, "handle": "m1",
                                           "inputs": {"a": {"store": "ghost"}},
                                           "outputs": ["OUT"]})))
    check("exec-unknown-store",
          (not rh.get("ok")) and rh["error"]["code"] == "unknown-store", rh.get("error"))

    # -- fail loud: malformed frames ---------------------------------------
    for tag, frame in [
            ("frame-short", b"\x00\x00"),
            ("frame-length", dumps("ping", {"id": 1})[:-3]),
            ("frame-json", struct.pack(">I", 5) + b"nope!")]:
        try:
            loads(frame)
            check(tag, False, "accepted")
        except FrameError as e:
            check(tag, True, str(e)[:60])
    rh, _ = loads(sv.handle(dumps("frobnicate", {"id": 2})))
    check("unknown-type", (not rh.get("ok")) and rh["error"]["code"] == "unknown-type",
          rh.get("error"))

    # -- server stays model-blind ------------------------------------------
    src = open(os.path.join(ROOT, "protocol", "loopback.py")).read()
    lines = src.split("===== SERVER", 1)[1].split("===== END SERVER", 1)[0].splitlines()
    body = "\n".join(l for l in lines[1:] if "model-stack" not in l)
    hits = [w for w in ("qwen", "tokenizer", "transformers", "huggingface")
            if re.search(w, body, re.IGNORECASE)]
    check("server-blind", not hits, f"forbidden refs: {hits}")
    for fn in ("protocol/tcp.py", "protocol/serve_tcp.py"):
        tsrc = open(os.path.join(ROOT, fn)).read()
        thits = [w for w in ("qwen", "tokenizer", "transformers", "huggingface")
                 if re.search(w, tsrc, re.IGNORECASE)]
        check(f"blind-{fn}", not thits, f"forbidden refs: {thits}")

    # -- TCP: three-way agreement (direct == loopback == sockets) ---------
    import threading
    from protocol.tcp import TcpServer, TcpClient, TransportError
    from protocol.loopback import dumps as _dumps
    tsv = TcpServer(Server(), port=0, timeout=30)
    th = threading.Thread(target=tsv.serve_forever, daemon=True)
    th.start()
    try:
        tcl = TcpClient("127.0.0.1", tsv.port, timeout=30)
        h, _ = tcl.program("m1", CHAIN)
        check("tcp-program", h.get("ok"), h.get("error"))
        h, tout = tcl.exec("m1", {"a": A, "b": B, "c": C}, ["OUT"])
        triple = (np.array_equal(norm(tout["OUT"]), norm(ref["OUT"])) and
                  np.array_equal(norm(tout["OUT"]), norm(out["OUT"])))
        check("tcp-three-way", h.get("ok") and triple,
              "tcp == direct == loopback, bit-exact")
        _, tout2 = tcl.exec("m1", {"a": A, "b": B, "c": C}, ["OUT"])
        check("tcp-deterministic",
              np.array_equal(norm(tout["OUT"]), norm(tout2["OUT"])),
              "bit-exact re-run over sockets")
        tcl.close()

        # -- second connection refused loud --------------------------------
        # Busy-race guard: the server frees tcl's slot asynchronously, so
        # acquire a provably-served socket (failed probes are closed
        # first); it becomes `a` with no accept/close gap by construction.
        import socket as _s
        a, ah = _await_served_raw(tsv.port, timeout=10, first_id=501)
        check("tcp-ping-a", a is not None and ah.get("ok"), ah)
        if a is None:
            print("RESULT:", f"FAILURES: {FAIL}")
            sys.exit(1)
        b = _s.create_connection(("127.0.0.1", tsv.port), timeout=5)
        b.settimeout(5)
        from protocol.loopback import loads as _loads
        b.sendall(_dumps("ping", {"id": 502}))
        buf = b.recv(4)
        import struct as _st
        (n,) = _st.unpack(">I", buf)
        rbody = b""
        while len(rbody) < n:
            rbody += b.recv(n - len(rbody))
        bh, _ = _loads(buf + rbody)
        check("tcp-busy-loud", (not bh.get("ok")) and
              bh.get("error", {}).get("code") == "busy", bh)
        a.close()
        b.close()

        # -- stamp provenance refused --------------------------------------
        # Same race after a.close()/b.close(): acquire a served client.
        tcl2 = _await_served_client(tsv.port, timeout=10)
        check("stamp-quiescent", tcl2 is not None, "slot never freed")
        if tcl2 is None:
            print("RESULT:", f"FAILURES: {FAIL}")
            sys.exit(1)
        h, _ = tcl2.program("srv", CHAIN, backend="served",
                            config={"exe": "/nonexistent/qdec"})
        check("stamp-missing-refused",
              (not h.get("ok")) and h["error"]["code"] == "stamp-mismatch",
              h.get("error"))
        h, _ = tcl2.program("srv", CHAIN, backend="served",
                            config={"exe": "/nonexistent/qdec"},
                            stamp={"smax": 16})
        check("stamp-exe-missing-refused", not h.get("ok") and
              h["error"]["code"] == "stamp-mismatch", h.get("error"))
        tcl2.close()

        # -- blackholed peer fails loud, no hang ----------------------------
        import time as _t
        t0 = _t.time()
        try:
            TcpClient("127.0.0.1", 1, timeout=2)
            check("tcp-blackhole", False, "connected to nothing")
        except TransportError as e:
            check("tcp-blackhole", _t.time() - t0 < 10, f"{e}"[:80])
    finally:
        tsv.stop()
        tsv.close()

    # -- M3: heterogeneous dispatch (CPU: two lattice nodes) --------------
    import threading as _th
    from protocol.loopback import Server as _Server
    from protocol.pipeline import PipeController, RouteError
    cpu = TcpServer(_Server(allowed=["lattice"]), port=0, timeout=30)
    gpu = TcpServer(_Server(allowed=["lattice", "served"]), port=0,
                    timeout=30)
    thc, thg = (_th.Thread(target=s.serve_forever, daemon=True)
                for s in (cpu, gpu))
    thc.start()
    thg.start()
    try:
        from protocol.tcp import TcpClient as _TcpClient
        ctl = PipeController()
        c_caps = ctl.add_node("cpu", _TcpClient("127.0.0.1", cpu.port,
                                                timeout=30))
        g_caps = ctl.add_node("gpu", _TcpClient("127.0.0.1", gpu.port,
                                                timeout=30))
        check("hetero-advertise", c_caps == ["lattice"] and
              g_caps == ["lattice", "served"], f"{c_caps} {g_caps}")
        h = ctl.program_auto("ichain", CHAIN, backend="lattice")
        check("hetero-route-lattice", ctl.placement("ichain") == ("cpu", "lattice"),
              ctl.log)
        h, hout = ctl.exec("ichain", {"a": A, "b": B, "c": C}, ["OUT"])
        check("hetero-exec-lattice",
              h.get("ok") and np.array_equal(norm(hout["OUT"]), norm(ref["OUT"])),
              h.get("error"))
        try:
            ctl.program("cpu", "nope", CHAIN, backend="served")
            check("hetero-node-refuses-served", False, "accepted")
        except RouteError as e:
            check("hetero-node-refuses-served", True, str(e)[:70])
        try:
            ctl.program_auto("nope2", CHAIN, backend="served")
            check("hetero-served-to-gpu", False, "accepted without stamp")
        except RouteError as e:
            # routed to the gpu node (not cpu), refused there for stamp:
            # placement proven by where it failed.
            check("hetero-served-to-gpu", "stamp" in str(e), str(e)[:90])
        # integer node never saw anything but lattice: its LIST proves it.
        h, _ = ctl.nodes["cpu"].list()
        check("hetero-cpu-clean", h.get("programs") == ["ichain"], h)
        try:
            ctl2 = PipeController()
            ctl2.program_auto("x", CHAIN, backend="served")
            check("hetero-no-served-loud", False, "routed nowhere")
        except RouteError as e:
            check("hetero-no-served-loud", True, str(e)[:70])
        # dead stage fails fast (no GPU needed for the refusal itself).
        try:
            _TcpClient("127.0.0.1", 1, timeout=2)
            check("hetero-dead-stage", False, "connected to nothing")
        except TransportError:
            check("hetero-dead-stage", True, "loud, no hang")
    finally:
        cpu.stop()
        cpu.close()
        gpu.stop()
        gpu.close()

    # -- M3: 2-way pipelined Paris (GPU; SKIP without snapshot/nvcc) ------
    # S=16 keeps builds ~80s each: 2 stage binaries + 1 full reference.
    # Stages (0-13/14-27) ferry f32 H; caches stay sticky per stage.
    _pipe_ok = True
    try:
        from chain.qwen7b import load7b as _load7b, snapshot_ok as _sok
        from chain.qwen7b import input_sources as _isrc, clear_7b_cache as _clr
    except ImportError as e:
        check("pipe-skip", True, f"no chain ({e})")
        _pipe_ok = False
    if _pipe_ok and not _sok():
        print("SKIP pipe-paris (needs snapshot)")
    elif _pipe_ok and shutil.which("nvcc") is None:
        print("SKIP pipe-paris (needs nvcc + GPU)")
    elif _pipe_ok:
        import subprocess as _sp
        S, NL = 16, 28
        HID, KVD, PER = 3584, 512, 7
        SPLITS = [(0, 13), (14, 27)]
        base = "/tmp/protogate"
        g7, tok7 = _load7b()
        E7 = np.ascontiguousarray(
            g7("model.embed_tokens.weight"), dtype=np.float16)
        _clr(g7)
        import gc as _gc
        _gc.collect()
        from protocol.pipe_build import build_pipeline, build_stage
        cfgs = build_pipeline(g7, dict(_isrc()), S, SPLITS, base)
        # full single-node reference, same float mode (bit-exact peer:
        # same listings, same f32 files, one binary instead of two)
        fcfg = build_stage(g7, dict(_isrc()), S, 0, 27, True,
                           os.path.join(base, "full"))
        check("pipe-full-built", os.path.isfile(fcfg["exe"]), fcfg["exe"])
        if os.path.isfile(fcfg["exe"]):
            n0 = TcpServer(_Server(), port=0, timeout=600)
            n1 = TcpServer(_Server(), port=0, timeout=600)
            t0 = _th.Thread(target=n0.serve_forever, daemon=True)
            t1 = _th.Thread(target=n1.serve_forever, daemon=True)
            t0.start()
            t1.start()
            try:
                ctl = PipeController()
                ctl.add_node("s0", _TcpClient("127.0.0.1", n0.port, timeout=600))
                ctl.add_node("s1", _TcpClient("127.0.0.1", n1.port, timeout=600))
                # register stages (stamp-checked provenance)
                regs = []
                for i, cfg in enumerate(cfgs):
                    node = f"s{i}"
                    argv = cfg["argv"]
                    nlay = cfg["hi"] - cfg["lo"] + 1
                    n_in = len(argv) - (2 + 2 * nlay if cfg["last"]
                                        else 1 + 2 * nlay)
                    if cfg["last"]:
                        outs = {"logits": {"idx": n_in, "dtype": "f32",
                                           "shape": [1, 152064]},
                                "choice": {"idx": n_in + 1, "dtype": "i64",
                                           "shape": [1]}}
                    else:
                        outs = {"H": {"idx": n_in, "dtype": "f32",
                                      "shape": [1, HID]}}
                    for L in range(cfg["lo"], cfg["hi"] + 1):
                        nlay = cfg["hi"] - cfg["lo"] + 1
                        base_j = n_in + (2 if cfg["last"] else 1)
                        # BLOCKED outputs (all-K then all-V), not interleaved
                        outs[f"knew{L}"] = {"idx": base_j + (L - cfg["lo"]),
                                            "dtype": "f32",
                                            "shape": [S * KVD]}
                        outs[f"vnew{L}"] = {"idx": base_j + nlay + (L - cfg["lo"]),
                                            "dtype": "f32",
                                            "shape": [S * KVD]}
                    h, _ = ctl.nodes[node].program(
                        f"st{i}", cfg["text"], backend="served",
                        config={"exe": cfg["exe"], "argv_files": argv,
                                "live": {"step_x0": 0, "step_pos": 1,
                                         "step_am": 2},
                                "outputs": outs,
                                "err_path": os.path.join(cfg["work"], "pipe.err")},
                        stamp=dict(cfg["stamp"]))
                    regs.append(h.get("ok"))
                    ctl.routes[f"st{i}"] = (node, "served")
                    ctl.log.append((f"st{i}", node, "served"))
                check("pipe-registered", all(regs), regs)
                if all(regs):
                    ids = tok7("The capital of France is",
                               return_tensors="pt")["input_ids"][0].tolist()
                    onames0 = (["H"] + [f"knew{L}" for L in range(0, 14)] +
                               [f"vnew{L}" for L in range(0, 14)])
                    onames1 = (["logits", "choice"] +
                               [f"knew{L}" for L in range(14, 28)] +
                               [f"vnew{L}" for L in range(14, 28)])

                    def relay(out, name, dst, expect):
                        # checked relay: wire payload must carry exactly
                        # `expect` elements; a short/empty payload must
                        # fail here, never truncate a cache file silently
                        # (a killed mid-write run once poisoned a workdir).
                        a = np.asarray(out[name], dtype=np.float32)
                        assert a.size == expect, (name, a.shape, expect)
                        a.tofile(dst)
                        assert os.path.getsize(dst) == expect * 4, dst

                    def pstep(tok_id, n):
                        xx = np.zeros((1, HID), dtype=np.float32)
                        xx[0] = E7[np.array([tok_id])].astype(np.float32)
                        am = np.zeros((PER, S), dtype=np.int64)
                        am[:, :n + 1] = 1
                        h0, o0 = ctl.exec(
                            "st0", {"step_x0": xx,
                                    "step_pos": np.array([n], dtype=np.int64),
                                    "step_am": am}, onames0)
                        assert h0.get("ok"), h0.get("error")
                        H = np.asarray(o0["H"], dtype=np.float32)
                        for L in range(0, 14):
                            relay(o0, f"knew{L}",
                                  os.path.join(base, "stage0_0-13",
                                               f"ckP{L}.bin"), S * KVD)
                            relay(o0, f"vnew{L}",
                                  os.path.join(base, "stage0_0-13",
                                               f"cvP{L}.bin"), S * KVD)
                        h1, o1 = ctl.exec(
                            "st1", {"step_x0": H,
                                    "step_pos": np.array([n], dtype=np.int64),
                                    "step_am": am}, onames1)
                        assert h1.get("ok"), h1.get("error")
                        for L in range(14, 28):
                            relay(o1, f"knew{L}",
                                  os.path.join(base, "stage1_14-27",
                                               f"ckP{L}.bin"), S * KVD)
                            relay(o1, f"vnew{L}",
                                  os.path.join(base, "stage1_14-27",
                                               f"cvP{L}.bin"), S * KVD)
                        return (np.asarray(o1["logits"]).reshape(-1),
                                int(np.asarray(o1["choice"]).reshape(-1)[0]),
                                {L: np.asarray(o1[f"knew{L}"]).tobytes()
                                 if L >= 14 else
                                 np.asarray(o0[f"knew{L}"]).tobytes()
                                 for L in range(28)})

                    for i, t in enumerate(ids):
                        pstep(t, i)
                    seq, gen = list(ids), []
                    plg, plc, pkn = None, None, None
                    for _ in range(2):
                        plg, plc, pkn = pstep(seq[-1], len(seq))
                        seq.append(plc)
                        gen.append(plc)
                    # Release stage binaries BEFORE the single starts:
                    # stages hold ~28GB managed weights; the float
                    # single holds ~28GB more; together they OOM this
                    # box (measured: OOM-killed at 26GB file-rss).
                    # Bytes are compared after, not simultaneously.
                    n0.stop()
                    n0.close()
                    n1.stop()
                    n1.close()
                    # DROP kills the stage binaries server-side (stopping
                    # the TCP front does NOT reap spawned qdec procs).
                    # Without this they keep ~28GB managed weights and
                    # the single OOMs at startup (measured, twice).
                    for _node, _prog in (("s0", "st0"), ("s1", "st1")):
                        _h, _ = ctl.nodes[_node].drop(_prog)
                        assert _h.get("ok"), _h.get("error")
                    # single-node reference: pipe-built float single (same
                    # mode as the stages, so bit-exactness is meaningful).
                    # demo_paris argv pattern: 4 live + 56 caches + 196
                    # weights + 2, then BLOCKED K0..K27 + V0..V27 outputs.
                    fargv, fn_in = fcfg["argv"], None
                    _nlay = 28
                    fn_in = len(fargv) - (2 + 2 * _nlay)
                    fout = {"logits": {"idx": fn_in, "dtype": "f32",
                                       "shape": [1, 152064]},
                            "choice": {"idx": fn_in + 1, "dtype": "i64",
                                       "shape": [1]}}
                    for _L in range(28):
                        fout[f"knew{_L}"] = {"idx": fn_in + 2 + _L,
                                             "dtype": "f32", "shape": [S * KVD]}
                        fout[f"vnew{_L}"] = {"idx": fn_in + 2 + _nlay + _L,
                                             "dtype": "f32", "shape": [S * KVD]}
                    from chain.serve import ServedExe as _SE
                    fwork = os.path.join(base, "full")
                    NL = 28
                    flives = fcfg["lives"]
                    fflo = os.path.join(fwork, "step_lg.bin")
                    ffou = os.path.join(fwork, "step_ou.bin")
                    full = _SE(fcfg["exe"], fargv,
                               err_path=os.path.join(fwork, "g.err"))
                    full.start()

                    def fstep(tok_id, n):
                        xx = np.zeros((1, HID), dtype=np.float32)
                        xx[0] = E7[np.array([tok_id])].astype(np.float32)
                        xx.tofile(flives["x0"])
                        np.array([n], dtype=np.int64).tofile(flives["pos"])
                        am = np.zeros((PER, S), dtype=np.int64)
                        am[:, :n + 1] = 1
                        am.tofile(flives["attmask7"])
                        full.step()
                        lg = np.fromfile(fflo, dtype=np.float32).reshape(-1)
                        ou = int(np.fromfile(ffou, dtype=np.int64)[0])
                        kn = {}
                        for L in range(NL):
                            a = os.path.join(fwork, f"step_K{L}.bin")
                            kn[L] = open(a, "rb").read()
                            assert len(kn[L]) == S * KVD * 4, (L, len(kn[L]))
                            np.fromfile(a, dtype=np.float32).tofile(
                                os.path.join(fwork, f"ckP{L}.bin"))
                            b = os.path.join(fwork, f"step_V{L}.bin")
                            assert os.path.getsize(b) == S * KVD * 4, (L, b)
                            np.fromfile(b, dtype=np.float32).tofile(
                                os.path.join(fwork, f"cvP{L}.bin"))
                        return lg, ou, kn

                    for i, t in enumerate(ids):
                        fstep(t, i)
                    fseq, fgen = list(ids), []
                    flg, fou, fkn = None, None, None
                    for _ in range(2):
                        flg, fou, fkn = fstep(fseq[-1], len(fseq))
                        fseq.append(fou)
                        fgen.append(fou)
                    full.close()
                    klay = all(pkn[L] == fkn[L] for L in range(28))
                    check("pipe-kv-exact", klay,
                          "per-layer knew bytes pipeline == single")
                    check("pipe-logits-exact",
                          np.array_equal(plg, flg), "final LOGITS identical")
                    check("pipe-choice-exact", plc == fou,
                          f"pipe {plc} vs single {fou}")
                    ptext = tok7.decode(gen, skip_special_tokens=True)
                    check("pipe-paris", "Paris" in ptext, repr(ptext))
                    stext = tok7.decode(fgen, skip_special_tokens=True)
                    check("pipe-text-match", ptext == stext,
                          f"{ptext!r} vs {stext!r}")
            finally:
                n0.stop()
                n0.close()
                n1.stop()
                n1.close()

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
