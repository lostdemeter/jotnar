"""Loopback client+server for the Jotnar wire protocol (v1, Milestone 1).

In-process pair speaking the exact wire bytes from docs/PROTOCOL.md
(u32BE length + JSON header + raw payload) through memory queues --
no sockets yet (Milestone 2). The codec is shared; the sections below
are split so the gate can prove the server half stays model-blind.

Client = builder output plus framing (smart side). Server = existing
chain execution behind a dispatch table (dumb side): lattice run_text
or a ServedExe step over an out-of-band-built binary. Neither side
compiles in v1 (compile is controller-side emitter work).
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

PROTO_V = 1


class FrameError(ValueError):
    """Malformed frame (length/JSON/payload mismatch)."""


def dumps(msg_type, fields, payload=b""):
    """Encode one frame: u32BE len + JSON header + payload bytes."""
    header = {"v": PROTO_V, "type": msg_type}
    header.update(fields)
    if "id" not in header:
        raise FrameError("message needs an id")
    h = json.dumps(header, sort_keys=True).encode("utf-8")
    body = h + b"\n" + bytes(payload)
    return struct.pack(">I", len(body)) + body


def loads(frame):
    """Decode one frame -> (header dict, payload bytes). Raises FrameError."""
    if len(frame) < 4:
        raise FrameError(f"frame too short for length: {len(frame)}")
    (n,) = struct.unpack(">I", frame[:4])
    body = frame[4:]
    if len(body) != n:
        raise FrameError(f"length prefix {n} != body {len(body)}")
    try:
        h, payload = body.split(b"\n", 1)
        header = json.loads(h.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise FrameError(f"bad header: {e}")
    if not isinstance(header, dict) or header.get("v") != PROTO_V:
        raise FrameError(f"bad version/kind: {header!r}"[:120])
    if "id" not in header or "type" not in header:
        raise FrameError("header needs id + type")
    return header, payload


# ===== SERVER (model-blind half: controller-only imports forbidden below) =====
# Gate tests/test_protocol.py extracts this section and rejects any
# model-stack reference. Keep it that way.

def _t_decode(stream, payload):
    kind, dtype, shape = stream["kind"], stream.get("dtype", ""), tuple(stream["shape"])
    if kind == "F":
        dt = {"f32": np.float32, "f64": np.float64,
              "f16": np.float16}.get(dtype)
        if dt is None:
            raise ValueError(f"unknown float dtype {dtype}")
        return np.frombuffer(payload, dtype=dt).reshape(shape).copy()
    if kind == "I":
        return np.frombuffer(payload, dtype=np.int64).reshape(shape).copy()
    if kind == "T":
        n = 1
        for d in shape:
            n *= d
        w = np.frombuffer(payload, dtype=np.int64)
        if w.size != 3 * n:
            raise ValueError(f"T payload {w.size} != 3*{n}")
        s, e, z = w[:n].reshape(shape), w[n:2 * n].reshape(shape), w[2 * n:].reshape(shape)
        return (s.astype(np.int8), e.astype(np.int32), z.astype(np.uint8))
    raise ValueError(f"unknown kind {kind}")


def _t_encode(value, shape=None):
    if isinstance(value, (tuple, list)) and len(value) == 3:
        arrs = [np.asarray(a) for a in value]
        shape = list(arrs[0].shape)
        s, e, z = (a.astype(np.int64).ravel() for a in arrs)
        return {"kind": "T", "dtype": "i64x3", "shape": shape}, \
            np.concatenate([s, e, z]).tobytes()
    a = np.ascontiguousarray(value)
    if a.dtype == np.float16:
        return {"kind": "F", "dtype": "f16", "shape": list(a.shape)}, a.tobytes()
    a = np.ascontiguousarray(value)
    if a.dtype == np.float32:
        return {"kind": "F", "dtype": "f32", "shape": list(a.shape)}, a.tobytes()
    if a.dtype == np.float64:
        return {"kind": "F", "dtype": "f64", "shape": list(a.shape)}, a.tobytes()
    if a.dtype == np.int64 or np.issubdtype(a.dtype, np.integer):
        b = np.ascontiguousarray(a, dtype=np.int64)
        return {"kind": "I", "dtype": "i64", "shape": list(b.shape)}, b.tobytes()
    raise ValueError(f"unencodable output dtype {a.dtype}")


class Server:
    """Dumb model-blind node: program registry + blob shelf + dispatch."""

    def __init__(self):
        self.programs = {}  # name -> {listing, backend, config, exe?}
        self.stores = {}    # name -> {stream, payload}

    # -- transport entry -------------------------------------------------
    def handle(self, frame):
        try:
            header, payload = loads(frame)
        except FrameError as e:
            return dumps("error", {"id": 0, "ok": False,
                                   "error": {"code": "malformed", "msg": str(e)}})
        hid, typ = header.get("id"), header.get("type")
        fn = {"ping": self._ping, "program": self._program,
              "store": self._store, "drop": self._drop,
              "list": self._list, "exec": self._exec}.get(typ)
        if fn is None:
            return self._err(hid, "unknown-type", f"no such message {typ!r}")
        try:
            return fn(hid, header, payload)
        except ValueError as e:  # fail loud, never silent wrong bytes
            return self._err(hid, "node", str(e))

    def _err(self, hid, code, msg):
        return dumps("error", {"id": hid, "ok": False,
                               "error": {"code": code, "msg": msg}})

    def _ok(self, hid, typ, fields, payload=b""):
        out = {"id": hid, "ok": True}
        out.update(fields)
        return dumps(typ, out, payload)

    # -- dispatch ---------------------------------------------------------
    def _ping(self, hid, header, payload):
        from chain.asm_ops import REGISTRY
        return self._ok(hid, "pong", {"pong": {
            "impl": "jotnar-node", "v": PROTO_V,
            "backends": ["lattice", "served"],
            "mnemonics": len(REGISTRY)}})

    def _program(self, hid, header, payload):
        from chain import asm as ASM
        from chain.asm_ops import REGISTRY, SIGS
        name, listing = header["name"], payload.decode("utf-8")
        backend = header.get("backend", "lattice")
        if backend not in ("lattice", "served"):
            return self._err(hid, "unknown-type", f"backend {backend!r}")
        try:  # fail fast: unknown opcodes refused at PROGRAM time
            ASM.assemble(listing, REGISTRY, SIGS)
        except Exception as e:
            return self._err(hid, "npattern", f"listing refused: {e}")
        replaced = name in self.programs
        old = self.programs.get(name)
        if old and old.get("exe") is not None:
            try:
                old["exe"].close()
            except Exception:
                pass
        cfg = header.get("config", {})
        if backend == "served":
            # Build provenance, not convention: the client states the
            # stamp it wants; the binary's build.json next to the exe
            # states what was built. Anything else refuses loud.
            want = header.get("stamp")
            if not isinstance(want, dict):
                return self._err(hid, "stamp-mismatch",
                                 "served PROGRAM needs a stamp")
            import json as _js
            try:
                have = _js.load(open(os.path.join(
                    os.path.dirname(cfg.get("exe", "")), "build.json")))
                have = have.get("stamp", have)  # pipe builds nest it
            except OSError as e:
                return self._err(hid, "stamp-mismatch",
                                 f"no build.json next to exe: {e}")
            bad = [k for k in want if have.get(k) != want[k]]
            if bad:
                return self._err(hid, "stamp-mismatch",
                                 f"binary stamp differs on {bad}")
        self.programs[name] = {"listing": listing, "backend": backend,
                               "config": cfg, "exe": None}
        return self._ok(hid, "program-ok",
                        {"handle": name, "replaced": replaced})

    def _store(self, hid, header, payload):
        name = header["name"]
        replaced = name in self.stores
        self.stores[name] = {"stream": header["stream"], "payload": bytes(payload)}
        return self._ok(hid, "store-ok", {"name": name, "replaced": replaced,
                                          "bytes": len(payload)})

    def _drop(self, hid, header, payload):
        name = header.get("name", "")
        if name in self.stores:
            del self.stores[name]
            return self._ok(hid, "drop-ok", {"name": name})
        if name in self.programs:
            prog = self.programs.pop(name)
            if prog.get("exe") is not None:
                try:
                    prog["exe"].close()
                except Exception:
                    pass
            return self._ok(hid, "drop-ok", {"name": name})
        return self._ok(hid, "drop-ok", {"name": name, "missing": True})

    def _list(self, hid, header, payload):
        return self._ok(hid, "list-ok",
                        {"programs": sorted(self.programs),
                         "stores": sorted(self.stores)})

    def _exec(self, hid, header, payload):
        name = header["handle"]
        prog = self.programs.get(name)
        if prog is None:
            return self._err(hid, "unknown-handle", f"no program {name!r}")
        try:
            inputs = {}
            for stream, ref in header["inputs"].items():
                if "store" in ref:
                    blob = self.stores.get(ref["store"])
                    if blob is None:
                        return self._err(hid, "unknown-store",
                                         f"no store {ref['store']!r}")
                    inputs[stream] = _t_decode(blob["stream"], blob["payload"])
                else:
                    off, ln = int(ref["off"]), int(ref["len"])
                    seg = payload[off:off + ln]
                    if len(seg) != ln:
                        return self._err(hid, "malformed",
                                         f"inline segment {stream} truncated")
                    st = {"kind": ref["kind"], "dtype": ref.get("dtype", ""),
                          "shape": ref["shape"]}
                    inputs[stream] = _t_decode(st, seg)
            if prog["backend"] == "lattice":
                outs = self._exec_lattice(prog, inputs, header["outputs"])
            else:
                outs = self._exec_served(prog, inputs, header["outputs"])
        except KeyError as e:
            return self._err(hid, "arity", f"missing binding {e}")
        except ValueError as e:
            return self._err(hid, "arity", str(e))
        body, offs, meta = b"", {}, {}
        for stream, (st, chunk) in outs.items():
            meta[stream] = dict(st, offset=len(body))
            body += chunk
            offs[stream] = True
        return self._ok(hid, "exec-ok", {"outputs": meta}, body)

    def _exec_lattice(self, prog, inputs, wanted):
        from chain import asm as ASM
        from chain.asm_ops import REGISTRY, SIGS
        try:  # fail safe: re-check at EXEC time, not just PROGRAM time
            ASM.assemble(prog["listing"], REGISTRY, SIGS)
        except Exception as e:
            raise ValueError(f"npattern: {e}")
        try:
            feeds = ASM.run_text(prog["listing"], REGISTRY, dict(inputs), sigs=SIGS)
        except KeyError as e:
            raise KeyError(str(e))
        outs = {}
        for stream in wanted:
            if stream not in feeds:
                raise KeyError(stream)
            st, chunk = _t_encode(feeds[stream])
            outs[stream] = (st, chunk)
        return outs

    def _exec_served(self, prog, inputs, wanted):
        from chain.serve import ServedExe
        cfg = prog["config"]
        exe = prog.get("exe")
        if exe is None:
            exe = ServedExe(cfg["exe"], list(cfg["argv_files"]),
                            err_path=cfg.get("err_path"))
            exe.start()
            prog["exe"] = exe
        live = cfg.get("live", {})
        for stream, value in inputs.items():
            if stream not in live:
                raise KeyError(stream)
            a = np.ascontiguousarray(value)
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            a.tofile(cfg["argv_files"][live[stream]])
        exe.step()
        outs = {}
        for stream in wanted:
            meta = cfg["outputs"][stream]  # {idx, dtype, shape}
            raw = open(cfg["argv_files"][meta["idx"]], "rb").read()
            dt = {"f16": np.float16, "f32": np.float32,
                  "i64": np.int64}[meta["dtype"]]
            arr = np.frombuffer(raw, dtype=dt).reshape(meta["shape"]).copy()
            st, chunk = _t_encode(arr)
            outs[stream] = (st, chunk)
        return outs

# ===== END SERVER =====


class Client:
    """Smart side: framing helpers over a transport.

    transport: frame bytes -> response frame bytes. Defaults to the
    in-process server handle (loopback); TCP passes a socket sender.
    """

    def __init__(self, server=None, transport=None):
        if transport is None:
            transport = server.handle
        self.transport = transport
        self._id = 0

    def _roundtrip(self, typ, fields, payload=b""):
        self._id += 1
        fields = dict(fields, id=self._id)
        rh, rp = loads(self.transport(dumps(typ, fields, payload)))
        return rh, rp

    def ping(self):
        return self._roundtrip("ping", {})

    def program(self, name, listing, backend="lattice", config=None,
                stamp=None):
        fields = {"name": name, "backend": backend, "config": config or {}}
        if stamp is not None:
            fields["stamp"] = stamp
        return self._roundtrip("program", fields, listing.encode("utf-8"))

    def store(self, name, array):
        st, chunk = _t_encode(array)
        return self._roundtrip("store", {"name": name, "stream": st}, chunk)

    def drop(self, name):
        return self._roundtrip("drop", {"name": name})

    def list(self):
        return self._roundtrip("list", {})

    def exec(self, handle, inputs, outputs):
        """inputs: {stream: array} inline (encode here). Returns
        (header, {stream: decoded array})."""
        wire = {}
        inline = {}
        for stream, value in inputs.items():
            st, chunk = _t_encode(value)
            wire[stream] = dict(st, off=0, len=0)
            inline[stream] = chunk
        # inline payloads ride as segments of the frame payload, in
        # binding order; each ref carries its off/len (v1, no alignment).
        order = list(wire)
        body, off = b"", 0
        for s in order:
            wire[s]["off"], wire[s]["len"] = off, len(inline[s])
            body += inline[s]
            off += len(inline[s])
        rh, rp = self._roundtrip("exec", {"handle": handle, "inputs": wire,
                                          "outputs": outputs}, body)
        if not rh.get("ok"):
            return rh, {}
        outs = {}
        for stream, meta in rh["outputs"].items():
            n = 1
            for d in meta["shape"]:
                n *= d
            seg = rp[meta["offset"]:]
            if meta["kind"] == "F":
                if meta["dtype"] == "f16":
                    outs[stream] = np.frombuffer(
                        seg, dtype=np.float16,
                        count=n).reshape(meta["shape"]).astype(np.float32).copy()
                    continue
                dt = np.float32 if meta["dtype"] == "f32" else np.float64
                outs[stream] = np.frombuffer(seg, dtype=dt,
                                             count=n).reshape(meta["shape"]).copy()
            elif meta["kind"] == "I":
                outs[stream] = np.frombuffer(seg, dtype=np.int64,
                                             count=n).reshape(meta["shape"]).copy()
            else:
                w = np.frombuffer(seg, dtype=np.int64, count=3 * n)
                sh = tuple(meta["shape"])
                outs[stream] = (w[:n].reshape(sh).copy(),
                                w[n:2 * n].reshape(sh).copy(),
                                w[2 * n:].reshape(sh).copy())
        return rh, outs
