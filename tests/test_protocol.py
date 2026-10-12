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

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
