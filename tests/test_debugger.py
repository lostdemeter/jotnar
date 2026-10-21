"""Listing-debugger gate (docs/DEBUGGER.md constitution): CPU-only, seconds.

Legs (flagship 24x32 synthetic rgb, ~1s total):
- breakpoint/watch/step correctness: stepped values byte-identical
  to direct run() feeds at every stop (hook-routed execution ==
  legacy execution, verified not trusted).
- lesion-table integrity: every op lesionable (11 rows), deltas
  finite, zero-op control reads exactly zero.
- diff: self-diff identical; perturbed-payload diff reports the
  first divergence (op 0, the IN read).
- determinism: same session twice -> identical lesion table +
  stop log (byte-digests).
- fail-loud: bad breakpoint index / watch stream / lesion op /
  step-0 refuse with DebuggerError.
Usage: python3 tests/test_debugger.py
"""
import os
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


def sess():
    from chain import debugger as D
    from chain.asm_ops import REGISTRY, SIGS
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    H, W = 24, 32
    rgb = (np.arange(H * W * 3).reshape(H, W, 3) % 256).astype(np.uint8)
    return D.Session(text, REGISTRY, rgb, sigs=SIGS,
                     basedir=os.path.join(ROOT, "programs")), rgb


def main():
    from chain import asm as ASM
    from chain import debugger as D
    from chain.asm_ops import REGISTRY, SIGS

    s, rgb = sess()
    check("dbg-shape", s.n_ops == 11 and s.sinks == ["OUT"],
          f"{s.n_ops} ops, sinks {s.sinks}")

    # -- step/watch vs direct run (byte-identical at every stop) ------------
    text = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    ref = ASM.run_text(text, REGISTRY, rgb, sigs=SIGS,
                       basedir=os.path.join(ROOT, "programs"))
    s.watch(["OUT", "A", "D", "LIN"])
    s.break_at([3, 7])
    sources = {n for n, st in s.streams.items() if st["source"]}
    stops, ok = [], True
    while s.pos < s.n_ops:
        rec = s.step(100)
        stops.append(rec["pos"])
        produced = set(sources)
        for i in range(rec["pos"]):
            produced.update(s.prog[i][0])
        for w, v in rec["watched"].items():
            if w in produced:
                if v is D.MISSING or D.digest(v) != D.digest(ref[w]):
                    ok = False
            elif v is not D.MISSING:
                ok = False
    check("dbg-breaks", stops == [3, 7, 11], f"stops {stops}")
    check("dbg-step-values", ok, "watched == direct run at every stop")

    # -- single-step full walk ------------------------------------------------
    s.reset()
    n = 0
    while s.pos < s.n_ops:
        s.step(1)
        n += 1
    check("dbg-walk", n == 11 and s.pos == 11, f"{n} single steps")

    # -- lesion table integrity -----------------------------------------------
    rows = s.lesion_table()
    check("dbg-lesion-n", len(rows) == 11
          and [r["op_idx"] for r in rows] == list(range(11)),
          "every op lesionable, program order")
    import math
    check("dbg-lesion-finite",
          all(math.isfinite(r["worst"]) and r["worst"] > 0
              and set(r["deltas"]) == {"OUT"} for r in rows),
          "finite positive OUT deltas per op")
    check("dbg-lesion-order",
          rows[0]["worst"] >= rows[4]["worst"],
          f"early {rows[0]['worst']:.1f} >= late-detail {rows[4]['worst']:.1f}")
    c = s.control()
    check("dbg-control",
          all(v["maxabs"] == 0.0 and v["mean"] == 0.0
              for v in c["deltas"].values()),
          "zero-op control reads exactly zero")

    # -- diff ------------------------------------------------------------------
    d = s.diff(sess()[0])
    check("dbg-self-diff", d == {"first": None}, "identical sessions")
    rgb2 = rgb.copy()
    rgb2[0, 0, 0] ^= 1
    text2 = open(os.path.join(ROOT, "programs", "holo_flagship.asm")).read()
    other = D.Session(text2, REGISTRY, rgb2, sigs=SIGS,
                      basedir=os.path.join(ROOT, "programs"))
    d2 = s.diff(other)
    check("dbg-first-divergence",
          d2.get("first") == 0 and d2.get("stream") == "LIN",
          f"perturbed IN diverges at op 0 ({d2})")

    # -- determinism: session twice -> identical tables -------------------------
    a, _ = sess()
    a.watch(["OUT"])
    b, _ = sess()
    b.watch(["OUT"])
    check("dbg-determinism",
          a.lesion_table() == b.lesion_table()
          and a.stop_log() == b.stop_log(),
          "lesion table + stop log bit-identical")

    # -- fail-loud legs -----------------------------------------------------------
    for fn, exc in ((lambda: s.break_at([99]), "break"),
                    (lambda: s.break_at([-1]), "break-neg"),
                    (lambda: s.watch(["NOPE"]), "watch"),
                    (lambda: s.watch(["CONFIG"]), "watch-config"),
                    (lambda: s.lesion(11), "lesion"),
                    (lambda: s.lesion(-1), "lesion-neg"),
                    (lambda: s.step(0), "step-0")):
        try:
            fn()
            check(f"dbg-loud-{exc}", False, "must raise")
        except D.DebuggerError:
            check(f"dbg-loud-{exc}", True, "DebuggerError")

    print("RESULT:", "ALL OK" if not FAIL else f"FAILURES: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
