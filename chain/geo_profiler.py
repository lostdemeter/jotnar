"""Geometric code profiler: one tool unifying the hand-rolled speed-round
profiling (trace-`sec` attribution, census maps, /tmp/profrep workdirs).

Two halves, no emitter changes (BACKLOG item 1 shape):
- Static (no execution): census maps + advisor pass-counts -> predicted
  passes/bytes per op. Passes = non-literal reads + writes per op (the same
  write+read each eliminated internal edge saves: 2 passes, per fusion.py).
  Bytes are predicted linearly: pred_bytes_op = passes_op * total_bytes /
  total_passes, where total_bytes comes from the traced run's shapes. So the
  static half predicts TRAFFIC, not wall: compute varies by op class (a
  stencil pass costs more than a move pass), stated scope for v1.
- Dynamic (one traced run, after one untraced warmup that absorbs import /
  first-op effects): measured wall per op (trace `sec`), dispatch-vs-compute
  split, bytes per stream (trace in/out summaries: shape x dtype width) ->
  per-op bottleneck verdict (traffic/dispatch/compute-bound).
- `--budget` flag: milliseconds-in, per-op over/under out. Each op's fair
  share is static-weighted: allowance_op = budget * passes_op / total_passes.
  Burn-down lists worst overrun first.

Self-contained verdict rules (no magic constants; floor/roof come from the
run itself, so verdicts are deterministic given the trace):
  floor = min per-op wall (dispatch can be no larger than the fastest op);
  roof  = max per-op bytes/wall (best observed bandwidth in the run).
  dispatch-bound iff wall <= 2*floor AND bw < 0.30*roof (fast because tiny,
  not because a fast mover -- the roof op itself is always traffic-bound);
  traffic-bound  iff bw >= 0.30*roof; else compute-bound.
Threshold 0.30 sits in the measured gap (demo_img 12x16: max compute 0.17x
roof, min traffic 0.46x roof; flagship: SPLAT/SUB/ADD/SQUARE/GAIN/BETA <=
0.17x, SQRT and up >= 0.40x -- 5/5 stable verdicts on both, see BACKLOG).

Bytes per element by trace summary kind: triples 6 (s:i8 + e:i32 + z:u8),
float64 8, float32 4, float16 2, uint8/int8 1, int32 4, int64 8, mask/bool 1;
literals are immediates (0 traffic). Unknown/unrepresentable kinds fail loud
(ProfilerError) -- never silent wrong bytes. CPU-first; GPU legs were not
needed for a served-path profile here (both gated listings run on numpy).

Usage (CLI --budget is required):
  python3 -m chain.geo_profiler --listing programs/holo_flagship.asm \\
      --payload flagship:24x32 --budget 33
  python3 -m chain.geo_profiler --payload demo:12x16 --budget 33
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..")))

from chain import asm as ASM
from chain import census as CENSUS
from chain import fusion as FUSION

BAND = 0.40            # self-consistency bar: bytes-weighted MAPE <= 40%
TRAFFIC_FRAC = 0.30    # bw >= 0.30 * roof -> traffic-bound (measured gap)
DISPATCH_MULT = 2.0    # wall <= 2 * floor (and slow mover) -> dispatch-bound

BYTES_PER_ELEM = {
    "triples": 6, "float64": 8, "float32": 4, "float16": 2,
    "uint8": 1, "int8": 1, "int32": 4, "int64": 8,
    "mask": 1, "bool": 1, "literal": 0,
}


class ProfilerError(Exception):
    """Profiler misuse or unanalyzable trace (never silent wrong bytes)."""


def stream_bytes(s):
    """Traffic bytes of one trace summary. Literals are immediates (0)."""
    if s.get("kind") == "literal" or s.get("shape") is None:
        return 0
    kind = s.get("kind")
    if kind not in BYTES_PER_ELEM:
        raise ProfilerError(f"unanalyzable stream kind '{kind}' -- "
                            f"cannot price bytes (summary {s})")
    n = 1
    for d in s["shape"]:
        n *= d
    return n * BYTES_PER_ELEM[kind]


def op_bytes(rec):
    """Measured traffic bytes of one trace record (in + out streams)."""
    return (sum(stream_bytes(v) for v in rec["in"].values())
            + sum(stream_bytes(v) for v in rec["out"].values()))


def _short_line(ln):
    """Display form of an origin string ('path:12 [via ...]' -> 'file:12')."""

    def one(s):
        s = s.strip()
        if ":" in s:
            p, r = s.rsplit(":", 1)
            return os.path.basename(p) + ":" + r
        return s
    parts = str(ln).split(" via ")
    return " via ".join(one(p) for p in parts)


def _verdict(wall_s, nbytes, floor_s, roof_bps):
    """Pure bottleneck verdict (deterministic; synthetic-testable)."""
    bw = (nbytes / wall_s) if wall_s > 0 else 0.0
    if wall_s <= DISPATCH_MULT * floor_s and bw < TRAFFIC_FRAC * roof_bps:
        return "dispatch", bw
    if bw >= TRAFFIC_FRAC * roof_bps:
        return "traffic", bw
    return "compute", bw


def profile_text(text, registry, payload, sigs=None, basedir=".",
                 budget_ms=None):
    """Full geometric profile. Returns a plain-dict report (see module
    docstring). Static half never executes; dynamic half is one traced run
    after one untraced warmup."""
    static = CENSUS.census_text(text, registry, sigs=sigs, basedir=basedir)
    advised = FUSION.advise(text, registry, sigs=sigs, basedir=basedir)
    fused = set()
    for ci, ch in enumerate(advised["chains"]):
        for op, ln in zip(ch["ops"], ch["lines"]):
            fused.add((op, ln, ci))
    fused_lines = {(op, ln) for op, ln, _ in fused}

    ASM.run_text(text, registry, payload, sigs=sigs, basedir=basedir)
    trace = []
    ASM.run_text(text, registry, payload, sigs=sigs, trace=trace,
                 basedir=basedir)
    if len(trace) != len(static["ops"]):
        raise ProfilerError(
            f"static/dynamic op count split: census {len(static['ops'])} "
            f"vs trace {len(trace)}")
    ops = []
    for o, rec in zip(static["ops"], trace):
        if (o["op"], o["line"]) != (rec["op"], rec["line"]):
            raise ProfilerError(
                f"static/dynamic order split: census "
                f"{(o['op'], o['line'])} vs trace {(rec['op'], rec['line'])}")
        passes = len(o["reads"]) + len(o["writes"])
        ops.append({"op": o["op"], "line": _short_line(o["line"]),
                    "class": o["class"], "reads": list(o["reads"]),
                    "writes": list(o["writes"]), "passes": passes,
                    "fused": (o["op"], o["line"]) in fused_lines,
                    "wall_s": float(rec["sec"]),
                    "bytes": int(op_bytes(rec))})
    total_passes = sum(o["passes"] for o in ops)
    total_bytes = sum(o["bytes"] for o in ops)
    total_wall = sum(o["wall_s"] for o in ops)
    if total_passes == 0:
        raise ProfilerError("zero static passes -- cannot apportion")
    rate = total_bytes / total_passes
    wmape = (sum(o["bytes"] * abs(o["bytes"] - o["passes"] * rate)
                 / o["bytes"] for o in ops if o["bytes"] > 0)
             / total_bytes) if total_bytes > 0 else 0.0
    floor_s = min(o["wall_s"] for o in ops)
    roof_bps = max((o["bytes"] / o["wall_s"] if o["wall_s"] > 0 else 0.0)
                   for o in ops)
    for o in ops:
        v, bw = _verdict(o["wall_s"], o["bytes"], floor_s, roof_bps)
        o["verdict"] = v
        o["bw_bps"] = float(bw)
        o["pred_bytes"] = float(o["passes"] * rate)
    budget = None
    if budget_ms is not None:
        rows = []
        for o in ops:
            allow = budget_ms / 1000.0 * o["passes"] / total_passes
            rows.append({"op": o["op"], "line": o["line"],
                         "allow_ms": allow * 1000.0,
                         "wall_ms": o["wall_s"] * 1000.0,
                         "over_ms": (o["wall_s"] - allow) * 1000.0,
                         "verdict": o["verdict"]})
        rows.sort(key=lambda r: -r["over_ms"])
        budget = {"budget_ms": float(budget_ms), "rows": rows,
                  "total_over_ms": total_wall * 1000.0 - float(budget_ms)}
    return {"ops": ops,
            "totals": {"n_ops": len(ops), "total_passes": total_passes,
                       "passes_saved": advised["summary"]["passes_saved"],
                       "n_chains": advised["summary"]["n_chains"],
                       "classes": static["classes"],
                       "total_bytes": total_bytes,
                       "total_wall_s": total_wall,
                       "rate_B_per_pass": rate,
                       "floor_s": floor_s, "roof_bps": roof_bps},
            "consistency": {"wmape": wmape, "band": BAND,
                            "ok": wmape <= BAND},
            "budget": budget}


def canonical(rep):
    """Determinism view: everything except absolute wall-clock leaves
    (wall_s, bw_bps, and wall-derived budget rows). Verdicts are kept --
    they must be stable on the gated listings (pasted evidence)."""
    out = {"ops": [], "totals": dict(rep["totals"]),
           "consistency": dict(rep["consistency"]), "budget": None}
    out["totals"].pop("total_wall_s", None)
    out["totals"].pop("floor_s", None)
    out["totals"].pop("roof_bps", None)
    for o in rep["ops"]:
        out["ops"].append({k: v for k, v in o.items()
                           if k not in ("wall_s", "bw_bps")})
    if rep["budget"] is not None:
        out["budget"] = {"budget_ms": rep["budget"]["budget_ms"],
                         "rows": [{"op": r["op"], "line": r["line"],
                                   "verdict": r["verdict"]}
                                  for r in rep["budget"]["rows"]]}
    return out


def format_report(rep, name="listing"):
    """Human-readable profile table (CLI + BACKLOG evidence paste)."""
    L = [f"geo-profiler: {name} ({rep['totals']['n_ops']} ops)"]
    L.append("op           line                  class       P  predB     "
             "B  wall_ms  bw_MBps verdict fused")
    for o in rep["ops"]:
        L.append(f"{o['op']:<12} {o['line']:<22}{o['class']:<11} "
                 f"{o['passes']:>2} {o['pred_bytes']:>6.0f} "
                 f"{o['bytes']:>6d} {o['wall_s'] * 1000:>8.3f} "
                 f"{o['bw_bps'] / 1e6:>8.1f} {o['verdict']:<8} "
                 f"{'Y' if o['fused'] else '-'}")
    t = rep["totals"]
    L.append(f"totals: passes={t['total_passes']} saved={t['passes_saved']} "
             f"({t['n_chains']} chains) bytes={t['total_bytes']} "
             f"wall={t['total_wall_s'] * 1000:.3f}ms "
             f"rate={t['rate_B_per_pass']:.0f}B/pass")
    c = rep["consistency"]
    L.append(f"consistency: WMAPE={c['wmape']:.3f} band={c['band']:.2f} "
             f"{'OK' if c['ok'] else 'FAIL'}")
    if rep["budget"] is not None:
        b = rep["budget"]
        L.append(f"budget {b['budget_ms']:.3f}ms burn-down "
                 f"(total over {b['total_over_ms']:+.3f}ms):")
        L.append("op           line                  allow_ms wall_ms  "
                 "over_ms verdict")
        for r in b["rows"]:
            L.append(f"{r['op']:<12} {r['line']:<22}{r['allow_ms']:>8.3f} "
                     f"{r['wall_ms']:>7.3f} {r['over_ms']:+8.3f} "
                     f"{r['verdict']}")
    return L


# -- CLI payloads (deterministic synthetics; same structure the served
# paths execute, small shapes for seconds) ------------------------------
def flagship_payload(H, W):
    import numpy as np
    return (np.arange(H * W * 3).reshape(H, W, 3) % 256).astype(np.uint8)


def demo_payload(H, W):
    import numpy as np
    from demo_img import build_img_prog, const_vals, tenc
    base = (np.arange(H * W).reshape(H, W) % 256).astype(np.int64)
    pay = {"r": tenc(base), "g": tenc(base + 1), "b": tenc(base + 2)}
    for k, v in const_vals().items():
        sh = (H - 2, W - 2) if k in ("ninth", "amount") else (H, W)
        pay[k] = tenc(np.full(sh, v))
    return build_img_prog(H, W).text(), pay


def main(argv=None):
    ap = argparse.ArgumentParser(description="geometric code profiler")
    ap.add_argument("--listing", default=None,
                    help="listing file (needs --payload for feeds)")
    ap.add_argument("--payload", default=None,
                    help="flagship:HxW | demo:HxW | demo built-in when "
                         "--listing is omitted")
    ap.add_argument("--budget", type=float, required=True,
                    help="frame budget in ms (required): per-op over/under")
    ns = ap.parse_args(argv)
    text, pay, name = None, None, None
    if ns.listing is not None:
        if ns.payload is None or ":" not in ns.payload:
            ap.error("--listing needs --payload flagship:HxW|demo:HxW")
        kind, shape = ns.payload.split(":")
        H, W = (int(x) for x in shape.split("x"))
        text = open(ns.listing).read()
        name = ns.listing
        pay = (flagship_payload(H, W) if kind == "flagship"
               else demo_payload(H, W)[1])
    elif ns.payload is not None:
        kind, shape = ns.payload.split(":")
        H, W = (int(x) for x in shape.split("x"))
        if kind == "flagship":
            ap.error("--payload flagship needs --listing "
                     "(no checked-in flagship at that shape)")
        text, pay = demo_payload(H, W)
        name = f"demo_img {H}x{W}"
    else:
        ap.error("give --listing or --payload")
    from chain.asm_ops import REGISTRY, SIGS
    basedir = os.path.dirname(os.path.abspath(ns.listing)) \
        if ns.listing else "programs"
    rep = profile_text(text, REGISTRY, pay, sigs=SIGS, basedir=basedir,
                       budget_ms=ns.budget)
    for ln in format_report(rep, name):
        print(ln)
    if not rep["consistency"]["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()