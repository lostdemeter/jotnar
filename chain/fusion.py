"""Fusion advisor (read-only): report fusible elementwise chains, change nothing.

Reads chain/census.py maps (producer/consumer/fan-out) and reports maximal
non-overlapping pointwise chains with per-chain legality + profitability.
Advisory only: no rewrites, no emitter changes, no new opcodes.

Legality (all three must hold per link, else boundary or refusal):
1. single consumer: producer stream fan-out == 1 (exactly this link).
   Fan-out >= 2 blocks (reported in `blocked`); fan-out == 1 also means
   intermediates can't escape as outputs (an interior member is consumed,
   hence never a sink).
2. pointwise class: both ops in FUSIBLE (census elementwise + GAIN, a
   pointwise map with broadcast reads). Reductions (MATMUL epilogue is
   second, not this), stencils, gathers, moves, selects, recur: boundaries.
   Unknown-class ops: unanalyzable + boundary (never guessed).
3. shapes/layouts agree: carried by the program's own runtime contract
   (shape mismatch fails loud at run time, independently gated), which
   pointwise maps preserve structurally; plus an explicit check where
   static layouts are KNOWN (both binary operands declared with layouts
   that differ -> unanalyzable). STATE-touching candidates: unanalyzable
   (order-sensitive, never fused blind).

Chains are paths (greedy longest-first, program-order tiebreak,
non-overlapping on ops, minimum length 2 -- singletons fuse nothing).
Fan-in side inputs stay live (documented path-model limitation: the luma
head reports its longest path, not the full multi-input region; DAG
regions are parked codegen). An explicit `outputs` set (default: sinks)
additionally blocks chains whose interior escapes as an output (served
binaries tap intermediates -- fusing through a tap needs multi-output
kernels, parked).
Profitability: each eliminated internal edge saves one write + one read
= 2 memory passes (stream sizes unknown statically: passes, not bytes).
Deterministic: program-order iteration throughout, no randomness.
"""
import numpy as np

from chain import asm as ASM
from chain import census as CENSUS


def fusible_set(registry=None, sigs=None):
    """Pointwise maps: census elementwise + GAIN (pointwise map with
    broadcast reads -- the fused kernel performs the same reads the
    program already does; broadcast compatibility rides the program's
    runtime shape contract)."""
    base = {op for op, cl in CENSUS.ACCESS.items() if cl == "elementwise"}
    return base | {"GAIN"}


class FusionError(Exception):
    """Advisor misuse (bad outputs set, not a program finding)."""


def advise(text, registry, sigs=None, basedir=".", outputs=None):
    """Static fusion report. Returns {"chains", "unanalyzable", "blocked",
    "summary"}. Never executes, never rewrites."""
    rep = CENSUS.census_text(text, registry, sigs=sigs, basedir=basedir)
    streams, ops = rep["streams"], rep["ops"]
    fusible = fusible_set()
    state = set(rep["state"])
    config, inp, _, _ = ASM.assemble(text, registry, sigs, basedir)
    layouts = {n: lay for n, lay in inp}
    produced = {n for n, s in streams.items() if s["producer"] is not None}

    if outputs is None:
        outputs = {n for n in produced if streams[n]["fanout"] == 0}
    else:
        outputs = set(outputs)

    unanalyzable = []

    def note(op, line, reason):
        unanalyzable.append({"op": op, "line": line, "reason": reason})

    for o in ops:
        if o["class"] == "unknown":
            note(o["op"], o["line"], "no access class -- cannot classify")
        if state and (set(o["reads"]) & state or set(o["writes"]) & state):
            note(o["op"], o["line"], "touches STATE -- order-sensitive")

    # valid links producer-op -> consumer-op (stream-mediated)
    by_op_line = {(o["op"], o["line"]): i for i, o in enumerate(ops)}
    prod_op = {}  # stream -> producing op index
    for i, o in enumerate(ops):
        for w in o["writes"]:
            prod_op[w] = i
    links = {}  # producer idx -> [(consumer idx, stream)]
    for s, st in streams.items():
        if st["producer"] is None or st["fanout"] != 1:
            continue
        cmn, cln, _ = st["consumers"][0]
        pi = prod_op.get(s)
        ci = by_op_line.get((cmn, cln))
        if pi is None or ci is None:
            continue
        po, co = ops[pi], ops[ci]
        if po["op"] not in fusible or co["op"] not in fusible:
            continue
        if state and (s in state):
            continue
        # declared-layout agreement where statically known
        lits = [a for a in co["reads"] if a in layouts and layouts[a]]
        if len({layouts[a] for a in lits}) > 1:
            note(co["op"], co["line"],
                 f"declared layouts disagree on inputs "
                 f"({[(a, layouts[a]) for a in lits]})")
            continue
        links.setdefault(pi, []).append((ci, s))

    # greedy longest-first non-overlapping paths, program-order tiebreak
    used = set()
    chains = []
    heads = [i for i, o in enumerate(ops)
             if o["op"] in fusible
             and not any(i in [c for c, _ in v] for v in links.values())]
    # candidate paths from every head, then longest-first claim
    cands = []
    for h in heads:
        path, cur = [h], h
        while True:
            nxt = [(c, s) for c, s in links.get(cur, [])]
            nxt = [(c, s) for c, s in nxt]
            if not nxt:
                break
            c, s = nxt[0]
            path.append(c)
            cur = c
        if len(path) >= 2:
            cands.append(path)
    cands.sort(key=lambda p: (-len(p), ops[p[0]]["line"]))
    for path in cands:
        if any(i in used for i in path):
            continue
        # interior escape check: non-final members must not be outputs
        seq_streams = []
        ok = True
        for a, b in zip(path, path[1:]):
            s = next(s for c, s in links[a] if c == b)
            seq_streams.append(s)
            if s in outputs:
                ok = False
        if not ok:
            continue
        for i in path:
            used.add(i)
        last = ops[path[-1]]
        out_s = ops[path[-1]]["writes"][-1] if ops[path[-1]]["writes"] else None
        cons = streams.get(out_s, {}).get("consumers", []) if out_s else []
        boundary = "sink" if not cons else (
            f"{cons[0][0]}:{cons[0][1]}")
        chains.append({
            "ops": [ops[i]["op"] for i in path],
            "lines": [ops[i]["line"] for i in path],
            "streams": seq_streams,
            "out": ops[path[-1]]["writes"][-1]
            if ops[path[-1]]["writes"] else None,
            "passes_saved": 2 * (len(path) - 1),
            "boundary": boundary,
        })

    blocked = {n: {"fanout": streams[n]["fanout"],
                   "consumers": [c[0] for c in streams[n]["consumers"]]}
               for n in produced
               if streams[n]["fanout"] >= 2}
    total_ops = sum(len(c["ops"]) for c in chains)
    total_pass = sum(c["passes_saved"] for c in chains)
    return {"chains": chains, "unanalyzable": unanalyzable,
            "blocked": blocked,
            "summary": {"n_ops": rep["n_ops"], "n_fused_ops": total_ops,
                        "n_chains": len(chains),
                        "passes_saved": total_pass,
                        "classes": rep["classes"]}}
