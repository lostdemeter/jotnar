"""Access-shape census (v1.3 gate 1): how data MOVES, not its statistics.

A listing is a spatial program: streams with geometry flow through ops that
read them in specific patterns. The census maps that movement statically
(zero execution): every stream gets producer op+line, consumer ops+lines,
and fan-out; every op gets an access class. Shared streams (fan-out >= 2)
are reuse made visible -- the same reuse the promotion rule prices.

Access classes: elementwise (per-point map), moves (exact, no arithmetic),
gather (reindex by ids), reduce (many-to-fewer: matmuls, norms, pools),
select (verdict-gated mux), stencil (neighborhood/sample reads: convs,
warps, blends), broadcast (geometry-expanding: scalars to fields).
"""
from collections import defaultdict

from chain import asm as ASM

ACCESS = {
    # elementwise maps
    "SUB": "elementwise", "MUL": "elementwise", "ADD": "elementwise",
    "DIV": "elementwise", "SQUARE": "elementwise", "SIGMOID": "elementwise",
    "SILU": "elementwise", "GELU": "elementwise", "CLIP": "elementwise",
    "PRELU": "elementwise", "SQRT": "elementwise", "STATIC": "elementwise",
    "BETA_V5": "elementwise", "SRGB_DECODE": "elementwise",
    "SRGB_ENCODE": "elementwise", "RESCALE": "elementwise",
    # exact moves (no arithmetic)
    "TRANSPOSE": "moves", "RESHAPE2": "moves", "RESHAPE3": "moves",
    "PERMUTE3": "moves", "CONCAT": "moves", "SLICE": "moves",
    # reindex / selection / expansion
    "GATHER": "gather",
    "SELECT": "select",
    "BETA": "broadcast", "GAIN": "broadcast",
    # reductions (many-to-fewer)
    "MATMUL": "reduce", "BATCH_MATMUL": "reduce", "POOLAVG": "reduce",
    "ARGMAX": "reduce", "RMSNORM": "reduce", "LUMA": "reduce",
    "SOFTMAX": "reduce", "ROTARY": "elementwise",
    # neighborhood / sample reads
    "SPLAT_BLUR": "stencil", "ISO_BLUR": "stencil", "GAUSS": "stencil",
    "CONV": "stencil", "DECONV": "stencil", "INTERP": "stencil",
    "WARP": "stencil", "MIXDYAD": "stencil",
}


def census_text(text, registry, sigs=None, basedir=".", stdlib=None):
    """Static access census of a listing. Returns report dict:
    streams: {name: {producer: (op, line) | None (source), consumers:
    [(op, line, version)], fanout}, ops: [{op, line, class, reads, writes}],
    shared: {name: fanout} for fanout >= 2 (reuse made visible),
    classes: {class: count}, state: [STATE names], shadowed: [IN names
    reassigned as OUT -- seed vs carried versions differ, see versions].
    Versions (v1.1 of this instrument ignored them): IN seeds read at v0,
    each OUT assignment bumps (non-IN streams start at v0 on first assign).
    Single-assign streams behave exactly as before (producer tuple +
    fan-out counts unchanged)."""
    config, inp, bound, state = ASM.assemble(text, registry, sigs, basedir,
                                           stdlib)
    producers, consumers = {}, defaultdict(list)
    ops = []
    in_names = {n for n, _ in inp}
    ver = {n: 0 for n, _ in inp}  # IN seeds read at v0
    versions = defaultdict(set)
    for n in in_names:
        versions[n].add(0)
    for outs, mn, fn, args, ln, sig in bound:
        reads = [a for a in args if not ASM._is_literal(a)]
        for a in reads:
            consumers[a].append((mn, ln, ver.get(a, 0)))
            versions[a].add(ver.get(a, 0))
        for o in outs:
            ver[o] = ver.get(o, -1) + 1
            producers[o] = (mn, ln)
            versions[o].add(ver[o])
        ops.append({"op": mn, "line": ln, "class": ACCESS.get(mn, "unknown"),
                    "reads": reads, "writes": list(outs)})
    streams = {}
    for name in set(producers) | set(consumers):
        streams[name] = {"producer": producers.get(name),
                         "consumers": consumers.get(name, []),
                         "fanout": len(consumers.get(name, [])),
                         "source": name in in_names,
                         "versions": sorted(versions.get(name, set()))}
    shadowed = sorted(n for n in in_names if n in producers)
    shared = {n: s["fanout"] for n, s in streams.items() if s["fanout"] >= 2}
    classes = defaultdict(int)
    for o in ops:
        classes[o["class"]] += 1
    uncovered = sorted({o["op"] for o in ops if o["class"] == "unknown"})
    return {"streams": streams, "ops": ops, "shared": shared,
            "classes": dict(classes), "uncovered": uncovered,
            "n_streams": len(streams), "n_ops": len(ops),
            "state": list(state), "shadowed": shadowed}
