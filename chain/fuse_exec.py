"""Fused lattice execution: run advisor chains without materializing.

run_fused(text, chain_lines, payload, ...) mirrors chain/asm.run()
exactly -- same assemble, same op functions, same order, same checks
-- with one change: ops whose bound line is in chain_lines write
their outputs to a side env instead of feeds (reads check env first
via ChainMap, so even hypothetical feeds-reading ops behave). Chain
intermediates never materialize as feeds entries; the chain-final op
always materializes (boundary output). Same functions + same order =
byte-identical by construction (gated anyway, never trusted).

Consumes advise() output mechanically: chain_lines is the report's
`lines` for one chain (the gate asserts identity with the advisor's
current report -- hand-wiring a chain without the advisor link is a
scope violation). Generic over chains: zero per-chain code here.
Wall savings on lattice are feeds-dict traffic only (numpy still
allocates per-op temporaries in both arms) -- reported, never
asserted beyond the row. No emitter changes; CUDA emission of fused
chains is blocked (T-kind + missing BETA_V5/GAIN/SRGB patterns --
see docs/FUSION.md), not attempted here.
"""
from collections import ChainMap

import numpy as np

from chain import asm as ASM


def run_fused(text, chain_lines, payload, registry=None, sigs=None,
              basedir=".", trace=None, overrides=None):
    """Execute with the given bound lines fused (locals, not feeds).
    Returns (feeds, fused, env): feeds holds everything EXCEPT fused
    interior streams (assert their absence to prove non-materialization;
    their values live in env for byte-compare), fused counts ops run
    into the side env (== len(chain_lines) minus chain-finals on
    success)."""
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    config, inp, bound, _ = ASM.assemble(text, registry or _R,
                                         sigs or _S, basedir)
    chain_lines = set(chain_lines)
    overrides = dict(overrides or {})
    pending = set(overrides)
    if isinstance(inp, str):
        inp = [(inp, None)]
    in_names = [n for n, _ in inp]
    in_lay = {n: (l or ASM._UNKNOWN) for n, l in inp}
    if len(in_names) == 1 and not isinstance(payload, dict):
        feeds = {in_names[0]: payload, "CONFIG": config}
        layouts = {in_names[0]: in_lay[in_names[0]]}
    else:
        if not isinstance(payload, dict):
            raise ASM.AsmError(
                f"multi-IN program needs dict payload, got {type(payload)}")
        missing = [n for n in in_names if n not in payload]
        if missing:
            raise ASM.AsmError(f"payload missing streams: {missing}")
        feeds = {n: payload[n] for n in in_names}
        feeds["CONFIG"] = config
        layouts = dict(in_lay)
    env = {}
    fused = 0
    prog = bound
    # chain-final ops (consumed outside the chain, or sinks) materialize
    # to feeds; interior ops stay in env. Derived from the bound consumer
    # map -- no caller annotation, no per-chain code.
    consumers = {}
    for outs2, _, _, args2, ln2, _ in prog:
        for a in args2:
            if ASM._is_literal(a):
                continue
            consumers.setdefault(a, []).append(ln2)
    for outs, mn, fn, args, ln, sig in prog:
        view = ChainMap(env, feeds)
        vals = [view[a] if a in view else float(a) for a in args]
        out_lays = ASM._check_layouts(outs, mn, args, ln, sig, layouts)
        for a in args:
            if ASM._is_literal(a) or a not in layouts:
                continue
            lay = layouts[a]
            if lay == ASM._UNKNOWN or ":" not in lay:
                continue
            want = lay.split(":")[0]
            got = ASM._kind_of(view[a]) if a in view else ASM._UNKNOWN
            if got != ASM._UNKNOWN and got != want:
                raise ASM.AsmError(
                    f"line {ln} ({mn}): stream '{a}' declared {lay} "
                    f"but holds {got}")
        if trace is not None:
            import time
            in_sum = {}
            for a in args:
                ssum = ASM._summarize(view[a]) if a in view else {
                    "kind": "literal", "value": a}
                ssum["layout"] = layouts.get(
                    a, "F:SCALAR" if ASM._is_literal(a) else ASM._UNKNOWN)
                in_sum[a] = ssum
            trace.append({"line": ln, "op": mn,
                          "in": in_sum,
                          "out": {}, "sec": 0.0, "_t0": time.perf_counter()})
        ASM._check_shapes(mn, vals, ln)
        out = fn(vals, config, view)
        out = (out,) if len(outs) == 1 else tuple(out)
        if len(out) != len(outs):
            raise ASM.AsmError(
                f"line {ln} ({mn}): runtime arity break: {outs}")
        Con = [c for o in outs for c in consumers.get(o, [])]
        if ln in chain_lines and Con and all(c in chain_lines for c in Con):
            fused += 1
            store = env
        else:
            store = feeds
        for name, val in zip(outs, out):
            store[name] = val
            layouts[name] = out_lays[name]
        for name in outs:
            if name in overrides:
                (env if name in env else feeds)[name] = overrides[name]
                pending.discard(name)
        if trace is not None:
            rec = trace[-1]
            rec["out"] = {}
            for name, val in zip(outs, out):
                ssum = ASM._summarize(val)
                ssum["layout"] = out_lays[name]
                rec["out"][name] = ssum
            rec["sec"] = time.perf_counter() - rec.pop("_t0")
    if pending:
        raise ASM.AsmError(f"overrides never assigned: {sorted(pending)} "
                           f"(typo, or stream optimized out -- probes must bite)")
    return feeds, fused, env
