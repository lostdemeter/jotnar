"""Assembly v0.1: structure listings as executable text (stage-1 demo).

Format (minimal, linear, fail-loud):
  # comment / blank lines ignored
  CONFIG key value        # frozen config (beta, file paths); unknown keys fail
  IN name                 # declares the program input stream
  OUT [, OUT2] = MNEMONIC(arg, ...)   # one structure invocation; arity checked
Mnemonics resolve against REGISTRY (structure name -> python impl).
Unknown mnemonic, wrong arity, or missing input stream fails LOUD at
assemble time (never mid-run): the listing fully specifies the computation,
no hidden Python. Multi-output (AS, COH = SPLAT_BLUR(A)) is explicit --
no implicit registers, no magic feeds (v0.1 discipline).

This is stage 1 of the assembly program: the level is proven FAITHFUL
(listing output == hand-written chain output, bit-exact). Stage 2 proves it
GENERATIVE (new micro-AI written blind in listings). See docs/ASSEMBLY.md.
"""
import os

import numpy as np


class AsmError(Exception):
    pass


def parse(text):
    """text -> (config dict, input names, [(outs, mnemonic, args)])."""
    config, inp, prog = {}, [], []
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split(None, 1)
        head = toks[0].upper()
        if head == "CONFIG":
            kv = toks[1].split()
            if len(kv) != 2:
                raise AsmError(f"line {ln}: CONFIG needs key+value")
            config[kv[0]] = kv[1]
        elif head == "IN":
            names = [n.strip() for n in toks[1].split(",") if n.strip()]
            if not names:
                raise AsmError(f"line {ln}: IN needs a name")
            for nm in names:
                if nm in inp:
                    raise AsmError(f"line {ln}: duplicate IN {nm}")
                inp.append(nm)
        elif "=" in line:
            left, right = line.split("=", 1)
            outs = [o.strip() for o in left.split(",") if o.strip()]
            right = right.strip()
            # mnemonic with optional parens: OP(a, b) or OP a, b
            if "(" in right:
                mn, rest = right.split("(", 1)
                mn = mn.strip().upper()
                if not rest.endswith(")"):
                    raise AsmError(f"line {ln}: unbalanced parens")
                args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
            else:
                parts = right.split(None, 1)
                mn = parts[0].upper()
                args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []
                args = [a for a in args if a]
            if not outs or not mn:
                raise AsmError(f"line {ln}: bad instruction")
            prog.append((outs, mn, args))
        else:
            raise AsmError(f"line {ln}: unparseable: {raw!r}")
    if not inp:
        raise AsmError("no IN declared")
    return config, inp, prog


def assemble(text, registry):
    """Bind mnemonics + check arity/inputs statically (before any execution).
    Returns (config, inp, bound prog). Unknown mnemonic / arity mismatch /
    use-before-def fails here -- the static verifier stub (full verifier with
    scales+geometry is backlog item: range estimator + seam chart)."""
    config, inp, prog = parse(text)
    bound, defined = [], set(inp)
    for outs, mn, args in prog:
        if mn not in registry:
            raise AsmError(f"unknown mnemonic: {mn} "
                           f"(known: {sorted(registry)})")
        fn, arity_in, arity_out = registry[mn]
        if len(args) != arity_in:
            raise AsmError(f"{mn}: wants {arity_in} args, got {len(args)}")
        if len(outs) != arity_out:
            raise AsmError(f"{mn}: produces {arity_out}, got {len(outs)} outs")
        for a in args:
            if a not in defined and not _is_literal(a):
                raise AsmError(f"{mn}: input stream '{a}' not defined yet")
        bound.append((outs, fn, args))
        defined.update(outs)
    return config, inp, bound


def _is_literal(a):
    try:
        float(a)
        return True
    except ValueError:
        return False


def run(bound, config, inp, payload):
    """Execute bound program over a feeds dict. Returns feeds (all streams).
    Convention: an op with arity_out==1 produces ONE stream (even when the
    value is itself a tuple, e.g. triples); arity_out>1 must return a tuple
    of that length."""
    _, _, prog = bound
    # single-IN listings take a bare payload (backward compat); multi-IN
    # listings take {name: value}. Unknown payload keys fail loud.
    if isinstance(inp, str):
        inp = [inp]
    if len(inp) == 1 and not isinstance(payload, dict):
        feeds = {inp[0]: payload, "CONFIG": config}
    else:
        if not isinstance(payload, dict):
            raise AsmError(f"multi-IN program needs dict payload, got {type(payload)}")
        missing = [n for n in inp if n not in payload]
        if missing:
            raise AsmError(f"payload missing streams: {missing}")
        feeds = {n: payload[n] for n in inp}
        feeds["CONFIG"] = config
    for outs, fn, args in prog:
        vals = [feeds[a] if a in feeds else float(a) for a in args]
        out = fn(vals, config, feeds)
        # arity was checked at assemble time: one declared output takes the
        # whole return value (even a triples-tuple); N outputs take an N-tuple.
        out = (out,) if len(outs) == 1 else tuple(out)
        if len(out) != len(outs):
            raise AsmError(f"runtime arity break: {outs}")
        for name, val in zip(outs, out):
            feeds[name] = val
    return feeds


def run_text(text, registry, payload):
    """Parse + assemble + execute. Returns feeds."""
    config, inp, bound = assemble(text, registry)
    return run((config, inp, bound), config, inp, payload)
