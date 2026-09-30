"""Assembly v0.2: structure listings as executable text + typed streams.

Format (minimal, linear, fail-loud):
  # comment / blank lines ignored
  CONFIG key value        # frozen config (beta, file paths); unknown keys fail
  IN name [AS layout]     # declares program input streams, optionally typed
                          # (one AS applies to all names on the line)
  OUT [, OUT2] = MNEMONIC(arg, ...)   # one structure invocation; arity checked
  STATE name [, ...]            # loop-carried streams (repeat() only): must be
                                # IN-declared (seed) AND assigned as OUT every
                                # iteration; shapes fixed across iterations
                                # (dynamic shapes refused loudly -- ITERATE limit).
Layouts "KIND:GEOM" (kinds F float / T triples / I integer-exact / U8 bytes;
geom vocabulary open: HW/HWC/SEQ/HEADS/HW2/SCALAR...). Mnemonics resolve
against REGISTRY (structure name -> python impl); per-op layout signatures
live in SIGS (missing entry = all-wildcard, backward compatible).
Gradual typing: unannotated streams are UNKNOWN (unify with anything, never
fail); concrete-vs-concrete mismatches fail LOUD naming op+line+stream.
Literals have layout F:SCALAR. "$VAR" unifies whole layout strings;
"$VAR^T" derives (transpose rule); "*" matches anything, binds nothing.
Limits (stated, Batch 3): no rank arithmetic (MATMUL is wildcard; phi-core
asserts inside fail loud), no dynamic shapes, checks run at execute time
(layouts ride payloads, unknown until then).
"""
import os

import numpy as np


class AsmError(Exception):
    pass


def parse(text):
    """text -> (config dict, [(name, layout|None)], [(outs, mn, args, ln)], [state])."""
    config, inp, prog, state = {}, [], [], []
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
        elif head == "RANGE":
            # range declarations feed the static estimator (ranges.py);
            # run() ignores them (values, not intervals, flow at runtime).
            parts = line.split()
            if len(parts) != 4:
                raise AsmError(f"line {ln}: RANGE needs name+lo+hi")
            try:
                float(parts[2])
                float(parts[3])
            except ValueError:
                raise AsmError(f"line {ln}: RANGE bounds must be numeric")
        elif head == "STATE":
            names = [n.strip() for n in toks[1].split(",") if n.strip()]
            if not names:
                raise AsmError(f"line {ln}: STATE needs a name")
            for nm in names:
                if nm in state:
                    raise AsmError(f"line {ln}: duplicate STATE {nm}")
                state.append(nm)
        elif head == "IN":
            rest = toks[1]
            up = rest.upper()
            if " AS " in up:
                idx = up.index(" AS ")
                names = [n.strip() for n in rest[:idx].split(",") if n.strip()]
                lay = rest[idx + 4:].strip() or None
            else:
                names = [n.strip() for n in rest.split(",") if n.strip()]
                lay = None
            if not names:
                raise AsmError(f"line {ln}: IN needs a name")
            for nm in names:
                if nm in [n for n, _ in inp]:
                    raise AsmError(f"line {ln}: duplicate IN {nm}")
                inp.append((nm, lay))
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
            prog.append((outs, mn, args, ln))
        else:
            raise AsmError(f"line {ln}: unparseable: {raw!r}")
    if not inp:
        raise AsmError("no IN declared")
    for nm in state:
        if nm not in [n for n, _ in inp]:
            raise AsmError(f"STATE {nm} must also be IN-declared (seed)")
    assigned = set(o for outs, _, _, _ in prog for o in outs)
    for nm in state:
        if nm not in assigned:
            raise AsmError(f"STATE {nm} is never assigned as OUT (no carry)")
    return config, inp, prog, state


def assemble(text, registry, sigs=None):
    """Bind mnemonics + check arity/inputs statically (before any execution).
    Returns (config, inp, bound, state) with bound entries
    (outs, mn, fn, args, ln, sig). Unknown mnemonic / arity mismatch /
    use-before-def fails here. Layouts check at run time (payloads unknown
    until then); verify() below replays unification over DECLARED layouts
    for parse-time conflicts. Full verifier with scales+geometry is backlog:
    range estimator + seam chart."""
    sigs = sigs or {}
    config, inp, prog, state = parse(text)
    bound, defined = [], set(n for n, _ in inp)
    for outs, mn, args, ln in prog:
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
        bound.append((outs, mn, fn, args, ln, sigs.get(mn)))
        defined.update(outs)
    return config, inp, bound, state


def _is_literal(a):
    try:
        float(a)
        return True
    except ValueError:
        return False


_UNKNOWN = "UNKNOWN"


def _resolve(pat, actual, bindings):
    """Unify one layout pattern against an actual layout. Returns the
    resulting layout (concrete or UNKNOWN). Raises AsmError on concrete
    mismatch (reported with op context by the caller)."""
    if pat == "*":
        return actual
    if actual == _UNKNOWN:
        return pat if not pat.startswith("$") else _UNKNOWN
    if pat.startswith("$"):
        base, derived = (pat[:-2], True) if pat.endswith("^T") else (pat, False)
        if base in bindings:
            if bindings[base] != _UNKNOWN and bindings[base] != actual:
                raise AsmError(f"layout conflict: {base} is {bindings[base]}, got {actual}")
            got = bindings[base] if bindings[base] != _UNKNOWN else actual
        else:
            got = actual
        bindings[base] = got
        if got == _UNKNOWN:
            return _UNKNOWN
        return got + "^T" if derived else got
    if pat != actual:
        raise AsmError(f"layout mismatch: want {pat}, got {actual}")
    return pat


def _check_layouts(outs, mn, args, ln, sig, layouts):
    """Unify one instruction's layouts. Returns {out: layout}. Raises
    AsmError naming op+line+stream on concrete conflict. Shared by run()
    (live layouts) and verify() (declared-only layouts)."""
    in_pats, out_pats = sig if sig is not None else (["*"] * len(args), ["*"] * len(outs))
    bindings = {}
    try:
        for a, pat in zip(args, in_pats):
            actual = layouts.get(a, "F:SCALAR" if _is_literal(a) else _UNKNOWN)
            _resolve(pat, actual, bindings)
        out_lays = []
        for pat in out_pats:
            if pat == "*":
                out_lays.append(_UNKNOWN)
            elif pat.startswith("$"):
                base = pat[:-2] if pat.endswith("^T") else pat
                got = bindings.get(base, _UNKNOWN)
                out_lays.append(_UNKNOWN if got == _UNKNOWN else (got + "^T" if pat.endswith("^T") else got))
            else:
                out_lays.append(pat)
    except AsmError as e:
        raise AsmError(f"line {ln} ({mn}): {e}")
    return dict(zip(outs, out_lays))


def verify(text, registry, sigs=None):
    """Static layout pass (no payloads, no execution): replays unification
    over DECLARED layouts only (IN AS + concrete sig patterns). Returns
    (errors, report): errors = list of conflict strings derivable without
    values; report has resolved/total coverage counts (gradual typing means
    absence of proof -- verify() reports coverage honestly)."""
    config, inp, bound, _ = assemble(text, registry, sigs)
    layouts = {n: (l or _UNKNOWN) for n, l in inp}
    errors = []
    for outs, mn, fn, args, ln, sig in bound:
        try:
            layouts.update(_check_layouts(outs, mn, args, ln, sig, layouts))
        except AsmError as e:
            errors.append(str(e))
            for o in outs:
                layouts[o] = _UNKNOWN
    total = len(layouts)
    resolved = sum(1 for v in layouts.values() if v != _UNKNOWN)
    return errors, {"resolved": resolved, "total": total,
                    "layouts": dict(layouts)}


def run(bound, config, inp, payload):
    """Execute bound program over a feeds dict. Returns feeds (all streams).
    Convention: an op with arity_out==1 produces ONE stream (even when the
    value is itself a tuple, e.g. triples); arity_out>1 must return a tuple
    of that length. Layouts check per instruction (parallel dict; values
    untouched): concrete-vs-concrete mismatch fails naming op+line+stream."""
    _, _, prog = bound
    # single-IN listings take a bare payload (backward compat); multi-IN
    # listings take {name: value}. Unknown payload keys fail loud.
    if isinstance(inp, str):
        inp = [(inp, None)]
    in_names = [n for n, _ in inp]
    in_lay = {n: (l or _UNKNOWN) for n, l in inp}
    if len(in_names) == 1 and not isinstance(payload, dict):
        feeds = {in_names[0]: payload, "CONFIG": config}
        layouts = {in_names[0]: in_lay[in_names[0]]}
    else:
        if not isinstance(payload, dict):
            raise AsmError(f"multi-IN program needs dict payload, got {type(payload)}")
        missing = [n for n in in_names if n not in payload]
        if missing:
            raise AsmError(f"payload missing streams: {missing}")
        feeds = {n: payload[n] for n in in_names}
        feeds["CONFIG"] = config
        layouts = dict(in_lay)
    for outs, mn, fn, args, ln, sig in prog:
        vals = [feeds[a] if a in feeds else float(a) for a in args]
        out_lays = _check_layouts(outs, mn, args, ln, sig, layouts)
        out = fn(vals, config, feeds)
        # arity was checked at assemble time: one declared output takes the
        # whole return value (even a triples-tuple); N outputs take an N-tuple.
        out = (out,) if len(outs) == 1 else tuple(out)
        if len(out) != len(outs):
            raise AsmError(f"line {ln} ({mn}): runtime arity break: {outs}")
        for name, val in zip(outs, out):
            feeds[name] = val
            layouts[name] = out_lays[name]
    return feeds


def run_text(text, registry, payload, sigs=None):
    """Parse + assemble + execute. Returns feeds."""
    config, inp, bound, _ = assemble(text, registry, sigs)
    return run((config, inp, bound), config, inp, payload)


def _shapes_of(v):
    """Structural shape of a feed value (triples tuple -> first plane shape;
    None -> None). For STATE shape-stability checks."""
    if v is None:
        return None
    if isinstance(v, tuple) and len(v) == 3 and hasattr(v[0], "shape"):
        return tuple(v[0].shape)
    if hasattr(v, "shape"):
        return tuple(v.shape)
    return None


def repeat(text, registry, payload, n, sigs=None, grow=()):
    """Run a listing n times threading STATE streams (loop-carried state).
    Non-STATE IN streams take LISTS of length n (one payload per iteration;
    strict -- no implicit broadcasting, length bugs fail loud). STATE streams
    take a single seed from payload and update from outputs each iteration;
    every STATE name must be assigned every iteration (else fail loud).
    grow: subset of STATE names allowed APPEND-ONLY growth along axis 0
    (the KV pattern: cache rows accumulate; all other streams keep fixed
    geometry, refused loudly on change). Growth is logged per iteration and
    reported (no silent shape drift -- the ITERATE discipline).
    Dynamic shapes beyond append-only, and data-dependent termination
    (WHILE), are out of scope -- stated, see GAPS.md. Returns
    (per-iteration OUT feeds list, final feeds)."""
    config, inp, bound, state = assemble(text, registry, sigs)
    in_names = [nm for nm, _ in inp]
    if not isinstance(payload, dict):
        raise AsmError("repeat needs dict payload")
    grow = set(grow)
    for nm in grow:
        if nm not in state:
            raise AsmError(f"grow stream '{nm}' is not a STATE stream")
    seqs = {}
    for nm in in_names:
        if nm not in payload:
            raise AsmError(f"payload missing streams: {[nm]}")
        v = payload[nm]
        if nm in state:
            seqs[nm] = v
        else:
            if not isinstance(v, list) or len(v) != n:
                raise AsmError(f"non-STATE stream '{nm}' needs a list of {n} payloads")
            seqs[nm] = v
    # v1 fixed-geometry doctrine: every list payload shares spatial (H, W).
    # Varying geometry across frames is refused loudly here (dynamic shapes
    # are out of scope) rather than dying inside an op's broadcast.
    geoms = set()
    for nm, v in seqs.items():
        if nm in state or not isinstance(v, list):
            continue
        for i, item in enumerate(v):
            sh = _shapes_of(item)
            if sh is not None and len(sh) >= 2:
                geoms.add(sh[:2])
    if len(geoms) > 1:
        raise AsmError(f"sequence geometry varies across frames: {sorted(geoms)} (fixed geometry only)")
    histories, carry, shapes, growth = [], {}, {}, {nm: [] for nm in grow}
    for it in range(n):
        feeds = {nm: (seqs[nm][it] if nm not in state else seqs[nm] if it == 0 else carry[nm])
                 for nm in in_names}
        feeds["CONFIG"] = config
        got = run((config, inp, bound), config, inp, feeds)
        for nm in state:
            if nm not in got:
                raise AsmError(f"iteration {it}: STATE '{nm}' not assigned")
            sh = _shapes_of(got[nm])
            if nm in shapes and shapes[nm] is not None and sh is not None:
                if sh != shapes[nm] and nm not in grow:
                    raise AsmError(f"iteration {it}: STATE '{nm}' shape changed {shapes[nm]} -> {sh} (dynamic shapes refused)")
                if nm in grow:
                    if len(sh) != len(shapes[nm]) or sh[1:] != shapes[nm][1:]:
                        raise AsmError(f"iteration {it}: STATE '{nm}' grew off-axis {shapes[nm]} -> {sh} (append-only along axis 0)")
                    if sh[0] < shapes[nm][0]:
                        raise AsmError(f"iteration {it}: STATE '{nm}' shrank {shapes[nm]} -> {sh} (append-only)")
            if sh is not None:
                shapes[nm] = sh
                if nm in grow:
                    growth[nm].append(sh)
            carry[nm] = got[nm]
        histories.append(got)
    return histories, histories[-1], growth
