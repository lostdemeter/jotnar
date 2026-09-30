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


def _head_of(line):
    toks = line.split(None, 1)
    return toks[0].upper() if toks else ""


def expand(text, basedir=".", origin="<?>", defs=None, stack=()):
    """Pre-pass: IMPORT splicing + DEF collection + CALL expansion.
    Returns (flat_lines, defs) where flat_lines = [(text, origin_str)] and
    origin_str names the source ("file:12" or "file:12 via CALL@site:5").
    Rules (all fail loud): IMPORT cycles; nested DEF; CONFIG/STATE inside
    DEF; recursive CALL (expansion stack); arity mismatch on CALL;
    duplicate DEF names. Expansion is textual macro semantics with
    per-call-site namespacing (prefix name#k.): no runtime call overhead,
    no closures -- the expanded listing is the program (inspectable)."""
    import os as _os
    if defs is None:
        defs = {}
    raw = [(l, f"{origin}:{i}") for i, l in enumerate(text.splitlines(), 1)]
    flat, i, counter = [], 0, [0]
    # Pre-scan current-file IN declarations so CALL-site contract checks see
    # caller layouts (IMPORTed INs stay UNKNOWN -> deferred to post-expansion
    # verify + runtime; gradual typing across files, stated).
    in_lay0 = {}
    for _raw in raw:
        _code = _raw[0].split("#", 1)[0].strip()
        _toks = _code.split(None, 1)
        if _toks and _toks[0].upper() == "IN" and len(_toks) > 1:
            _rest, _up = _toks[1], _toks[1].upper()
            if " AS " in _up:
                _idx = _up.index(" AS ")
                for _nm in _rest[:_idx].split(","):
                    _nm = _nm.strip()
                    if _nm:
                        in_lay0[_nm] = _rest[_idx + 4:].strip() or None
    # NOTE: raw lines get comment-stripped when READ (code = ... below);
    # expanded lines emitted by _expand_call derive from already-stripped
    # code, so parse() must NOT re-strip them (generated names contain '#',
    # e.g. twice#1.T -- re-stripping truncates them: caught by gate).
    # Convention: flat entries are (text, origin, stripped_bool).

    def _expand_call(mn_args, outs, site, dstack=()):
        # mn_args like "name(p, q)"; outs = caller out names
        mm = __import__("re").match(r"(\w+)\s*\((.*)\)\s*$", mn_args)
        if not mm:
            raise AsmError(f"{site}: bad CALL syntax (want name(a, b))")
        name, argstr = mm.group(1), mm.group(2)
        actuals = [a.strip() for a in argstr.split(",") if a.strip()]
        if name not in defs:
            raise AsmError(f"{site}: CALL of unknown DEF '{name}'")
        if name in dstack:
            raise AsmError(f"{site}: recursive CALL '{name}' (def stack {dstack})")
        D = defs[name]
        fins, fouts, body = D["fins"], D["fouts"], D["body"]
        if len(actuals) != len(fins):
            raise AsmError(f"{site}: CALL {name} wants {len(fins)} args, got {len(actuals)}")
        if len(outs) != len(fouts):
            raise AsmError(f"{site}: CALL {name} produces {len(fouts)}, got {len(outs)} outs")
        # contract check: formal AS layouts vs caller DECLARED layouts.
        # Both known + concrete + unequal -> fail naming CALL site AND DEF
        # origin. UNKNOWN either side defers (post-expansion verify + runtime
        # cover it). Formal $VAR layouts are meaningless here -> fail loud.
        for fml, act in zip(fins, actuals):
            fl = D["fin_lay"].get(fml)
            if fl is not None and fl.startswith("$"):
                raise AsmError(f"{site}: DEF '{name}' formal '{fml}' must not use $VAR layouts")
            al = in_lay0.get(act)
            if fl and al and fl != al:
                raise AsmError(f"{site}: CALL {name}('{act}' is {al}) violates "
                               f"formal '{fml}' wants {fl} (DEF at {D['origin']})")
        counter[0] += 1
        pre = f"{name}#{counter[0]}."
        mapping = dict(zip(fins, actuals))
        mapping.update(zip(fouts, outs))
        out = []
        for btext, borig, _ in body:
            code = btext.split("#", 1)[0].strip()
            if not code:
                continue
            toks = code.split(None, 1)
            if toks and toks[0].upper() == "CALL":
                sub = _expand_call(toks[1], [], f"{borig} via CALL@{site}",
                                   dstack + (name,))
                out.extend(sub)
                continue
            # rewrite stream names: formals -> actuals, locals -> prefixed
            def _rw(m):
                w = m.group(0)
                if w in mapping:
                    return mapping[w]
                return pre + w
            import re as _re
            # split off LHS outs vs RHS to map correctly
            if "=" in code:
                left, right = code.split("=", 1)
                lo = [o.strip() for o in left.split(",")]
                lo = [mapping.get(o, pre + o) if o not in mapping else mapping[o] for o in lo]
                # mnemonic head stays; args rewritten. Head splits at the
                # paren (NOT whitespace -- "ADD(x, x)" split on space gives
                # head "ADD(x,"; caught by gate, documented).
                import re as _re2
                rs = right.strip()
                # NOTE: split CALL form FIRST ("CALL name(args)" has a space
                # before the paren -- naive paren-split yields head "CALL
                # name" and silently keeps it as a mnemonic; caught by gate).
                _mcall = _re2.match(r"(?i)(CALL)\s+(\w+)\s*\((.*)\)\s*$", rs)
                if _mcall:
                    sub = _expand_call(f"{_mcall.group(2)}({ _mcall.group(3)})",
                                       lo, f"{borig} via CALL@{site}",
                                       dstack + (name,))
                    out.extend(sub)
                    continue
                if "(" in rs:
                    h2, r2 = rs.split("(", 1)
                    head, rest = h2.strip(), "(" + r2
                else:
                    parts = rs.split(None, 1)
                    head = parts[0]
                    rest = parts[1] if len(parts) > 1 else ""
                if True:  # head/rest always paired by construction above
                    if head.upper() == "CALL":
                        # assignment-form inner call: recurse with the stack
                        # (recursion detected via stack, not refused here).
                        inner = rest[1:-1] if (rest.startswith("(") and rest.endswith(")")) else rest
                        mm2 = __import__("re").match(r"(\w+)\s*\((.*)\)\s*$", inner)
                        if not mm2:
                            raise AsmError(f"{borig}: bad CALL syntax (want name(a, b))")
                        sub = _expand_call(inner, lo,
                                           f"{borig} via CALL@{site}",
                                           dstack + (name,))
                        out.extend(sub)
                        continue
                    if rest.startswith("(") and rest.endswith(")"):
                        rargs = [a.strip() for a in rest[1:-1].split(",") if a.strip()]
                        rargs = [mapping.get(a, pre + a) if not _is_literal(a) else a for a in rargs]
                        right2 = f"{head}({', '.join(rargs)})"
                    else:
                        rargs = [a.strip() for a in rest.split(",") if a.strip()]
                        rargs = [mapping.get(a, pre + a) if not _is_literal(a) else a for a in rargs]
                        right2 = f"{head} {', '.join(rargs)}" if rargs else head
                    code = f"{', '.join(lo)} = {right2}"
            else:
                code = _re.sub(r"[A-Za-z_]\w*", _rw, code)
            out.append((code, f"{borig} via CALL@{site}", True))
        return out

    while i < len(raw):
        line, org = raw[i]
        i += 1
        code = line.split("#", 1)[0].strip()
        if not code:
            continue
        toks = code.split(None, 1)
        head = toks[0].upper()
        if head == "IMPORT":
            target = (toks[1].strip().strip("\"'") if len(toks) > 1 else "")
            if not target:
                raise AsmError(f"{org}: IMPORT needs a path")
            import os as _os2
            path = _os2.path.normpath(_os2.path.join(basedir, target))
            real = _os2.path.realpath(path)
            if real in stack:
                raise AsmError(f"{org}: IMPORT cycle ({real})")
            try:
                with open(path) as fh:
                    sub = fh.read()
            except OSError as e:
                raise AsmError(f"{org}: IMPORT cannot read {path}: {e}")
            flat.extend(expand(sub, _os2.path.dirname(path), path, defs,
                               stack + (real,))[0])
        elif head == "DEF":
            import re as _re3
            m = _re3.match(r"DEF\s+(\w+)\s*\((.*)\)\s*->\s*\((.*)\)\s*$", code)
            if not m:
                raise AsmError(f"{org}: bad DEF syntax (want DEF name(a, b) -> (c))")

            def _formals(spec, what):
                out = []
                for part in spec.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    up = part.upper()
                    if " AS " in up:
                        idx = up.index(" AS ")
                        nm, lay = part[:idx].strip(), part[idx + 4:].strip() or None
                    else:
                        nm, lay = part, None
                    if not nm.replace("_", "").isalnum() or not nm[0].isalpha():
                        raise AsmError(f"{org}: bad {what} name '{nm}'")
                    out.append((nm, lay))
                return out

            name = m.group(1)
            fins = _formals(m.group(2), "formal")
            fouts = _formals(m.group(3), "return")
            if name in defs:
                raise AsmError(f"{org}: duplicate DEF '{name}'")
            body = []
            while i < len(raw):
                bline, borg = raw[i]
                i += 1
                bcode = bline.split("#", 1)[0].strip()
                if not bcode:
                    continue
                btoks = bcode.split(None, 1)
                if btoks[0].upper() == "END":
                    break
                if btoks[0].upper() == "DEF":
                    raise AsmError(f"{borg}: nested DEF not allowed")
                if btoks[0].upper() in ("CONFIG", "STATE", "IN", "IMPORT", "RANGE"):
                    raise AsmError(f"{borg}: {btoks[0].upper()} not allowed inside DEF (program-global only)")
                body.append((bline, borg, False))
            else:
                raise AsmError(f"{org}: DEF '{name}' missing END")
            defs[name] = {"fins": [n for n, _ in fins],
                          "fouts": [n for n, _ in fouts],
                          "fin_lay": {n: l for n, l in fins},
                          "fout_lay": {n: l for n, l in fouts},
                          "body": body, "origin": org}
        elif head == "CALL":
            raise AsmError(f"{org}: CALL needs outputs (want x = CALL name(...))")
        else:
            # assignment-form CALL: OUTs = CALL name(args)
            if "=" in code:
                left, right = code.split("=", 1)
                rparts = right.strip().split(None, 1)
                if rparts and rparts[0].upper() == "CALL":
                    if len(rparts) < 2:
                        raise AsmError(f"{org}: CALL needs a callee (want x = CALL name(...))")
                    outs = [o.strip() for o in left.split(",") if o.strip()]
                    flat.extend(_expand_call(rparts[1], outs, org))
                    continue
            flat.append((line, org, False))
    return flat, defs


def parse(text, basedir="."):
    """text -> (config dict, [(name, layout|None)], [(outs, mn, args, ln)], [state]).
    DEF/IMPORT/CALL expand first (origins tracked); the listing below sees
    flat lines with origin strings ("file:12", "file:12 via CALL@site:5")."""
    config, inp, prog, state = {}, [], [], []
    flat, _ = expand(text, basedir, origin=basedir)
    for raw, org, pre_stripped in flat:
        ln = org
        line = raw if pre_stripped else raw.split("#", 1)[0].strip()
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


def assemble(text, registry, sigs=None, basedir="."):
    """Bind mnemonics + check arity/inputs statically (before any execution).
    Returns (config, inp, bound, state) with bound entries
    (outs, mn, fn, args, ln, sig). Unknown mnemonic / arity mismatch /
    use-before-def fails here. Layouts check at run time (payloads unknown
    until then); verify() below replays unification over DECLARED layouts
    for parse-time conflicts. Full verifier with scales+geometry is backlog:
    range estimator + seam chart."""
    sigs = sigs or {}
    config, inp, prog, state = parse(text, basedir)
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


def verify(text, registry, sigs=None, basedir="."):
    """Static layout pass (no payloads, no execution): replays unification
    over DECLARED layouts only (IN AS + concrete sig patterns). Returns
    (errors, report): errors = list of conflict strings derivable without
    values; report has resolved/total coverage counts (gradual typing means
    absence of proof -- verify() reports coverage honestly) plus the DEF
    interface table (composition contracts visible in one place)."""
    config, inp, bound, _ = assemble(text, registry, sigs, basedir)
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
    _, defs = expand(text, basedir, origin=basedir)
    iface = {n: {"formals": [(f, d["fin_lay"].get(f)) for f in d["fins"]],
                 "returns": [(f, d["fout_lay"].get(f)) for f in d["fouts"]],
                 "origin": d["origin"]} for n, d in defs.items()}
    return errors, {"resolved": resolved, "total": total,
                    "layouts": dict(layouts), "defs": iface}


def _summarize(v):
    """Cheap per-stream summary for traces (shapes always; value stats for
    numerics; triples decoded coarsely). Never fails: summarization must not
    break the program it observes (unrepresentable values -> best effort)."""
    try:
        if isinstance(v, tuple) and len(v) == 3 and all(hasattr(x, "shape") for x in v):
            import numpy as _np
            try:
                from phi_core import lattice as _S
                dec = _S.decode(_np.ascontiguousarray(v[0]),
                                _np.ascontiguousarray(v[1]))
                m = (1 - _np.ascontiguousarray(v[2]).astype(float))
                a = dec * m
                return {"kind": "triples", "shape": tuple(v[0].shape),
                        "min": float(_np.min(a)), "max": float(_np.max(a)),
                        "mean": float(_np.mean(a))}
            except Exception:
                return {"kind": "triples",
                        "shape": tuple(v[0].shape)}
        import numpy as _np
        a = _np.ascontiguousarray(v)
        if a.dtype == bool:
            return {"kind": "mask", "shape": tuple(a.shape),
                    "true": int(a.sum()), "n": int(a.size)}
        if np.issubdtype(a.dtype, np.number):
            with np.errstate(all="ignore"):
                return {"kind": str(a.dtype), "shape": tuple(a.shape),
                        "min": float(_np.min(a)), "max": float(_np.max(a)),
                        "mean": float(_np.mean(a.astype(float)))}
        return {"kind": str(a.dtype), "shape": tuple(a.shape)}
    except Exception as e:
        return {"kind": "unrepresentable", "note": str(e)[:60]}


def _kind_of(v):
    """Value kind for layout cross-check: T (triples tuple), F (float
    array), I (int/bool array), U8 (uint8 array), else UNKNOWN."""
    import numpy as _np
    if isinstance(v, tuple) and len(v) == 3 and all(hasattr(x, "shape") for x in v):
        return "T"
    if isinstance(v, _np.ndarray):
        if v.dtype == bool or np.issubdtype(v.dtype, np.integer):
            return "U8" if v.dtype == np.uint8 else "I"
        if np.issubdtype(v.dtype, np.floating):
            return "F"
        return _UNKNOWN
    if isinstance(v, (int, float)):
        return "F"
    return _UNKNOWN


def run(bound, config, inp, payload, trace=None):
    """Execute bound program over a feeds dict. Returns feeds (all streams).
    Convention: an op with arity_out==1 produces ONE stream (even when the
    value is itself a tuple, e.g. triples); arity_out>1 must return a tuple
    of that length. Layouts check per instruction (parallel dict; values
    untouched): concrete-vs-concrete mismatch fails naming op+line+stream.
    trace: None (default, old behavior exactly) or a list to append per-op
    records {line, op, in: {stream: summary}, out: {...}, sec}. Summaries
    are best-effort and never affect values (see _summarize)."""
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
        # kind cross-check: a DECLARED layout's kind prefix must match the
        # value (triple-ops receiving float arrays silently compute garbage
        # -- caught by gate, documented). UNKNOWN layouts skip (gradual).
        for a in args:
            if _is_literal(a) or a not in layouts:
                continue
            lay = layouts[a]
            if lay == _UNKNOWN or ":" not in lay:
                continue
            want = lay.split(":")[0]
            got = _kind_of(feeds[a]) if a in feeds else _UNKNOWN
            if got != _UNKNOWN and got != want:
                raise AsmError(f"line {ln} ({mn}): stream '{a}' declared {lay} but holds {got}")
        if trace is not None:
            import time
            in_sum = {}
            for a in args:
                ssum = _summarize(feeds[a]) if a in feeds else {"kind": "literal", "value": a}
                ssum["layout"] = layouts.get(a, "F:SCALAR" if _is_literal(a) else _UNKNOWN)
                in_sum[a] = ssum
            trace.append({"line": ln, "op": mn,
                          "in": in_sum,
                          "out": {}, "sec": 0.0, "_t0": time.perf_counter()})
        _check_shapes(mn, vals, ln)
        out = fn(vals, config, feeds)
        # arity was checked at assemble time: one declared output takes the
        # whole return value (even a triples-tuple); N outputs take an N-tuple.
        out = (out,) if len(outs) == 1 else tuple(out)
        if len(out) != len(outs):
            raise AsmError(f"line {ln} ({mn}): runtime arity break: {outs}")
        for name, val in zip(outs, out):
            feeds[name] = val
            layouts[name] = out_lays[name]
        if trace is not None:
            rec = trace[-1]
            rec["out"] = {}
            for name, val in zip(outs, out):
                ssum = _summarize(val)
                ssum["layout"] = out_lays[name]
                rec["out"][name] = ssum
            rec["sec"] = time.perf_counter() - rec.pop("_t0")
    return feeds


def format_trace(log):
    """Human-readable trace lines: L12 SPLAT_BLUR A:T:HW(48,48) -> AS:... 0.213s."""
    lines = []
    for rec in log:
        def fmt(name, s):
            sh = "x".join(str(d) for d in s.get("shape", ())) or "?"
            lay = s.get("layout", "")
            extra = ""
            if "mean" in s:
                extra = f" ~{s['mean']:.3g}"
            elif "true" in s:
                extra = f" T{s['true']}/{s.get('n', '?')}"
            return f"{name}{(':' + lay) if lay else ''}({sh}){extra}"
        ins = " ".join(fmt(n, s) for n, s in rec["in"].items())
        outs = " ".join(fmt(n, s) for n, s in rec["out"].items())
        lines.append(f"L{rec['line']:>3} {rec['op']:<12} {ins} -> {outs}  {rec['sec']:.3f}s")
    total = sum(r["sec"] for r in log)
    lines.append(f"total {total:.3f}s over {len(log)} ops")
    return lines


def run_text(text, registry, payload, sigs=None, trace=None, basedir="."):
    """Parse + assemble + execute. Returns feeds (trace list filled if given)."""
    config, inp, bound, _ = assemble(text, registry, sigs, basedir)
    return run((config, inp, bound), config, inp, payload, trace=trace)


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


def _check_shapes(mn, vals, ln):
    """Per-op stream-geometry rules (run time: shapes are KNOWN here, unlike
    layouts which may stay UNKNOWN -- this pass needs no annotations).
    Strictness is deliberate: silent numpy broadcasting across streams hid
    real bugs (the (16,16)-vs-(16,3) class); every legitimate exception is
    an explicit rule below, not an accident. Rules keyed by mnemonic read
    positional stream values (literals skipped). Raises AsmError."""
    import numpy as _np

    def _sh(v):
        if isinstance(v, tuple) and len(v) == 3 and hasattr(v[0], "shape"):
            return tuple(v[0].shape)
        if hasattr(v, "shape"):
            return tuple(v.shape)
        return None

    def _same(*vs, what):
        have = [(i, _sh(v)) for i, v in enumerate(vs) if _sh(v) is not None]
        shapes = {s for _, s in have}
        if len(shapes) > 1:
            raise AsmError(
                f"line {ln} ({mn}): {what} shape mismatch "
                + ", ".join(f"arg{i}={s}" for i, s in have))

    if mn in ("ADD", "SUB", "MUL", "DIV"):
        _same(*vals, what="elementwise")
    elif mn == "SELECT":
        _same(vals[1], vals[2], what="SELECT branches")
        ms = _sh(vals[0])
        bs = _sh(vals[1])
        if ms is not None and bs is not None and ms != bs:
            raise AsmError(
                f"line {ln} ({mn}): mask {ms} vs branches {bs}")
    elif mn in ("MATMUL", "BATCH_MATMUL"):
        a, b = _sh(vals[0]), _sh(vals[1])
        if a is not None and b is not None:
            if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
                raise AsmError(
                    f"line {ln} ({mn}): inner dims {a} vs {b} (did you forget TRANSPOSE?)")
    elif mn == "WARP":
        f, fl = _sh(vals[0]), _sh(vals[1])
        if f is not None and fl is not None and f != fl[:2]:
            raise AsmError(
                f"line {ln} ({mn}): feature {f} vs flow {fl} spatial mismatch")
    elif mn == "ARGMAX":
        t = _sh(vals[0])
        if t is not None and not (0 <= int(float(vals[1])) < len(t)):
            raise AsmError(
                f"line {ln} ({mn}): axis {vals[1]} out of range for shape {t}")
    elif mn == "GATHER":
        w, ids = _sh(vals[0]), vals[1]
        if w is not None and hasattr(ids, "max"):
            import numpy as _nn
            ids_a = _nn.ascontiguousarray(ids)
            if ids_a.size and (int(ids_a.min()) < 0 or int(ids_a.max()) >= w[0]):
                raise AsmError(
                    f"line {ln} ({mn}): ids out of range [0,{w[0]})")


def repeat(text, registry, payload, n, sigs=None, grow=(), basedir="."):
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
    config, inp, bound, state = assemble(text, registry, sigs, basedir)
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
