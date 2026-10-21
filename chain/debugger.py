"""Listing debugger: breakpoints, watch, step, lesion tables (CPU lattice).

Session drives the EXISTING op loop (`chain.asm.run`) two ways --
never reimplementing execution, never rewriting listings:
- stepping / breakpoints / watch: a store_hook routes per-op writes
  to debugger-owned env (ChainMap env-first reads execute identical
  values -- the hook's proven contract) and raises a private _Pause
  at the stop op; each step is a fresh bounded run (O(n^2), stated --
  the debugger is for small listings; served/CUDA stepping is out
  of scope, not smuggled).
- lesions: the EXISTING overrides mechanism (v1.3 probes, built for
  exactly this) with zeros_like OUTs; deltas vs the reference run.
Read-only w.r.t. listings (assemble + run only). Deterministic:
same session twice -> identical tables (byte-digests; flagship
24x32 verified 3/3 bit-identical pre-round).
"""
import hashlib

from chain import asm as ASM
from chain import census as CENSUS


class DebuggerError(Exception):
    """Debugger misuse (bad index/stream/op -- never silent)."""


class _Pause(Exception):
    """Private control flow: stop the op loop at the target op."""


MISSING = "<unproduced>"


def _short(ln):
    return str(ln)


def _zeros_like(v):
    """Zeros matching a feed value (per-component for triples tuples)."""
    import numpy as np
    if isinstance(v, tuple):
        return tuple(_zeros_like(x) for x in v)
    if isinstance(v, np.ndarray):
        return np.zeros_like(v)
    raise DebuggerError(f"cannot zero exotic feed value {type(v)}")


def _as_float(a):
    import numpy as np
    return np.ascontiguousarray(a, dtype=np.float64)


def digest(v):
    """Deterministic digest of a feed value (strict table compare)."""
    import numpy as np
    if isinstance(v, tuple):
        return "(" + ",".join(digest(x) for x in v) + ")"
    if isinstance(v, np.ndarray):
        a = np.ascontiguousarray(v)
        h = hashlib.sha256(a.tobytes()).hexdigest()[:16]
        return f"{a.dtype}{list(a.shape)}:{h}"
    if isinstance(v, float):
        return repr(v)
    return f"{type(v).__name__}:{v!r}"


class _Hook:
    """store_hook that snapshots env and pauses at stop_at (ops executed
    before the stop are fully written; the stop op is not executed)."""

    def __init__(self, stop_at):
        self.env = {}
        self.stop_at = stop_at
        self.count = 0

    def dest(self, outs, ln):
        if self.count >= self.stop_at:
            raise _Pause()
        self.count += 1
        return True


class Session:
    """A debugging session over one assembled listing + payload."""

    def __init__(self, text, registry, payload, sigs=None, basedir=".",
                 stdlib=None):
        self.text = text
        self.registry = registry
        self.payload = payload
        self.sigs = sigs
        self.basedir = basedir
        self.stdlib = stdlib
        config, inp, bound, _ = ASM.assemble(text, registry, sigs,
                                             basedir, stdlib)
        self._runargs = (config, inp, bound)
        self.prog = bound
        self.n_ops = len(bound)
        rep = CENSUS.census_text(text, registry, sigs=sigs,
                                 basedir=basedir, stdlib=stdlib)
        self.streams = rep["streams"]
        self.sinks = sorted(n for n, s in rep["streams"].items()
                            if s["producer"] is not None
                            and s["fanout"] == 0)
        self.pos = 0
        self.breaks = set()
        self.watched = []
        self._ref = None

    # -- setup ------------------------------------------------------
    def break_at(self, indices):
        for i in indices:
            if not isinstance(i, int) or not 0 <= i <= self.n_ops:
                raise DebuggerError(
                    f"bad breakpoint {i!r} (want 0..{self.n_ops})")
        self.breaks = set(indices)
        return sorted(self.breaks)

    def watch(self, names):
        for n in names:
            if n not in self.streams:
                raise DebuggerError(
                    f"bad watch stream {n!r} (not produced/consumed; "
                    f"CONFIG is not watchable)")
        self.watched = list(names)
        return list(self.watched)

    def reset(self):
        self.pos = 0
        return 0

    # -- execution --------------------------------------------------
    def _bounded(self, stop_at):
        """Run ops [0, stop_at); returns the full feed snapshot. IN
        streams are pre-seeded (mirroring run()'s payload rule) so
        paused runs see inputs; hook-env writes then overwrite in
        order, so shadowed streams read latest."""
        config, inp, bound = self._runargs
        hook = _Hook(stop_at)
        names = inp if not isinstance(inp, str) else [(inp, None)]
        in_names = [n for n, _ in names]
        if len(in_names) == 1 and not isinstance(self.payload, dict):
            hook.env[in_names[0]] = self.payload
        else:
            for n in in_names:
                hook.env[n] = self.payload[n]
        hook.env["CONFIG"] = config
        try:
            ASM.run((config, inp, bound), config, inp,
                    self.payload, trace=None, overrides=None,
                    store_hook=hook)
        except _Pause:
            pass
        return dict(hook.env)

    def step(self, n=1):
        """Advance n ops unless a breakpoint hits first. Returns the
        stop record {pos, op, line, watched}."""
        if n < 1:
            raise DebuggerError(f"step needs n >= 1, got {n}")
        cands = [b for b in self.breaks if self.pos < b <= self.pos + n]
        stop_at = min(cands + [self.pos + n, self.n_ops])
        env = self._bounded(stop_at)
        self.pos = stop_at
        if self.pos < self.n_ops:
            outs, mn, _, _, ln, _ = self.prog[self.pos]
            nxt = {"op": mn, "line": _short(ln)}
        else:
            nxt = {"op": None, "line": None}
        return {"pos": self.pos, "next": nxt,
                "watched": {w: env.get(w, MISSING) for w in self.watched}}

    def run_full(self):
        """Reference full run (plain run_text, no hooks)."""
        if self._ref is None:
            self._ref = ASM.run_text(
                self.text, self.registry, self.payload, sigs=self.sigs,
                trace=None, basedir=self.basedir, stdlib=self.stdlib)
        return self._ref

    # -- lesions ----------------------------------------------------
    def lesion(self, op_idx):
        """Zero the op's OUTs (overrides) and diff final OUTs vs the
        reference run. Returns {op_idx, op, line, deltas, worst}."""
        import numpy as np
        if not isinstance(op_idx, int) or not 0 <= op_idx < self.n_ops:
            raise DebuggerError(
                f"bad lesion op {op_idx!r} (want 0..{self.n_ops - 1})")
        ref = self.run_full()
        outs, mn, _, _, ln, _ = self.prog[op_idx]
        if not outs:
            raise DebuggerError(f"op {op_idx} ({mn}) has no OUTs to lesion")
        ov = {o: _zeros_like(ref[o]) for o in outs}
        got = ASM.run_text(self.text, self.registry, self.payload,
                           sigs=self.sigs, trace=None,
                           basedir=self.basedir, stdlib=self.stdlib,
                           overrides=ov)
        deltas = {}
        for s in self.sinks:
            d = _as_float(got[s]).reshape(-1) - _as_float(ref[s]).reshape(-1)
            deltas[s] = {"maxabs": float(np.abs(d).max()),
                         "mean": float(np.abs(d).mean())}
        worst = max(v["maxabs"] for v in deltas.values())
        return {"op_idx": op_idx, "op": mn, "line": _short(ln),
                "outs": list(outs), "deltas": deltas, "worst": worst}

    def control(self):
        """Zero-op control: empty overrides must read all-zero deltas."""
        import numpy as np
        ref = self.run_full()
        got = ASM.run_text(self.text, self.registry, self.payload,
                           sigs=self.sigs, trace=None,
                           basedir=self.basedir, stdlib=self.stdlib,
                           overrides={})
        deltas = {}
        for s in self.sinks:
            d = _as_float(got[s]).reshape(-1) - _as_float(ref[s]).reshape(-1)
            deltas[s] = {"maxabs": float(np.abs(d).max()),
                         "mean": float(np.abs(d).mean())}
        return {"op_idx": None, "op": "CONTROL", "deltas": deltas}

    def lesion_table(self):
        """One lesion row per op, program order."""
        return [self.lesion(i) for i in range(self.n_ops)]

    # -- diff -------------------------------------------------------
    def diff(self, other):
        """Run-vs-run byte-compare: first op whose OUTs differ (exact
        array equality incl. dtype/shape), or first=None if identical."""
        import numpy as np

        def same(a, b):
            if isinstance(a, tuple) or isinstance(b, tuple):
                return (isinstance(a, tuple) and isinstance(b, tuple)
                        and len(a) == len(b)
                        and all(same(x, y) for x, y in zip(a, b)))
            aa, bb = np.ascontiguousarray(a), np.ascontiguousarray(b)
            return aa.dtype == bb.dtype and aa.shape == bb.shape and bool(
                (aa == bb).all())

        for i in range(min(self.n_ops, other.n_ops)):
            outs, mn, _, _, ln, _ = self.prog[i]
            ea = self._bounded(i + 1)
            eb = other._bounded(i + 1)
            for o in outs:
                if o not in eb:
                    return {"first": i, "op": mn, "line": _short(ln),
                            "stream": o, "note": "absent in other"}
                if not same(ea[o], eb[o]):
                    d = (_as_float(ea[o]).reshape(-1)
                         - _as_float(eb[o]).reshape(-1))
                    return {"first": i, "op": mn, "line": _short(ln),
                            "stream": o,
                            "maxabs": float(np.abs(d).max())}
        if self.n_ops != other.n_ops:
            return {"first": min(self.n_ops, other.n_ops),
                    "note": "op-count differs"}
        return {"first": None}

    # -- tables -----------------------------------------------------
    def stop_log(self):
        """Deterministic stop log: step through all ops watching."""
        self.reset()
        log = []
        while self.pos < self.n_ops:
            rec = self.step(1)
            _, mn, _, _, ln, _ = self.prog[rec["pos"] - 1]
            log.append({"pos": rec["pos"], "op": mn, "line": _short(ln),
                        "watched": {w: (digest(v) if v is not MISSING
                                        else MISSING)
                                    for w, v in rec["watched"].items()}})
        return log


def format_lesion_table(rows):
    """Human-readable lesion rows (spec evidence paste)."""
    L = ["op_idx op           line  outs            worst_maxabs "
         "sink:maxabs,mean..."]
    for r in rows:
        det = " ".join(f"{s}:{v['maxabs']:.4g},{v['mean']:.4g}"
                       for s, v in r["deltas"].items())
        L.append(f"{r['op_idx']:>6d} {r['op']:<12} {r['line']:<5} "
                 f"{','.join(r['outs']):<15} {r['worst']:<12.4g} {det}")
    return L


# -- torch lesions ----------------------------------------------------
# Same semantics as listing lesions: zero module outputs (forward
# hooks, zeros_like — shapes/dtypes preserved, only values change),
# rerun, diff final outputs vs the reference run. CPU, no_grad,
# eval mode throughout: no training, no weight writes, ever.


def _torch_zeros(out):
    import torch
    if isinstance(out, tuple):
        return tuple(_torch_zeros(x) for x in out)
    if isinstance(out, torch.Tensor):
        return torch.zeros_like(out)
    raise DebuggerError(f"cannot zero exotic module output {type(out)}")


class TorchSession:
    """Lesions over a fixed torch net + reference run().

    net: torch.nn.Module (set to eval here; weights never written).
    run: () -> {name: Tensor}, deterministic, side-effect free.
    leaves: lesionable module paths (default: all LEAF modules =
    no children, root excluded).
    """

    def __init__(self, net, run, leaves=None):
        import torch
        self.net = net
        self.net.eval()
        self._run = run
        if leaves is None:
            leaves = sorted(n for n, m in net.named_modules()
                            if n and not list(m.children()))
        self.leaves = list(leaves)
        for p in self.leaves:
            try:
                self.net.get_submodule(p)
            except AttributeError:
                raise DebuggerError(f"bad lesion module {p!r}")
        self._ref = None

    def reference(self):
        import torch
        if self._ref is None:
            with torch.no_grad():
                self._ref = {k: v.detach().clone() for k, v in
                             self._run().items()}
        return self._ref

    def lesion(self, path):
        """Zero the module's outputs (hook) and diff final outputs.
        Rows carry the hook fire count -- zero fires means the module
        was never called on this path (bypassed/dead weight), NOT a
        measured zero: the overrides-analogue tripwire (a lesion that
        never bites is green wallpaper, refused as a finding)."""
        import numpy as np
        import torch
        if path not in self.leaves:
            raise DebuggerError(
                f"bad lesion module {path!r} (want one of {self.leaves})")
        ref = self.reference()
        mod = self.net.get_submodule(path)
        fires = [0]

        def _hook(m, i, o):
            fires[0] += 1
            return _torch_zeros(o)

        handle = mod.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                got = {k: v.detach().clone() for k, v in
                       self._run().items()}
        finally:
            handle.remove()
        deltas = {}
        for s, rv in ref.items():
            d = (got[s].double().reshape(-1)
                 - rv.double().reshape(-1)).numpy()
            deltas[s] = {"maxabs": float(np.abs(d).max()),
                         "mean": float(np.abs(d).mean())}
        worst = max(v["maxabs"] for v in deltas.values())
        return {"module": path, "fires": fires[0], "deltas": deltas,
                "worst": worst}

    def control(self):
        """Zero-module control: no hooks must read exactly zero."""
        import numpy as np
        import torch
        ref = self.reference()
        with torch.no_grad():
            got = {k: v.detach().clone() for k, v in
                   self._run().items()}
        deltas = {}
        for s, rv in ref.items():
            d = (got[s].double().reshape(-1)
                 - rv.double().reshape(-1)).numpy()
            deltas[s] = {"maxabs": float(np.abs(d).max()),
                         "mean": float(np.abs(d).mean())}
        return {"module": None, "deltas": deltas}

    def lesion_table(self):
        """One lesion row per leaf module, leaves order."""
        return [self.lesion(p) for p in self.leaves]


def format_torch_table(rows):
    """Human-readable torch lesion rows (spec evidence paste)."""
    L = ["module          fires worst_maxabs out:maxabs,mean..."]
    for r in rows:
        det = " ".join(f"{s}:{v['maxabs']:.4g},{v['mean']:.4g}"
                       for s, v in r["deltas"].items())
        L.append(f"{r['module']:<15} {r.get('fires', '?'):>5} "
                 f"{r['worst']:<12.4g} {det}")
    return L
