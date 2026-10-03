"""Python frontend builder v0.1: geometric assembly without hand-writing it.

Prog accumulates IN/CONFIG/IMPORT/DEF/op/CALL lines and emits asm text
(or compiles straight through). Stream names are explicit (like the
listings) or auto (t1, t2, ...) when out=None. Literals pass through
as int/float args. The assembler remains the authority: builder output
is ordinary text, gated by assemble/run agreement, never trusted raw.

Composite helpers (the stdlib seed) belong here as plain functions
over Prog -- see attn_head() below. More arrive by writing them, not
by extending the class.
"""
from chain.asm import AsmError


class Prog:
    def __init__(self, name="<?>"):
        self.name = name
        self._config = []
        self._imports = []
        self._ins = []
        self._lines = []
        self._auto = 0

    # -- declarations ------------------------------------------------
    def config(self, k, v):
        self._config.append((k, v))
        return self

    def imp(self, path):
        self._imports.append(path)
        return self

    def inp(self, *names):
        for n in names:
            if n in self._ins:
                raise AsmError(f"builder: duplicate IN '{n}'")
            self._ins.append(n)
        return self

    # -- code ----------------------------------------------------------
    def _fresh(self, hint="t"):
        self._auto += 1
        return f"{hint}{self._auto}"

    @staticmethod
    def _render(a):
        if isinstance(a, bool):
            raise AsmError("builder: bool args refused (int/float/str only)")
        if isinstance(a, int):
            return str(a)
        if isinstance(a, float):
            return repr(a)
        if isinstance(a, str):
            return a
        raise AsmError(f"builder: bad arg {a!r} (want stream/int/float)")

    def op(self, mn, *args, out=None):
        o = out or self._fresh()
        self._lines.append(("op", o, mn, [self._render(a) for a in args]))
        return o

    def defn(self, name, formals, body):
        """DEF block: body is fn(sub: Prog) building DEF-local lines and
        returning the output stream name(s) (str or list). Formals become
        the DEF's INs."""
        sub = Prog(name=f"{self.name}:{name}")
        sub.inp(*formals)
        outs = body(sub)
        if isinstance(outs, str):
            outs = [outs]
        self._lines.append(("def", name, list(formals), sub._lines,
                            list(outs)))
        return self

    def call(self, name, *args, out=None, nout=1):
        o = out or (self._fresh() if nout == 1 else
                    [self._fresh() for _ in range(nout)])
        self._lines.append(("call", o, name, [self._render(a) for a in args]))
        return o

    # -- emission ------------------------------------------------------
    def text(self):
        L = []
        for k, v in self._config:
            L.append(f"CONFIG {k} {v}")
        for p in self._imports:
            L.append(f'IMPORT "{p}"')

        def emit(lines, ind, out):
            for ln in lines:
                if ln[0] == "op":
                    _, o, mn, args = ln
                    out.append(f"{'  ' * ind}{o} = {mn}({', '.join(args)})")
                elif ln[0] == "call":
                    _, o, name, args = ln
                    o_txt = o if isinstance(o, str) else \
                        f"({', '.join(o)})"
                    out.append(f"{'  ' * ind}{o_txt} = CALL {name}"
                               f"({', '.join(args)})")
                elif ln[0] == "def":
                    _, name, formals, body, outs = ln
                    out.append(f"{'  ' * ind}DEF {name}"
                               f"({', '.join(formals)}) -> "
                               f"({', '.join(outs)})")
                    emit(body, ind + 1, out)
                    out.append(f"{'  ' * ind}END")
        for n in self._ins:
            L.append(f"IN {n}")
        emit(self._lines, 0, L)
        return "\n".join(L) + "\n"

    def compile(self, target, sample=None, outputs=None, registry=None,
                sigs=None, basedir="."):
        """Straight to a backend (no text round-trip needed, but text()
        always shows what compiled). Origin marks generated code."""
        from chain.emit_c import compile_program
        return compile_program(self.text(), target, sample=sample,
                               outputs=outputs, registry=registry,
                               sigs=sigs, basedir=basedir,
                               origin=f"builder:{self.name}")


# -- stdlib seed: composites as plain functions ------------------------

def attn_head(p, x, wq, wk, wv, pos, cmask, out="CTX",
              scale_head2=False):
    """Single attention head over (S,D)->(S,Dh): QKV, RoPE, scores,
    head-temp, causal mask, shift+wide-softmax, context. Returns the
    context stream name. Matches lm_headt head structure (head2 gets
    TBETA scaling when scale_head2)."""
    q = p.op("MATMUL", x, wq)
    k = p.op("MATMUL", x, wk)
    v = p.op("MATMUL", x, wv)
    # NOTE: Dh slicing is the caller's job (SLICE before this call);
    # this composite takes head-narrow streams.
    qr = p.op("ROTARY", q, pos)
    kr = p.op("ROTARY", k, pos)
    kt = p.op("TRANSPOSE", kr)
    sc = p.op("BATCH_MATMUL", qr, kt)
    if scale_head2:
        ht = p.op("TBETA", sc)
        sc = p.op("MUL", sc, ht)
    neg = p.op("BETA", sc)
    ms = p.op("SELECT", cmask, sc, neg)
    s = p.op("TSHIFT", ms)
    pr = p.op("SOFTMAX_WIDE", s)
    return p.op("BATCH_MATMUL", pr, v, out=out)
