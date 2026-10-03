"""Pattern-table codegen core + C backend (v0.1, lean).

Shader model: one bound program (chain.asm.assemble IR) -> any backend.
A backend is an opcode->pattern table plus prologue/epilogue finishing
touches and a value model (SIGS layouts -> target types). Adding an
opcode = one registry row + one pattern per backend; anything missing
fails loud (NoPattern) -- never silently wrong.

v0.1 scope: GATHER + ARGMAX only (bigram_lm.asm proves the concept end
to end). Tier-0/1/2 patterns arrive as backends earn them; the interface
below is already the full one, so growth is additive.

Value model (C): I streams -> int64_t arrays; T triples -> three
parallel int arrays (s/e/z, dtypes from the sample); F streams -> NOT
YET (NoPattern -- float codegen is the next tier, not this spike).
Shapes come from a sample payload (specialization by example: frozen
dims V/C bake in as constants -- recompilation on shape change is
stated, not hidden; batch dim N stays dynamic). The emitted binary
links libc only: no phi-core, no numpy, no FPU on this path.
"""
import os
import subprocess

import numpy as np

from chain.asm import AsmError, assemble


class NoPattern(AsmError):
    """Opcode has no pattern for this backend (missing, not wrong)."""


class Backend:
    """Target interface. Subclass per backend; the driver is shared."""

    name = "base"

    def value_repr(self, layout):
        raise NotImplementedError

    def prologue(self, prog):
        raise NotImplementedError

    def pattern(self, mn, outs, args, sig, ctx):
        raise NoPattern(f"{self.name}: no pattern for {mn}")

    def epilogue(self, prog):
        raise NotImplementedError


_C_INT = {
    "int8": "int8_t", "int16": "int16_t", "int32": "int32_t",
    "int64": "int64_t", "uint8": "uint8_t", "uint16": "uint16_t",
    "uint32": "uint32_t", "uint64": "uint64_t",
}


def _c_type(dt):
    dt = np.dtype(dt).name
    if dt in _C_INT:
        return _C_INT[dt]
    if dt == "float32":
        return "float"
    if dt == "float64":
        return "double"
    raise AsmError(f"C backend: no C type for dtype {dt}")


def _kind_of_sample(v):
    if isinstance(v, tuple) and len(v) == 3:
        return "T"
    v = np.asanyarray(v)
    return "I" if np.issubdtype(v.dtype, np.integer) else "F"


def infer_shapes(bound, sample):
    """Static shape/dtype pass over bound IR using sample payload.

    Returns (streams, order) where streams[name] = (kind, sub, shape):
    kind I -> sub = C type string; kind T -> sub = (s,e,z C types).
    Only GATHER/ARGMAX rules exist in v0.1; anything else -> NoPattern.
    """
    _, inp, prog = bound
    in_names = [n for n, _ in inp]
    streams = {}
    for n in in_names:
        if n not in sample:
            raise AsmError(f"C backend: sample missing IN stream '{n}'")
        k = _kind_of_sample(sample[n])
        if k == "T":
            s, e, z = sample[n]
            streams[n] = ("T", (_c_type(s.dtype), _c_type(e.dtype),
                                _c_type(z.dtype)),
                          tuple(np.shape(s)))
        elif k == "I":
            a = np.ascontiguousarray(sample[n])
            streams[n] = ("I", _c_type(a.dtype), tuple(a.shape))
        else:
            raise NoPattern(f"C backend: float IN stream '{n}' not yet")

    def arg_stream(a):
        return a if a in streams else None

    for outs, mn, _fn, args, ln, _sig in prog:
        if mn == "GATHER":
            if len(args) != 2 or arg_stream(args[0]) is None \
                    or arg_stream(args[1]) is None:
                raise AsmError(f"line {ln} (GATHER): need (table, ids)")
            tk, tsub, tsh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            if tk != "T" or ik != "I" or len(tsh) != 2:
                raise AsmError(
                    f"line {ln} (GATHER): v0.1 needs T-table (V,C) + I ids,"
                    f" got {args[0]}:{tsh} {args[1]}:{ish}")
            n = ish[0] if len(ish) == 1 else 1 if len(ish) == 0 else None
            if n is None:
                raise AsmError(f"line {ln} (GATHER): ids must be (N,) or "
                               f"scalar, got {ish}")
            streams[outs[0]] = ("T", tsub, (n, tsh[1]))
        elif mn == "ARGMAX":
            if len(args) < 1 or arg_stream(args[0]) is None:
                raise AsmError(f"line {ln} (ARGMAX): need input stream")
            sk, ssub, ssh = streams[args[0]]
            ax = int(float(args[1])) if len(args) > 1 and \
                args[1] not in streams else -1
            if len(ssh) == 1:
                out_sh = ()
            elif len(ssh) == 2 and ax % 2 == 1:
                out_sh = (ssh[0],)
            else:
                raise AsmError(
                    f"line {ln} (ARGMAX): v0.1 handles 1D or 2D-axis-1,"
                    f" got shape {ssh} axis {ax}")
            streams[outs[0]] = ("I", "int64_t", out_sh)
        else:
            raise NoPattern(f"C backend: no pattern for {mn} (line {ln})")
    return streams


class CBackend(Backend):
    """C99 backend. Optimized patterns (backend owns its speed):

    - GATHER is zero-copy (row pointers, no memcpy) -- the table is
      already row-major; indexing IS the gather.
    - ARGMAX is a single pass keeping (class, key); ties keep first by
      strictly-greater replace (matches op_argmax bit-for-bit).
    - Frozen dims bake in (#define) so loops bound to constants; batch N
      stays dynamic. Tune via cflags (default -O2 -march=native).
    """

    name = "c"
    DEFAULT_CFLAGS = ("-O2", "-march=native", "-std=c99", "-Wall")

    def value_repr(self, layout):
        kind = (layout or "").split(":")[0]
        return {"I": "int64_t", "F": "double"}.get(kind, "void/*T*/")

    def prologue(self, ctx):
        L = ["#include <stdint.h>", "#include <stdio.h>",
             "#include <stdlib.h>", ""]
        for k, v in ctx["dims"].items():
            L.append(f"#define DIM_{k} {v}L")
        return "\n".join(L) + "\n"

    def pattern(self, mn, outs, args, sig, ctx):
        S = ctx["streams"]
        if mn == "GATHER":
            t, ids = args
            st, et, zt = S[t][1]
            o = outs[0]
            return (
                f"/* {o} = GATHER({t}, {ids}) -- zero-copy rows */\n"
                f"const {st} *g_rows_s[{ids}_N];\n"
                f"const {et} *g_rows_e[{ids}_N];\n"
                f"const {zt} *g_rows_z[{ids}_N];\n"
                f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {{\n"
                f"  g_rows_s[g_i] = {t}_s + {ids}[g_i] * DIM_C;\n"
                f"  g_rows_e[g_i] = {t}_e + {ids}[g_i] * DIM_C;\n"
                f"  g_rows_z[g_i] = {t}_z + {ids}[g_i] * DIM_C;\n"
                f"}}",
                o)
        if mn == "ARGMAX":
            src = args[0]
            sk = S[src][0]
            if sk != "T" or src not in ctx["gathered"]:
                raise AsmError(f"ARGMAX C pattern v0.1 needs a gathered T "
                               f"row-stream, got '{src}' ({sk})")
            st, _et, _zt = S[src][1]
            _ = st
            o = outs[0]
            return (
                f"/* {o} = ARGMAX({src}, 1) -- class rank + exponent, "
                f"ties first */\n"
                f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                f"  int64_t best = 0; int bcls = -1; int64_t bkey = 0;\n"
                f"  for (int64_t a_j = 0; a_j < DIM_C; ++a_j) {{\n"
                f"    {S[src][1][0]} s = g_rows_s[a_i][a_j];\n"
                f"    {S[src][1][1]} e = g_rows_e[a_i][a_j];\n"
                f"    {S[src][1][2]} z = g_rows_z[a_i][a_j];\n"
                f"    int cls = (!z && s > 0) ? 2 : (z ? 1 : 0);\n"
                f"    int64_t key = (!z && s > 0) ? (int64_t)e : "
                f"(z ? 0 : -(int64_t)e);\n"
                f"    if (bcls < 0 || cls > bcls || "
                f"(cls == bcls && key > bkey)) {{\n"
                f"      best = a_j; bcls = cls; bkey = key;\n"
                f"    }}\n"
                f"  }}\n"
                f"  {o}[a_i] = best;\n"
                f"}}",
                None)
        nop = super().pattern(mn, outs, args, sig, ctx)
        return nop, None

    def epilogue(self, ctx):
        return ""


def compile_program(text, target="c", sample=None, outputs=None,
                    registry=None, sigs=None, basedir="."):
    """Frontend entry: asm text -> target source. Returns dict with
    source/backend/streams/dims/config. Sample payload required (shapes).
    """
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    config, inp, bound, _ = assemble(text, registry or _R, sigs or _S,
                                     basedir=basedir)
    if target != "c":
        raise NoPattern(f"no '{target}' backend yet (have: c, lattice-ref)")
    if sample is None:
        raise AsmError("compile_program needs sample payload (shapes)")
    if not outputs:
        raise AsmError("compile_program needs outputs=[...] (OUT streams)")
    be = CBackend()
    streams = infer_shapes((config, inp, bound), sample)
    in_names = [n for n, _ in inp]
    # dims: frozen 2D-table dims bake in; batch N dynamic per run.
    dims, dyn = {}, {}
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "T" and len(sh) == 2:
            dims["V"], dims["C"] = sh
    for o in outputs:
        k, sub, sh = streams[o]
        n_dyn = sh[0] if len(sh) == 1 else 1
        dyn[o] = n_dyn
    ctx = {"streams": streams, "dims": dims, "dyn": dyn,
           "inputs": in_names, "outputs": outputs, "gathered": set()}
    parts = [be.prologue(ctx)]
    parts.append("/* inputs: argv files, order below */")
    # input declarations: tables as malloc'd globals, ids dynamic
    decls = []
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            decls.append(f"static {st} *{n}_s; static {et} *{n}_e;"
                         f" static {zt} *{n}_z;")
        else:
            decls.append(f"static int64_t *{n}; static int64_t {n}_N;")
    for o in outputs:
        decls.append(f"static int64_t *{o}; static int64_t {o}_N;")
    parts.append("\n".join(decls))
    parts.append("int main(int argc, char **argv) {")
    # argv: for each IN: file(s); then out file per output.
    # T table -> 3 files; I -> 1 file. Order = inputs order, then outputs.
    n_argv = sum(3 if streams[n][0] == "T" else 1 for n in in_names)
    n_argv += len(outputs)
    parts.append(f"  if (argc != {n_argv + 1}) return 9;")
    idx = 1
    loads = []
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            loads.append(f"  /* load {n} (V x C triples) */")
            for comp, ct in (("s", st), ("e", et), ("z", zt)):
                loads.append(
                    f"  {{ FILE *f = fopen(argv[{idx}], \"rb\");"
                    f" if (!f) return 1;\n"
                    f"    {n}_{comp} = malloc(DIM_V * DIM_C * sizeof({ct}));"
                    f" if (fread({n}_{comp}, sizeof({ct}), DIM_V * DIM_C, f)"
                    f" != (size_t)(DIM_V * DIM_C)) return 2; fclose(f); }}")
                idx += 1
        else:
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); {n}_N = ftell(f)/sizeof(int64_t);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n} = malloc({n}_N * sizeof(int64_t));"
                f" if (fread({n}, 8, {n}_N, f) != (size_t){n}_N) return 2;"
                f" fclose(f); }}")
            idx += 1
    for o in outputs:
        loads.append(f"  {o}_N = {ctx['inputs'][0]}_N; "
                     f"{o} = malloc({o}_N * 8);")
    parts.append("\n".join(loads))
    _, _, prog = (config, inp, bound)
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"].add(gathered)
        first_in = ctx["inputs"][0]
        pat = pat.replace(f"{outs[0]}_N", f"{first_in}_N")
        parts.append(f"  /* L{ln}: {mn} */")
        parts.append("  " + pat.replace("\n", "\n  "))
    for o in outputs:
        parts.append(f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                     f" return 3;\n"
                     f"    fwrite({o}, 8, {o}_N, f); fclose(f); }}")
        idx += 1
    parts.append("  return 0;\n}")
    parts.append(be.epilogue(ctx))
    return {"source": "\n".join(parts), "backend": be, "streams": streams,
            "dims": dims, "config": config, "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1}


def build(source, workdir, name="prog", cc="cc", cflags=None):
    """Compile emitted source -> exe path. Backend owns its speed flags."""
    cflags = list(cflags or CBackend.DEFAULT_CFLAGS)
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".c")
    exe = os.path.join(workdir, name)
    with open(src, "w") as f:
        f.write(source)
    r = subprocess.run([cc] + cflags + [src, "-o", exe],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"C build failed:\n{r.stderr}")
    return exe
