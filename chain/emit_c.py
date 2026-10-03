"""Pattern-table codegen core + C backend (v0.2).

Shader model: one bound program (chain.asm.assemble IR) -> any backend.
A backend is an opcode->pattern table plus prologue/epilogue finishing
touches and a value model (SIGS layouts -> target types). Adding an
opcode = one registry row + one shape rule + one pattern per backend;
anything missing fails loud (NoPattern) -- never silently wrong.

Agreement classes (stated, per-path):
  - int paths (I/T streams, GATHER/ARGMAX/SELECT-masks): BIT-EXACT.
  - float paths (F streams, decoded weights): EPSILON. The lattice
    computes in fixed-point (m_acc/m_cov); the C backend computes the
    same real-valued functions in float64. The gate is CALIBRATED
    (first measurement x safety, regression guard) -- not preregistered,
    labeled honestly in the test.
Value model (C): I -> int64_t; T triples -> parallel int arrays;
F -> double. Data-prep (Python side) decodes T inputs to F64 doubles;
the emitted binary links libc+libm only.

v0.2 patterns: GATHER (T->T exact + F->F rows), ARGMAX (T rows + F),
RMSNORM, MATMUL, BATCH_MATMUL (2D/3D), SLICE (2D ax0/1), ROTARY,
TRANSPOSE (2D), TBETA/BETA, MUL/ADD (same-shape F), SELECT (F),
TSHIFT, SOFTMAX_WIDE, CONCAT (2D). Speed hooks (backend owns them):
ikj matmul + restrict + calloc-zeroed outputs, zero-copy gathers,
baked frozen dims, -O2 -march=native default; arena/BLAS are listed
finishing touches, not this cut.
"""
import math
import os
import subprocess

import numpy as np

from chain.asm import AsmError, assemble


class NoPattern(AsmError):
    """Opcode has no pattern for this backend (missing, not wrong)."""


def _short_site(ln):
    """Condense an origin string for emitted comments: basename the file
    part, keep :line and via-suffixes. Errors keep full sites."""
    import os as _os
    parts = str(ln).split(" via ")
    head = parts[0]
    if ":" in head:
        f, rest = head.rsplit(":", 1)
        head = f"{_os.path.basename(f) or f}:{rest}"
    return " via ".join([head] + parts[1:])


class Backend:
    """Target interface. Subclass per backend; the driver is shared."""

    name = "base"
    emits = False  # set True when pattern() covers a compilable set

    def value_repr(self, layout):
        raise NotImplementedError

    def prologue(self, ctx):
        raise NotImplementedError

    def pattern(self, mn, outs, args, sig, ctx):
        raise NoPattern(f"{self.name}: no pattern for {mn}")

    def epilogue(self, ctx):
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


def _int_lit(a, what):
    try:
        f = float(a)
    except (TypeError, ValueError):
        raise AsmError(f"{what}: need integral literal, got {a!r}")
    if not f.is_integer():
        raise AsmError(f"{what}: need integral literal, got {a!r}")
    return int(f)


def infer_shapes(bound, sample):
    """Static shape/dtype pass over bound IR using sample payload.

    streams[name] = (kind, sub, shape): kind I -> sub=C type; T -> sub=
    (s,e,z C types); F -> sub="double". Shape dims are ints (frozen) or
    None (batch-dynamic, from a GATHER ids stream). Rules exist only for
    v0.2 opcodes; anything else -> NoPattern.
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
        elif k == "F":
            a = np.ascontiguousarray(sample[n], dtype=np.float64)
            streams[n] = ("F", "double", tuple(a.shape))

    def S(a):
        return streams.get(a)

    def dyn1(ish):
        if len(ish) == 1:
            return None
        if len(ish) == 0:
            return 1
        return "BAD"

    for outs, mn, _fn, args, ln, _sig in prog:
        o = outs[0] if outs else None
        where = f"line {ln} ({mn})"
        if mn == "GATHER":
            tk, tsub, tsh = streams[args[0]]
            ik, _, ish = streams[args[1]]
            n = dyn1(ish)
            if n == "BAD" or ik != "I" or len(tsh) != 2:
                raise AsmError(f"{where}: v0.2 needs table (V,C) + "
                               f"(N,)/scalar ids")
            if tk == "T":
                streams[o] = ("T", tsub, (n, tsh[1]))
            elif tk == "F":
                streams[o] = ("F", "double", (n, tsh[1]))
            else:
                raise AsmError(f"{where}: GATHER table must be T/F")
        elif mn == "ARGMAX":
            sk, _, ssh = streams[args[0]]
            ax = _int_lit(args[1], f"{where} axis") if len(args) > 1 \
                and args[1] not in streams else -1
            if sk not in ("T", "F", "I"):
                raise AsmError(f"{where}: ARGMAX needs T/F/I input")
            if len(ssh) == 1:
                out_sh = ()
            elif len(ssh) == 2 and ax % 2 == 1:
                out_sh = (ssh[0],)
            else:
                raise AsmError(f"{where}: v0.2 ARGMAX handles 1D or "
                               f"2D-axis-1, got {ssh} ax {ax}")
            streams[o] = ("I", "int64_t", out_sh)
        elif mn == "RMSNORM":
            xk, xsub, xsh = streams[args[0]]
            wk, _, wsh = streams[args[1]]
            if xk not in ("F", "T") or xk != wk or len(xsh) < 1 \
                    or tuple(wsh) != (xsh[-1],):
                raise NoPattern(f"{where}: RMSNORM needs same-kind "
                                "x(...,D)+w(D,)")
            streams[o] = (xk, xsub, xsh)
        elif mn in ("MATMUL", "BATCH_MATMUL"):
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak not in ("F", "T") or ak != bk:
                raise NoPattern(f"{where}: {mn} needs same-kind inputs")
            tsub = asub if ak == "T" else "double"
            if len(ash) == 2 and len(bsh) == 2 and ash[1] == bsh[0]:
                streams[o] = (ak, tsub, (ash[0], bsh[1]))
            elif len(ash) == 3 and len(bsh) == 3 \
                    and (ash[0] == bsh[0] or bsh[0] == 1) \
                    and ash[2] == bsh[1]:
                streams[o] = (ak, tsub, (ash[0], ash[1], bsh[2]))
            else:
                raise AsmError(f"{where}: {mn} shape mismatch {ash}/{bsh}")
        elif mn == "SLICE":
            sk, ssub, ssh = streams[args[0]]
            ax = _int_lit(args[1], f"{where} axis")
            lo = _int_lit(args[2], f"{where} lo")
            hi = _int_lit(args[3], f"{where} hi")
            if sk not in ("F", "T") or len(ssh) != 2 or ax not in (0, 1):
                raise NoPattern(f"{where}: v0.2 SLICE handles 2D ax0/1 F/T")
            dd = list(ssh)
            if dd[ax] is not None and not (0 <= lo <= hi <= dd[ax]):
                raise AsmError(f"{where}: SLICE [{lo},{hi}) of dim {dd[ax]}")
            dd[ax] = hi - lo
            streams[o] = (sk, ssub, tuple(dd))
        elif mn == "ROTARY":
            xk, xsub, xsh = streams[args[0]]
            pk, _, psh = streams[args[1]]
            if xk not in ("F", "T") or len(xsh) != 2 or xsh[1] is None \
                    or xsh[1] % 2 or pk != "I" or len(psh) != 1:
                raise NoPattern(f"{where}: ROTARY needs F/T (S,Deven) + "
                                "I pos (S,)")
            if xsh[0] is not None and tuple(psh) != (xsh[0],):
                raise AsmError(f"{where}: ROTARY pos len {psh} vs rows "
                               f"{xsh[0]}")
            streams[o] = (xk, xsub, xsh)
        elif mn == "TRANSPOSE":
            sk, _, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: v0.2 TRANSPOSE handles 2D F/T")
            streams[o] = (sk, streams[args[0]][1], (ssh[1], ssh[0]))
        elif mn in ("TBETA", "BETA"):
            # ref is PURELY geometric (constant fill takes its shape;
            # values never read -- op_beta doctrine). F/T refs preserve
            # kind; I refs (masks) yield F fills (nonfpu callers pass T).
            rk, rsub, rsh = streams[args[0]]
            if rk == "I":
                streams[o] = ("F", "double", rsh)
            else:
                streams[o] = (rk, rsub, rsh)
        elif mn in ("MUL", "ADD", "SUB"):
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            if ak not in ("F", "T") or ak != bk \
                    or tuple(ash) != tuple(bsh):
                raise NoPattern(f"{where}: {mn} needs same-kind "
                                "same-shape (no broadcast in v0.2)")
            streams[o] = (ak, asub, ash)
        elif mn == "SQUARE":
            sk, ssub, ssh = streams[args[0]]
            if sk != "T":
                raise NoPattern(f"{where}: SQUARE needs T (integer "
                                "exactness lives in triples)")
            streams[o] = ("T", ssub, ssh)
        elif mn == "SELECT":
            mk, _, msh = streams[args[0]]
            ak, asub, ash = streams[args[1]]
            bk, _, bsh = streams[args[2]]
            if mk != "I" or ak not in ("F", "T") or ak != bk \
                    or tuple(ash) != tuple(bsh):
                raise NoPattern(f"{where}: SELECT needs I mask + same-kind "
                                "same-shape branches (v0.2)")
            # mask may be baked while branches are batch-dynamic: allow
            # axis-wise (equal OR branch-dynamic), guarded at runtime.
            for dm, da in zip(msh, ash):
                if dm != da and da is not None:
                    raise AsmError(f"{where}: SELECT mask {msh} vs {ash}")
            if len(msh) != len(ash):
                raise AsmError(f"{where}: SELECT rank mismatch")
            streams[o] = (ak, streams[args[1]][1], ash)
        elif mn in ("TSHIFT", "SOFTMAX_WIDE"):
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: {mn} needs 2D F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "SILU":
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T"):
                raise NoPattern(f"{where}: SILU needs F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "SOFTMAX":
            sk, ssub, ssh = streams[args[0]]
            if sk not in ("F", "T") or len(ssh) != 2:
                raise NoPattern(f"{where}: SOFTMAX needs 2D F/T")
            streams[o] = (sk, ssub, ssh)
        elif mn == "CONCAT":
            ak, asub, ash = streams[args[0]]
            bk, _, bsh = streams[args[1]]
            ax = _int_lit(args[2], f"{where} axis") if args[2] not in \
                streams else None
            if ak not in ("F", "T") or ak != bk or len(ash) != 2 or ax != 1 \
                    or ash[0] != bsh[0] or ash[1] is None or bsh[1] is None:
                raise NoPattern(f"{where}: v0.2 CONCAT handles 2D ax1 "
                                "same-kind, same rows, baked widths")
            streams[o] = (ak, asub, (ash[0], ash[1] + bsh[1]))
        else:
            raise NoPattern(f"C backend: no pattern for {mn} ({where})")
    return streams


class CBackend(Backend):
    """C99 backend. Optimized patterns (backend owns its speed):

    - GATHER zero-copy (row pointers); MATMUL ikj + restrict over
      calloc-zeroed outputs; frozen dims bake in; batch dims dynamic.
    - ARGMAX-T replicates lattice class/exponent order bit-for-bit;
      ARGMAX-F is plain max-first (float path: eps-agreement class).
    - Tune via cflags (default -O2 -march=native) and BLK (matmul
      blocking hook, default off at these sizes -- finishing touch).
    """

    name = "c"
    emits = True
    use_blas = False
    DEFAULT_CFLAGS = ("-O2", "-march=native", "-std=c99", "-Wall")
    BLAS_LIBS = ("-lgslcblas",)

    def value_repr(self, layout):
        kind = (layout or "").split(":")[0]
        return {"I": "int64_t", "F": "double"}.get(kind, "void/*T*/")

    def prologue(self, ctx):
        L = ["#include <stdint.h>", "#include <stdio.h>",
             "#include <stdlib.h>", "#include <string.h>",
             "#include <math.h>", "#include <time.h>", ""]
        if getattr(self, "use_blas", False):
            L.append("#include <gsl/gsl_cblas.h>")
        for k, v in ctx["dims"].items():
            L.append(f"#define DIM_{k} {v}L")
        th = ctx.get("theta", [])
        if th:
            L.append("static const double THETA[] = {%s};" %
                     ", ".join(repr(t) for t in th))
        return "\n".join(L) + "\n"

    # -- helpers -----------------------------------------------------
    def _we(self, S, x, ctx, where):
        """Row-width expr of stream x: baked int, or the batch var when
        the last dim is dynamic (all dynamic dims derive from ids)."""
        w = S[x][2][-1]
        return str(int(w)) if w is not None else ctx["ids_n"]

    @staticmethod
    def _w(S, x, where):
        """Baked row-width of stream x (last dim); dynamic fails loud."""
        w = S[x][2][-1]
        if w is None:
            raise NoPattern(f"C backend: dynamic width in {where} "
                            f"(finishing touch, not this cut)")
        return int(w)

    # -- patterns ----------------------------------------------------
    def pattern(self, mn, outs, args, sig, ctx):
        S = ctx["streams"]
        cfg = ctx["config"]
        ids_n = ctx["ids_n"]
        if mn == "GATHER":
            t, ids = args
            tk = S[t][0]
            o = outs[0]
            if tk == "T":
                st, et, zt = S[t][1]
                w = self._we(S, t, ctx, f"GATHER {o}")
                return (
                    f"/* {o} = GATHER({t}, {ids}) materialized + row views */\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {{\n"
                    f"  memcpy(&{o}_s[g_i * {w}], &{t}_s[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_s[0]));\n"
                    f"  memcpy(&{o}_e[g_i * {w}], &{t}_e[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_e[0]));\n"
                    f"  memcpy(&{o}_z[g_i * {w}], &{t}_z[{ids}[g_i] * {w}], "
                    f"{w} * sizeof({o}_z[0]));\n"
                    f"}}\n"
                    f"const {st} *g_rows_s[{ids}_N];\n"
                    f"const {et} *g_rows_e[{ids}_N];\n"
                    f"const {zt} *g_rows_z[{ids}_N];\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i) {{\n"
                    f"  g_rows_s[g_i] = &{o}_s[g_i * {w}];\n"
                    f"  g_rows_e[g_i] = &{o}_e[g_i * {w}];\n"
                    f"  g_rows_z[g_i] = &{o}_z[g_i * {w}];\n"
                    f"}}",
                    (o, w))
            if tk == "F":
                w = self._we(S, t, ctx, f"GATHER {o}")
                return (
                    f"/* {o} = GATHER({t}, {ids}) materialized rows */\n"
                    f"for (int64_t g_i = 0; g_i < {ids}_N; ++g_i)\n"
                    f"  memcpy(&{o}[g_i * {w}], &{t}[{ids}[g_i] * {w}], "
                    f"{w} * sizeof(double));",
                    (o, w))
            raise NoPattern("GATHER table must be T/F (v0.2)")
        if mn == "ARGMAX":
            src = args[0]
            sk = S[src][0]
            o = outs[0]
            if sk == "T":
                if src not in ctx["gathered"]:
                    raise AsmError(f"ARGMAX C pattern v0.2 needs a "
                                   f"gathered T row-stream, got '{src}'")
                st, et, zt = S[src][1]
                w = ctx["gathered"][src]
                return (
                    f"/* {o} = ARGMAX({src}, 1) -- class rank + exponent, "
                    f"ties first */\n"
                    f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                    f"  int64_t best = 0; int bcls = -1; int64_t bkey = 0;\n"
                    f"  for (int64_t a_j = 0; a_j < {w}; ++a_j) {{\n"
                    f"    {st} s = g_rows_s[a_i][a_j];\n"
                    f"    {et} e = g_rows_e[a_i][a_j];\n"
                    f"    {zt} z = g_rows_z[a_i][a_j];\n"
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
            if sk == "F":
                if len(S[src][2]) != 2:
                    raise NoPattern("ARGMAX-F v0.2 needs 2D input")
                w = self._we(S, src, ctx, f"ARGMAX {o}")
                return (
                    f"/* {o} = ARGMAX({src}) float max-first */\n"
                    f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                    f"  int64_t best = 0; double bv = {src}[a_i * {w}];\n"
                    f"  for (int64_t a_j = 1; a_j < {w}; ++a_j)\n"
                    f"    if ({src}[a_i * {w} + a_j] > bv) {{\n"
                    f"      bv = {src}[a_i * {w} + a_j]; best = a_j;\n"
                    f"    }}\n"
                    f"  {o}[a_i] = best;\n"
                    f"}}",
                    None)
            raise NoPattern(f"ARGMAX needs T/F input, got {sk} (v0.2)")
        # -- F64 patterns (double *X streams, malloc'd per op) -------
        if mn == "RMSNORM":
            x, w = args
            o = outs[0]
            if S[x][0] != "F" or S[w][0] != "F":
                raise NoPattern("C backend RMSNORM needs F (T lives "
                                "on nonfpu)")
            eps = float(cfg.get("eps_rms", 0.0)) if "eps_rms" in cfg \
                else (_ for _ in ()).throw(
                    AsmError("RMSNORM C pattern needs CONFIG eps_rms "
                             "(true float; counts convention is lattice-only)"))
            d = self._we(S, x, ctx, f"RMSNORM {o}")
            return (
                f"/* {o} = RMSNORM({x}) eps={eps} */\n"
                f"for (int64_t r_i = 0; r_i < {o}_N; ++r_i) {{\n"
                f"  double ss = 0;\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    ss += {x}[r_i * {d} + r_j] * {x}[r_i * {d} + r_j];\n"
                f"  double rs = 1.0 / sqrt(ss / {d} + {eps!r});\n"
                f"  for (int64_t r_j = 0; r_j < {d}; ++r_j)\n"
                f"    {o}[r_i * {d} + r_j] = {x}[r_i * {d} + r_j] * rs "
                f"* {w}[r_j];\n"
                f"}}",
                None)
        if mn in ("MATMUL", "BATCH_MATMUL"):
            a, b = args
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"C backend {mn} needs F (T matmul "
                                "lives on nonfpu)")
            co = ctx["count"][o]
            ash, bsh = S[a][2], S[b][2]
            zero = f"memset({o}, 0, {co} * sizeof(double));"
            if len(ash) == 2:
                m, k, n = ash[0], ash[1], bsh[1]
                # Any dynamic dim is batch-derived (only GATHER makes
                # None); baked otherwise. Contraction over batch is legal.
                me = str(m) if m is not None else ids_n
                ke = str(k) if k is not None else ids_n
                ne = str(n) if n is not None else ids_n
                return (
                    f"/* {o} = {mn} ikj, restrict */\n{zero}\n"
                    f"for (int64_t m_i = 0; m_i < {me}; ++m_i)\n"
                    f"  for (int64_t m_k = 0; m_k < {ke}; ++m_k) {{\n"
                    f"    double aik = {a}[m_i * {ke} + m_k];\n"
                    f"    double *restrict crow = &{o}[m_i * {ne}];\n"
                    f"    const double *restrict brow = &{b}[m_k * {ne}];\n"
                    f"    for (int64_t m_j = 0; m_j < {ne}; ++m_j)\n"
                    f"      crow[m_j] += aik * brow[m_j];\n"
                    f"  }}",
                    None)
            B, m, k, n = ash[0], ash[1], ash[2], bsh[2]
            if B is None:
                raise NoPattern(f"C backend: {mn} needs baked batch "
                                f"dim, got {ash}/{bsh}")
            me = str(m) if m is not None else ids_n
            ke = str(k) if k is not None else ids_n
            ne = str(n) if n is not None else ids_n
            if getattr(self, "use_blas", False):
                return (
                    f"/* {o} = {mn} batched cblas_dgemm x{B} */\n"
                    f"for (int64_t m_b = 0; m_b < {B}; ++m_b)\n"
                    f"  cblas_dgemm(CblasRowMajor, CblasNoTrans, "
                    f"CblasNoTrans, {me}, {ne}, {ke}, 1.0,\n"
                    f"    &{a}[m_b * ({me}) * ({ke})], {ke},\n"
                    f"    &{b}[m_b * ({ke}) * ({ne})], {ne}, 0.0,\n"
                    f"    &{o}[m_b * ({me}) * ({ne})], {ne});",
                    None)
            return (
                f"/* {o} = {mn} batched ikj */\n{zero}\n"
                f"for (int64_t m_b = 0; m_b < {B}; ++m_b)\n"
                f"  for (int64_t m_i = 0; m_i < {me}; ++m_i)\n"
                f"    for (int64_t m_k = 0; m_k < {ke}; ++m_k) {{\n"
                f"      double aik = {a}[(m_b * {me} + m_i) * {ke} + m_k];\n"
                f"      double *restrict crow = &{o}[(m_b * {me} + m_i)"
                f" * {ne}];\n"
                f"      const double *restrict brow = &{b}[(m_b * {ke} + m_k)"
                f" * {ne}];\n"
                f"      for (int64_t m_j = 0; m_j < {ne}; ++m_j)\n"
                f"        crow[m_j] += aik * brow[m_j];\n"
                f"    }}",
                None)
        if mn == "SLICE":
            x, ax, lo, hi = args[0], _int_lit(args[1], "SLICE ax"), \
                _int_lit(args[2], "SLICE lo"), _int_lit(args[3], "SLICE hi")
            o = outs[0]
            co = ctx["count"][o]
            in_d = S[x][2][ax]
            if in_d is None:
                raise NoPattern("C backend: SLICE with dynamic stride")
            if S[x][0] == "T":
                st, et, zt = S[x][1]
                if ax == 1:
                    L = [f"/* {o} = SLICE({x},1,{lo},{hi}) triples, exact */"]
                    for comp in ("s", "e", "z"):
                        L.append(
                            f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                            f"  memcpy(&{o}_{comp}[s_i * {hi - lo}], "
                            f"&{x}_{comp}[s_i * {in_d} + {lo}], "
                            f"{hi - lo} * sizeof({o}_{comp}[0]));")
                    return ("\n".join(L), None)
                w1 = S[x][2][1]
                if w1 is None:
                    raise NoPattern("C backend: SLICE-T ax0 dynamic width")
                co = ctx["count"][o]
                L = [f"/* {o} = SLICE({x},0,{lo},{hi}) triples, exact */"]
                for comp in ("s", "e", "z"):
                    L.append(
                        f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                        f"  memcpy(&{o}_{comp}[s_i * {w1}], "
                        f"&{x}_{comp}[(s_i + {lo}) * {w1}], "
                        f"{w1} * sizeof({o}_{comp}[0]));")
                return ("\n".join(L), None)
            if ax == 1:
                return (
                    f"/* {o} = SLICE({x},1,{lo},{hi}) */\n"
                    f"for (int64_t s_i = 0; s_i < {o}_N; ++s_i)\n"
                    f"  memcpy(&{o}[s_i * {hi - lo}], "
                    f"&{x}[s_i * {in_d} + {lo}], "
                    f"{hi - lo} * sizeof(double));",
                    None)
            w1 = self._we(S, x, ctx, f"SLICE {o} stride")
            return (
                f"/* {o} = SLICE({x},0,{lo},{hi}) rows */\n"
                f"memcpy({o}, &{x}[{lo} * {w1}], "
                f"{co} * sizeof(double));",
                None)
        if mn == "ROTARY":
            x, pos = args
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend ROTARY needs F (T lives "
                                "on nonfpu)")
            d = self._w(S, x, f"ROTARY {o}")
            nh = d // 2
            return (
                f"/* {o} = ROTARY({x}) runtime sincos */\n"
                f"for (int64_t t_i = 0; t_i < {o}_N; ++t_i) {{\n"
                f"  int64_t pp = {pos}[t_i];\n"
                f"  for (int64_t t_h = 0; t_h < {nh}; ++t_h) {{\n"
                f"    double ang = (double)pp * THETA[t_h];\n"
                f"    double c = cos(ang), s = sin(ang);\n"
                f"    double x0 = {x}[(t_i * {d}) + 2 * t_h];\n"
                f"    double x1 = {x}[(t_i * {d}) + 2 * t_h + 1];\n"
                f"    {o}[(t_i * {d}) + 2 * t_h] = x0 * c - x1 * s;\n"
                f"    {o}[(t_i * {d}) + 2 * t_h + 1] = x0 * s + x1 * c;\n"
                f"  }}\n"
                f"}}",
                None)
        if mn == "TRANSPOSE":
            x = args[0]
            o = outs[0]
            if S[x][0] == "T":
                st, et, zt = S[x][1]
                c = S[x][2][1]
                if c is None:
                    raise NoPattern("C backend: TRANSPOSE-T dynamic width")
                r = S[x][2][0]
                re = str(int(r)) if r is not None else f"{x}_N"
                L = [f"/* {o} = TRANSPOSE({x}) triples, exact move */"]
                for comp, ct in (("s", st), ("e", et), ("z", zt)):
                    L.append(
                        f"for (int64_t t_i = 0; t_i < {re}; ++t_i)\n"
                        f"  for (int64_t t_j = 0; t_j < {c}; ++t_j)\n"
                        f"    {o}_{comp}[t_j * {re} + t_i] = "
                        f"{x}_{comp}[t_i * {c} + t_j];")
                return ("\n".join(L), None)
            c = self._we(S, x, ctx, f"TRANSPOSE {o}")
            r = S[x][2][0]
            if r is None and x in ctx["inputs"]:
                raise NoPattern("C backend: TRANSPOSE of dynamic-row input")
            re = str(int(r)) if r is not None else f"{x}_N"
            return (
                f"/* {o} = TRANSPOSE({x}) */\n"
                f"for (int64_t t_i = 0; t_i < {re}; ++t_i)\n"
                f"  for (int64_t t_j = 0; t_j < {c}; ++t_j)\n"
                f"    {o}[t_j * {re} + t_i] = {x}[t_i * {c} + t_j];",
                None)
        if mn in ("TBETA", "BETA"):
            ref = args[0]
            o = outs[0]
            # ref is geometric only (constant fill); any kind accepted.
            co = ctx["count"][o]
            val = float(cfg.get("beta_b", float(cfg.get("beta", 0.5)))) \
                if mn == "TBETA" else float(cfg.get("beta", 0.5))
            return (f"/* {o} = {mn} ({val}) over {ref} shape */\n"
                    f"for (int64_t k_i = 0; k_i < {co}; ++k_i)\n"
                    f"  {o}[k_i] = {val!r};",
                    None)
        if mn in ("MUL", "ADD", "SUB"):
            a, b = args
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"C backend {mn} needs F (T arithmetic "
                                "lives on nonfpu)")
            co = ctx["count"][o]
            op = {"MUL": "*", "ADD": "+", "SUB": "-"}[mn]
            consts = ctx.get("consts", {})
            la = repr(float(consts[a])) if a in consts else f"{a}[e_i]"
            lb = repr(float(consts[b])) if b in consts else f"{b}[e_i]"
            tag = (" (const-folded %s%s)" % (a if a in consts else "",
                                             b if b in consts else ""))
            return (f"/* {o} = {a} {op} {b}{tag} */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                    f"  {o}[e_i] = {la} {op} {lb};",
                    None)
        if mn == "SILU":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend SILU needs F (T lives "
                                "on nonfpu)")
            co = ctx["count"][o]
            return (f"/* {o} = SILU({x}) */\n"
                    f"for (int64_t e_i = 0; e_i < {co}; ++e_i)\n"
                    f"  {o}[e_i] = {x}[e_i] / (1.0 + exp(-{x}[e_i]));",
                    None)
        if mn == "SELECT":
            m, a, b = args
            o = outs[0]
            co = ctx["count"][o]
            # Runtime shape contract (mask file elements vs branch count):
            # sample-S may differ from runtime-S; static baked counts lie.
            guard = f"if ({m}_N != {co}) return 21;\n  "
            if S[a][0] == "T":
                L = [f"/* {o} = SELECT({m}) triples, exact */\n  {guard}"
                     f"for (int64_t v_i = 0; v_i < {co}; ++v_i) {{"]
                for comp in ("s", "e", "z"):
                    L.append(f"  {o}_{comp}[v_i] = {m}[v_i] ? "
                             f"{a}_{comp}[v_i] : {b}_{comp}[v_i];")
                L.append("}")
                return ("\n".join(L), None)
            return (f"/* {o} = SELECT({m}) nonzero picks A */\n  {guard}"
                    f"for (int64_t v_i = 0; v_i < {co}; ++v_i)\n"
                    f"  {o}[v_i] = {m}[v_i] ? {a}[v_i] : {b}[v_i];",
                    None)
        if mn == "TSHIFT":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend TSHIFT needs F (T lives "
                                "on nonfpu)")
            d = self._we(S, x, ctx, f"TSHIFT {o}")
            return (
                f"/* {o} = TSHIFT({x}) row-max shift */\n"
                f"for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {{\n"
                f"  double mx = {x}[f_i * {d}];\n"
                f"  for (int64_t f_j = 1; f_j < {d}; ++f_j)\n"
                f"    if ({x}[f_i * {d} + f_j] > mx)\n"
                f"      mx = {x}[f_i * {d} + f_j];\n"
                f"  for (int64_t f_j = 0; f_j < {d}; ++f_j)\n"
                f"    {o}[f_i * {d} + f_j] = {x}[f_i * {d} + f_j] - mx;\n"
                f"}}",
                None)
        if mn == "SOFTMAX_WIDE":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("C backend SOFTMAX_WIDE needs F (T lives "
                                "on nonfpu)")
            d = self._we(S, x, ctx, f"SOFTMAX_WIDE {o}")
            return (
                f"/* {o} = SOFTMAX_WIDE({x}) stable rows */\n"
                f"for (int64_t w_i = 0; w_i < {o}_N; ++w_i) {{\n"
                f"  double mx = {x}[w_i * {d}];\n"
                f"  for (int64_t w_j = 1; w_j < {d}; ++w_j)\n"
                f"    if ({x}[w_i * {d} + w_j] > mx)\n"
                f"      mx = {x}[w_i * {d} + w_j];\n"
                f"  double se = 0;\n"
                f"  for (int64_t w_j = 0; w_j < {d}; ++w_j) {{\n"
                f"    {o}[w_i * {d} + w_j] = exp({x}[w_i * {d} + w_j] - mx);\n"
                f"    se += {o}[w_i * {d} + w_j];\n"
                f"  }}\n"
                f"  for (int64_t w_j = 0; w_j < {d}; ++w_j)\n"
                f"    {o}[w_i * {d} + w_j] /= se;\n"
                f"}}",
                None)
        if mn == "CONCAT":
            a, b = args[0], args[1]
            o = outs[0]
            da, db = S[a][2][1], S[b][2][1]
            if S[a][0] == "T":
                L = [f"/* {o} = CONCAT({a},{b},1) triples, exact */"]
                for comp in ("s", "e", "z"):
                    L.append(
                        f"for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {{\n"
                        f"  memcpy(&{o}_{comp}[n_i * {da + db}], "
                        f"&{a}_{comp}[n_i * {da}], "
                        f"{da} * sizeof({o}_{comp}[0]));\n"
                        f"  memcpy(&{o}_{comp}[n_i * {da + db} + {da}], "
                        f"&{b}_{comp}[n_i * {db}], "
                        f"{db} * sizeof({o}_{comp}[0]));\n}}")
                return ("\n".join(L), None)
            return (
                f"/* {o} = CONCAT({a},{b},1) */\n"
                f"for (int64_t n_i = 0; n_i < {o}_N; ++n_i) {{\n"
                f"  memcpy(&{o}[n_i * {da + db}], &{a}[n_i * {da}], "
                f"{da} * sizeof(double));\n"
                f"  memcpy(&{o}[n_i * {da + db} + {da}], &{b}[n_i * {db}], "
                f"{db} * sizeof(double));\n"
                f"}}",
                None)
        return super().pattern(mn, outs, args, sig, ctx)

    def epilogue(self, ctx):
        return ""


def _rows_expr(sh, ids_n):
    """Row-count value: first dim baked, batch source when dynamic, 1."""
    if not sh:
        return "1"
    if sh[0] is None:
        return "(%s)" % ids_n
    return str(int(sh[0]))


def _dim_expr(sh, ids_n):
    if not sh:
        raise AsmError("C backend: scalar streams need explicit shape")
    return " * ".join(ids_n if d is None else str(int(d)) for d in sh)



def find_consts(bound, inp, streams, sample):
    """Const-folding: single-valued F/T IN streams consumed SOLELY by
    elementwise MUL/ADD/SUB become compile-time literals (no argv file,
    no malloc). I streams never fold (ids/masks stay addressable).
    Shared by the C and CUDA drivers (same IR, same deal)."""
    _uses = {}
    _, _, _prog0 = bound
    for _outs, _mn, _fn, _args, _ln, _sg in _prog0:
        for _a in _args:
            _uses.setdefault(_a, set()).add(_mn)
    consts = {}
    for _n, _ in inp:
        if _n not in sample or _n not in streams:
            continue
        _k, _sub, _sh = streams[_n]
        if _k not in ("F", "T"):
            continue
        if not all(_m in ("MUL", "ADD", "SUB") for _m in _uses.get(_n, ())):
            continue
        if _k == "F":
            _u = np.unique(np.ascontiguousarray(sample[_n], dtype=np.float64))
            if len(_u) == 1:
                consts[_n] = float(_u[0])
        else:
            _ss, _ee, _zz = (np.ascontiguousarray(p) for p in sample[_n])
            _sig = (_ss.astype(np.int64) * 1000000000000
                    + _ee.astype(np.int64) * 10 + _zz.astype(np.int64))
            if np.unique(_sig).size == 1:
                consts[_n] = (int(_ss.flat[0]), int(_ee.flat[0]),
                              int(_zz.flat[0]))
    return consts

def compile_program(text, target="c", sample=None, outputs=None,
                    registry=None, sigs=None, basedir=".", origin=None,
                    use_blas=False, time_ops=False, use_fp16=False):
    """Frontend entry: asm text -> target source. Returns dict with
    source/backend/streams/dims/config. Sample payload required (shapes).
    """
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    config, inp, bound, _ = assemble(text, registry or _R, sigs or _S,
                                     basedir=basedir, origin=origin)
    from chain.backends import get_backend as _gb
    _be = _gb(target)  # unknown names fail loud here
    _be.use_blas = use_blas
    _be.use_fp16 = use_fp16
    if _be.name == "cuda":
        from chain.emit_cuda import compile_cuda as _cc
        return _cc(text, sample=sample, outputs=outputs,
                   registry=registry, sigs=sigs, basedir=basedir,
                   origin=origin,
                   use_fp16=getattr(_be, "use_fp16", False))
    if not _be.emits:
        # stub backend: surface its extension recipe, don't compile air.
        raise NoPattern(getattr(_be, "stub_note",
                                f"no emitting '{target}' backend yet"))
    if sample is None:
        raise AsmError("compile_program needs sample payload (shapes)")
    if not outputs:
        raise AsmError("compile_program needs outputs=[...] (OUT streams)")
    be = _be
    streams = infer_shapes((config, inp, bound), sample)
    consts = find_consts((config, inp, bound), inp, streams, sample)
    in_names = [n for n, _ in inp]
    dims = {}
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "T" and sh and all(d is not None for d in sh):
            for ax, d in enumerate(sh):
                dims[f"D{n}_{ax}"] = d
            if len(sh) == 2:
                dims[f"V_{n}"], dims[f"C_{n}"] = sh
    # batch-ids stream: first (N,) I input (drives dynamic dims);
    # else a dynamic-row F input (rows from file bytes / baked width).
    ids_n = None
    for n in in_names:
        k, _, sh = streams[n]
        if k == "I" and len(sh) == 1:
            ids_n = f"{n}_N"
            break
    if ids_n is None:
        for n in in_names:
            k, _, sh = streams[n]
            if k == "F" and sh and sh[0] is None and sh[-1] is not None:
                ids_n = f"{n}_N"
                break
    if ids_n is None:
        ids_n = "1"
    # rotary theta constants from head dim (baked by construction)
    theta = []
    for _on, _mn, _fn, _aa, _ln, _sg in bound:
        if _mn == "ROTARY":
            from chain.asm_ops import rope_tables  # noqa
            d = streams[_aa[0]][2][1]
            base = float(config.get("rope_base", 10000.0))
            th = np.power(float(base),
                          -2.0 * np.arange(d // 2, dtype=np.float64) / d)
            theta = [float(t) for t in th]
            break
    rope = None
    for _on, _mn, _fn, _aa, _ln, _sg in bound:
        if _mn == "ROTARY" and len(_aa) > 1 and _aa[1] in sample:
            _d = streams[_aa[0]][2][-1]
            if _d is None:
                continue
            _pos = np.ascontiguousarray(sample[_aa[1]], dtype=np.int64)
            _P = int(_pos.max(initial=0)) + 1
            from chain.asm_ops import rope_tables as _rt
            _base = float(config.get("rope_base", 10000.0))
            _ct, _st = _rt(_P, _d, _base)
            rope = {"P": _P, "D": _d, "cos": np.ascontiguousarray(_ct),
                    "sin": np.ascontiguousarray(_st)}
            break
    ctx = {"streams": streams, "dims": dims, "dyn": {}, "ids_n": ids_n,
           "inputs": in_names, "outputs": outputs, "gathered": {},
           "config": dict(config), "theta": theta, "rope": rope,
           "count": {}, "mns": [mn for _, mn, _, _, _, _ in bound],
           "consts": consts}
    for _xn, (_xk, _xs, _xsh) in streams.items():
        if _xk in ("F", "T") and _xsh:
            ctx["count"][_xn] = "(%s)" % _dim_expr(_xsh, ids_n)
        elif _xk == "I" and _xn in outputs:
            _osh = _xsh
            if _osh and all(d is not None for d in _osh):
                ctx["count"][_xn] = "(%s)" % int(
                    np.prod(list(_osh), dtype=np.int64))
            else:
                ctx["count"][_xn] = "(%s)" % ids_n
    parts = [be.prologue(ctx)]
    parts.append("/* inputs: argv files, order below */")
    decls = []
    for n in in_names:
        if n in consts:
            continue  # compile-time literal: no storage, no argv file
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            decls.append(f"static {st} *{n}_s; static {et} *{n}_e;"
                         f" static {zt} *{n}_z; static int64_t {n}_N;")
        elif k == "F":
            decls.append(f"static double *{n}; static int64_t {n}_N;")
        else:
            decls.append(f"static int64_t *{n}; static int64_t {n}_N;")
    for o in outputs:
        ok, osub, osh = streams[o]
        if ok == "F":
            decls.append(f"static double *{o};")
        elif ok == "T":
            st, et, zt = osub
            decls.append(f"static {st} *{o}_s; static {et} *{o}_e;"
                         f" static {zt} *{o}_z; static int64_t {o}_N;")
        else:
            decls.append(f"static int64_t *{o}; static int64_t {o}_N;")
    parts.append("\n".join(decls))
    # intermediate declarations (F/T streams malloc'd per op in-body)
    inter = set()
    interT = {}
    _, _, prog = (config, inp, bound)
    for outs, _mn2, _fn2, _aa2, _ln2, _sg2 in prog:
        for x in outs:
            if x not in in_names and x not in outputs:
                k, ksub, _ = streams[x]
                if k == "F":
                    inter.add(x)
                elif k == "T":
                    interT[x] = ksub
    if inter:
        parts.append("/* intermediates */")
        parts.append("\n".join(f"double *{x};" for x in sorted(inter)))
    if interT:
        parts.append("/* triple intermediates */")
        for x in sorted(interT):
            st, et, zt = interT[x]
            parts.append(f"{st} *{x}_s; {et} *{x}_e; {zt} *{x}_z;")
    parts.append("int main(int argc, char **argv) {")
    live_ins = [n for n in in_names if n not in consts]
    n_argv = sum(3 if streams[n][0] == "T" else 1 for n in live_ins)
    n_argv += sum(3 if streams[o][0] == "T" else 1 for o in outputs)
    parts.append(f"  if (argc != {n_argv + 1}) return 9;")
    idx = 1
    loads = []
    for n in in_names:
        if n in consts:
            continue
        k, sub, sh = streams[n]
        if k == "T":
            st, et, zt = sub
            if not sh or any(d is None for d in sh):
                raise NoPattern(f"C backend: T input '{n}' must be fully "
                                f"baked {sh}")
            cnt = " * ".join(f"DIM_D{n}_{ax}" for ax in range(len(sh)))
            loads.append(f"  {n}_N = {f'DIM_D{n}_0' if len(sh) else '1'}; "
                         f"/* rows */")
            loads.append(f"  /* load {n} triples ({' x '.join(str(d) for d in sh)}) */")
            for comp, ct in (("s", st), ("e", et), ("z", zt)):
                loads.append(
                    f"  {{ FILE *f = fopen(argv[{idx}], \"rb\");"
                    f" if (!f) return 1;\n"
                    f"    {n}_{comp} = malloc(({cnt})"
                    f" * sizeof({ct}));"
                    f" if (fread({n}_{comp}, sizeof({ct}), ({cnt}), f)"
                    f" != (size_t)({cnt})) return 2;"
                    f" fclose(f); }}")
                idx += 1
        elif k == "F":
            _k2, _s2, _sh2 = streams[n]
            if _sh2 and _sh2[-1] is None:
                raise NoPattern(f"C backend: fully-dynamic F input '{n}'")
            _wr = f"({_sh2[-1]} * 8)" if _sh2 else "8"
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); long {n}_B = ftell(f);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n}_N = {_sh2[0] if _sh2 and _sh2[0] is not None else f'{n}_B / {_wr}'};"
                f"\n    {n} = malloc({n}_B);"
                f" if (fread({n}, 1, {n}_B, f) != (size_t){n}_B) return 2;"
                f" fclose(f); }}")
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
    parts.append("\n".join(loads))
    # allocate intermediates + outputs (COUNT exprs; _N vars per stream)
    allocs = []
    for x in sorted(inter) + [o for o in outputs if streams[o][0] == "F"]:
        _k, _s, xsh = streams[x]
        cnt = _dim_expr(xsh, ids_n)
        allocs.append(f"  {x} = malloc(({cnt}) * sizeof(double));")
        # per-stream ROW-count var for patterns (<stream>_N)
        allocs.append(f"  int64_t {x}_N = {_rows_expr(xsh, ids_n)};")
    for o in outputs:
        if streams[o][0] == "I":
            _osh = streams[o][2]
            allocs.append(f"  {o}_N = {_rows_expr(_osh, ids_n)}; "
                          f"{o} = malloc({o}_N * 8);")
    for x in sorted(interT) + [o for o in outputs if streams[o][0] == "T"]:
        _k, _tsub, xsh = streams[x]
        _st, _et, _zt = _tsub
        cnt = _dim_expr(xsh, ids_n)
        for _comp, _ct in (("s", _st), ("e", _et), ("z", _zt)):
            allocs.append(f"  {x}_{_comp} = malloc(({cnt}) * sizeof({_ct}));")
        allocs.append(f"  int64_t {x}_N = {_rows_expr(xsh, ids_n)};")
    parts.append("\n".join(allocs))
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        parts.append(f"  /* L{_short_site(ln)}: {mn} */")
        body = pat.replace("\n", "\n  ")
        if time_ops:
            body = ("{ clock_t _t0 = clock();\n  " + body +
                    f"\n  fprintf(stderr, \"op {mn} %f\\n\", "
                    f"(double)(clock() - _t0) / CLOCKS_PER_SEC); }}")
        parts.append("  " + body)
    for o in outputs:
        ok, osub, _osh2 = streams[o]
        if ok == "T":
            st, et, zt = osub
            co = ctx["count"][o]
            for comp, ct in (("s", st), ("e", et), ("z", zt)):
                parts.append(
                    f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                    f" return 3;\n"
                    f"    fwrite({o}_{comp}, sizeof({ct}), {co}, f);"
                    f" fclose(f); }}")
                idx += 1
            continue
        el = "8" if ok == "I" else "sizeof(double)"
        co = ctx["count"][o]
        parts.append(f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                     f" return 3;\n"
                     f"    fwrite({o}, {el}, {co}, f); fclose(f); }}")
        idx += 1
    parts.append("  return 0;\n}")
    parts.append(be.epilogue(ctx))
    src = "\n".join(parts)
    _libs = ["-lm"]
    if getattr(be, "use_blas", False) and be.name == "c":
        _libs = _libs + list(CBackend.BLAS_LIBS)
    return {"source": src, "backend": be, "streams": streams,
            "dims": dims, "config": dict(config), "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1, "libs": _libs,
            "consts": consts}


def build(source, workdir, name="prog", cc="cc", cflags=None, libs=None):
    """Compile emitted source -> exe path. Backend owns its speed flags."""
    cflags = list(cflags or CBackend.DEFAULT_CFLAGS)
    libs = list(libs or ["-lm"])
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".c")
    exe = os.path.join(workdir, name)
    with open(src, "w") as f:
        f.write(source)
    r = subprocess.run([cc] + cflags + [src, "-o", exe] + libs,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"C build failed:\n{r.stderr}")
    return exe
