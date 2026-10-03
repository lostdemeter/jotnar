"""Non-FPU backend v0.2: integer-only emission (no float, no libm).

Two subgraphs, both bit-exact vs the lattice:
  exact moves (v0.1): T-kind GATHER/ARGMAX/TRANSPOSE/SLICE/CONCAT/SELECT.
  fixed-point arithmetic (v0.2): T-kind MUL/ADD/SUB/SQUARE via a ported
    bridge (to_fixed/from_fixed + tmul + binop at m_cov). Bridge tables
    (FRAC/COARSE/FINE) bake in as static const int64 (compiler side
    reads them from phi-core; the target never computes a float).
    int8 wraparound (tmul sign plane) relies on -fwrapv, stated.

Proof of "no FPU" (gated, not claimed): float_trap() over source +
libs=[] link + bit-exact agreement.

Still ahead (explicitly not this cut): matmul_int (chunked accumulate),
rmsnorm_int/softmaxN (LUT spans), rescale_ paths, CONV/SCAN.
"""
import re

import numpy as np

from chain.asm import AsmError
from chain.emit_c import Backend, CBackend, NoPattern

INTEGER_MN = {"GATHER", "ARGMAX", "TRANSPOSE", "SLICE", "CONCAT", "SELECT",
              "MUL", "ADD", "SUB", "SQUARE", "MATMUL", "BATCH_MATMUL"}
FX_MN = {"MUL", "ADD", "SUB", "SQUARE", "MATMUL", "BATCH_MATMUL"}

TRAP_RES = (
    re.compile(r"\b(float|double)\b"),
    re.compile(r"\b(sqrt|exp|sin|cos|tan|pow|fmax|fmin|fabs|round)\s*\("),
    re.compile(r"#include\s*<math\.h>"),
)


def float_trap(source):
    """Raise if the emitted source could touch an FPU. Comments are
    stripped first so the gate checks code, not prose."""
    code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    code = re.sub(r"//.*", "", code)
    for rx in TRAP_RES:
        m = rx.search(code)
        if m:
            raise AsmError(f"nonfpu float-trap: {m.group(0)!r} -- "
                           f"FPU use in integer-only emission")


def _frozen_scales():
    """Frozen M.json defaults (compile-time; CONFIG overrides per program)."""
    import json as _j
    import os as _os
    try:
        m = _j.load(open(_os.path.join(_os.path.dirname(
            _os.path.abspath(__file__)), "M.json")))
        return int(m["m_acc"]), int(m["m_cov"])
    except Exception:  # noqa: BLE001 -- fall back loud-neutrally
        return 31344, 33280


_FROZEN_ACC, _FROZEN_COV = _frozen_scales()


def _bake_vec(name, arr, per_line=16):
    flat = [str(int(v)) for v in np.ascontiguousarray(arr).reshape(-1)]
    lines = [", ".join(flat[i:i + per_line])
             for i in range(0, len(flat), per_line)]
    body = ",\n ".join(lines)
    return (f"static const int64_t {name}[{len(flat)}]"
            f" = {{\n {body}\n}};")


def fx_tables_and_helpers(m_cov, m_acc, need_square):
    """Bridge port: to_fixed/from_fixed + tmul + binop, tables baked.
    Faithful to phi-core lattice.py/numpy_ops semantics (int64 2^-18
    counts; BIAS 32768; e clipped to [0,65535]; int8 sign wraps)."""
    import phi_core.lattice as S
    L = [_bake_vec("NF_FRAC", S.L_FRAC()),
         _bake_vec("NF_COARSE", S.L_COARSE()),
         _bake_vec("NF_FINE", S.L_FINE()),
         f"static const int NF_M = {int(m_cov)};",
         f"static const int NF_MACC = {int(m_acc)};",
         f"static const int64_t NF_BASE = {int(S.L_FRAC()[0])};"]
    if need_square:
        import numpy as _np
        qs, qe, qz = S.encode(_np.array([0.0, 1.0]))
        qlo = int(S.to_fixed(qs[0:1], qe[0:1], qz[0:1], int(m_cov))[0])
        qhi = int(S.to_fixed(qs[1:2], qe[1:2], qz[1:2], int(m_cov))[0])
        L.append(f"static const int64_t NF_QLO = {qlo}LL;")
        L.append(f"static const int64_t NF_QHI = {qhi}LL;")
    L.append(r"""
static int64_t nf_to_fixed(int8_t s, int32_t e, uint8_t z, int m) {
  if (z) return 0;
  int64_t d = (int64_t)m - (int64_t)e;
  if (d < 0) return (int64_t)s * NF_BASE;
  if (d > 13312) return 0;
  return (int64_t)s * NF_FRAC[d];
}
static void nf_from_fixed(int64_t q, int m, int8_t *s, int32_t *e,
                          uint8_t *z) {
  *z = (q == 0);
  *s = (q > 0) ? 1 : -1;
  uint64_t a = (q >= 0) ? (uint64_t)q : (uint64_t)(-(q + 1)) + 1u;
  int bl = 0; uint64_t x = a;
  if (x >> 32) { bl += 32; x >>= 32; }
  if (x >> 16) { bl += 16; x >>= 16; }
  if (x >> 8) { bl += 8; x >>= 8; }
  if (x >> 4) { bl += 4; x >>= 4; }
  if (x >> 2) { bl += 2; x >>= 2; }
  if (x >> 1) { bl += 1; }
  bl += (a != 0);
  int shift = bl - 15;
  uint64_t mant = (shift >= 0) ? ((shift > 0) ? (a >> shift) : a)
                               : (a << (-shift));
  int t = shift + 14 - 18;
  int idx = t + 64;
  if (idx < 0) idx = 0;
  if (idx > 192) idx = 192;
  int64_t fi = (int64_t)mant - 16384;
  if (fi < 0) fi = 0;
  if (fi > 16383) fi = 16383;
  int64_t ev = (int64_t)m + NF_COARSE[idx] + NF_FINE[fi];
  if (ev < 0) ev = 0;
  if (ev > 65535) ev = 65535;
  *e = (int32_t)ev;
}
static void nf_binop(int8_t sa, int32_t ea, uint8_t za,
                     int8_t sb, int32_t eb, uint8_t zb, int is_sub,
                     int8_t *os, int32_t *oe, uint8_t *oz) {
  int64_t qa = nf_to_fixed(sa, ea, za, NF_M);
  int64_t qb = nf_to_fixed(sb, eb, zb, NF_M);
  int64_t q = is_sub ? (qa - qb) : (qa + qb);
  nf_from_fixed(q, NF_M, os, oe, oz);
}
static void nf_tmul(int8_t sa, int32_t ea, uint8_t za,
                    int8_t sb, int32_t eb, uint8_t zb,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  *os = (int8_t)((int)sa * (int)sb); /* wraps mod 256: -fwrapv */
  int64_t pe = (int64_t)ea + (int64_t)eb - 32768;
  if (pe < 0) pe = 0;
  if (pe > 65535) pe = 65535;
  *oe = (int32_t)pe;
  *oz = (uint8_t)(za | zb);
}
""")
    return "\n".join(L)


class NonFPUBackend(CBackend):
    """Integer-only C. Exact moves + fixed-point arithmetic; everything
    else refused loud."""

    name = "nonfpu"
    emits = True
    DEFAULT_CFLAGS = ("-O2", "-std=c99", "-Wall", "-fwrapv")
    DEFAULT_LIBS: tuple = ()

    def prologue(self, ctx):
        L = ["#include <stdint.h>", "#include <stdio.h>",
             "#include <stdlib.h>", "#include <string.h>", ""]
        for k, v in ctx["dims"].items():
            L.append(f"#define DIM_{k} {v}L")
        if any(mn in FX_MN for mn in ctx.get("mns", [])):
            cfg = ctx.get("config", {})
            m_cov = int(cfg.get("m_cov", _FROZEN_COV))
            m_acc = int(cfg.get("m_acc", _FROZEN_ACC))
            need_sq = "SQUARE" in ctx.get("mns", [])
            L.append(fx_tables_and_helpers(m_cov, m_acc, need_sq))
        return "\n".join(L) + "\n"

    def pattern(self, mn, outs, args, sig, ctx):
        if mn not in INTEGER_MN:
            raise NoPattern(
                f"nonfpu: no integer pattern for {mn} (fixed-point core "
                f"covers moves + MUL/ADD/SUB/SQUARE; matmul/norms ahead)")
        S = ctx["streams"]
        for a in args:
            if a in S and S[a][0] == "F":
                raise NoPattern(
                    f"nonfpu: float stream '{a}' refused (no FPU on this "
                    f"target -- decode nothing, quantize first)")
        if mn in FX_MN:
            return self._fx_pattern(mn, outs, args, ctx)
        return super().pattern(mn, outs, args, sig, ctx)

    def _fx_pattern(self, mn, outs, args, ctx):
        S = ctx["streams"]
        o = outs[0]
        co = ctx["count"][o]
        if mn == "MUL":
            a, b = args
            if S[a][0] != "T" or S[b][0] != "T":
                raise NoPattern("nonfpu MUL needs T triples")
            return (
                f"/* {o} = MUL({a},{b}) tmul exact */\n"
                f"for (int64_t f_i = 0; f_i < {co}; ++f_i)\n"
                f"  nf_tmul({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], "
                f"{b}_s[f_i], {b}_e[f_i], {b}_z[f_i], "
                f"&{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);",
                None)
        if mn in ("ADD", "SUB"):
            a, b = args
            if S[a][0] != "T" or S[b][0] != "T":
                raise NoPattern(f"nonfpu {mn} needs T triples")
            sub = 1 if mn == "SUB" else 0
            return (
                f"/* {o} = {mn}({a},{b}) bridge @ NF_M */\n"
                f"for (int64_t f_i = 0; f_i < {co}; ++f_i)\n"
                f"  nf_binop({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], "
                f"{b}_s[f_i], {b}_e[f_i], {b}_z[f_i], {sub}, "
                f"&{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);",
                None)
        if mn == "SQUARE":
            a = args[0]
            if S[a][0] != "T":
                raise NoPattern("nonfpu SQUARE needs T triples")
            return (
                f"/* {o} = SQUARE({a}) tmul-self + clip [NF_QLO,NF_QHI] */\n"
                f"for (int64_t f_i = 0; f_i < {co}; ++f_i) {{\n"
                f"  int8_t _s; int32_t _e; uint8_t _z;\n"
                f"  nf_tmul({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], "
                f"{a}_s[f_i], {a}_e[f_i], {a}_z[f_i], &_s, &_e, &_z);\n"
                f"  int64_t _q = nf_to_fixed(_s, _e, _z, NF_M);\n"
                f"  if (_q < NF_QLO) _q = NF_QLO;\n"
                f"  if (_q > NF_QHI) _q = NF_QHI;\n"
                f"  nf_from_fixed(_q, NF_M, &{o}_s[f_i], &{o}_e[f_i], "
                f"&{o}_z[f_i]);\n}}",
                None)
        if mn in ("MATMUL", "BATCH_MATMUL"):
            a, b = args
            if S[a][0] != "T" or S[b][0] != "T":
                raise NoPattern(f"nonfpu {mn} needs T triples")
            o = outs[0]
            ash, bsh = S[a][2], S[b][2]
            if len(ash) == 2:
                m, k, n = ash[0], ash[1], bsh[1]
                me = str(m) if m is not None else ctx["ids_n"]
                ke = str(k) if k is not None else ctx["ids_n"]
                ne = str(n) if n is not None else ctx["ids_n"]
                return (
                    f"/* {o} = {mn}({a},{b}) tmul+bridge @ NF_MACC */\n"
                    f"for (int64_t w_i = 0; w_i < {me}; ++w_i)\n"
                    f"  for (int64_t w_j = 0; w_j < {ne}; ++w_j) {{\n"
                    f"    int64_t acc = 0;\n"
                    f"    for (int64_t w_k = 0; w_k < {ke}; ++w_k) {{\n"
                    f"      int8_t _s; int32_t _e; uint8_t _z;\n"
                    f"      nf_tmul({a}_s[w_i * {ke} + w_k], "
                    f"{a}_e[w_i * {ke} + w_k], {a}_z[w_i * {ke} + w_k],\n"
                    f"              {b}_s[w_k * {ne} + w_j], "
                    f"{b}_e[w_k * {ne} + w_j], {b}_z[w_k * {ne} + w_j],\n"
                    f"              &_s, &_e, &_z);\n"
                    f"      acc += nf_to_fixed(_s, _e, _z, NF_MACC);\n"
                    f"    }}\n"
                    f"    nf_from_fixed(acc, NF_MACC, &{o}_s[w_i * {ne} + w_j],"
                    f" &{o}_e[w_i * {ne} + w_j], &{o}_z[w_i * {ne} + w_j]);\n"
                    f"  }}",
                    None)
            B = ash[0]
            if B is None:
                raise NoPattern("nonfpu batched matmul needs baked batch")
            m, k, n = ash[1], ash[2], bsh[2]
            me = str(m) if m is not None else ctx["ids_n"]
            ke = str(k) if k is not None else ctx["ids_n"]
            ne = str(n) if n is not None else ctx["ids_n"]
            bb = bsh[0]
            return (
                f"/* {o} = {mn}({a},{b}) batched"
                f"{' + B-broadcast' if bb == 1 else ''} */\n"
                f"for (int64_t w_b = 0; w_b < {B}; ++w_b)\n"
                f"  for (int64_t w_i = 0; w_i < {me}; ++w_i)\n"
                f"    for (int64_t w_j = 0; w_j < {ne}; ++w_j) {{\n"
                f"      int64_t acc = 0;\n"
                f"      int64_t _bo = {'0' if bb == 1 else 'w_b'};\n"
                f"      for (int64_t w_k = 0; w_k < {ke}; ++w_k) {{\n"
                f"        int8_t _s; int32_t _e; uint8_t _z;\n"
                f"        nf_tmul({a}_s[(w_b * {me} + w_i) * {ke} + w_k],\n"
                f"                {a}_e[(w_b * {me} + w_i) * {ke} + w_k],\n"
                f"                {a}_z[(w_b * {me} + w_i) * {ke} + w_k],\n"
                f"                {b}_s[(_bo * {ke} + w_k) * {ne} + w_j],\n"
                f"                {b}_e[(_bo * {ke} + w_k) * {ne} + w_j],\n"
                f"                {b}_z[(_bo * {ke} + w_k) * {ne} + w_j],\n"
                f"                &_s, &_e, &_z);\n"
                f"        acc += nf_to_fixed(_s, _e, _z, NF_MACC);\n"
                f"      }}\n"
                f"      nf_from_fixed(acc, NF_MACC,\n"
                f"        &{o}_s[(w_b * {me} + w_i) * {ne} + w_j],\n"
                f"        &{o}_e[(w_b * {me} + w_i) * {ne} + w_j],\n"
                f"        &{o}_z[(w_b * {me} + w_i) * {ne} + w_j]);\n"
                f"    }}",
                None)
        raise NoPattern(f"nonfpu: no fx pattern for {mn}")
