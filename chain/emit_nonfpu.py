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
              "MUL", "ADD", "SUB", "SQUARE", "MATMUL", "BATCH_MATMUL",
              "RMSNORM", "TSHIFT", "SOFTMAX_WIDE", "SOFTMAX",
              "TBETA", "BETA", "ROTARY", "SILU"}
FX_MN = {"MUL", "ADD", "SUB", "SQUARE", "MATMUL", "BATCH_MATMUL",
         "RMSNORM", "TSHIFT", "SOFTMAX_WIDE", "SOFTMAX",
         "TBETA", "BETA", "ROTARY", "ARGMAX", "SILU"}
SM_MN = {"TSHIFT", "SOFTMAX_WIDE", "SOFTMAX"}

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


def fx_tables_and_helpers(m_cov, m_acc, need_square, need_softmax,
                          eps_c, need_silu=False):
    """Bridge port: to_fixed/from_fixed + tmul + binop, tables baked.
    Faithful to phi-core lattice.py/numpy_ops semantics (int64 2^-18
    counts; BIAS 32768; e clipped to [0,65535]; int8 sign wraps).
    Softmax extras (EXP 262145 + FRAC_HI 8192 tables, isqrt, wide
    bridge) emit only when a softmax op is present; sigmoid extras
    (SIGX 65536 + SIG 524289 tables) only with SILU."""
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
    sm_extra = []
    sm_helpers = ""
    import numpy as _np
    import os as _os
    from phi_core import numpy_ops as _N
    if need_silu:
        # sigmoid tables mirror numpy_ops formulas exactly (load-or-build
        # frozen pattern); asymptotes exact, span +-16 in 2^-14 units.
        _lp = _os.path.join(_os.path.dirname(_os.path.abspath(_N.__file__)),
                            "..", "luts")
        try:
            _sigx = _np.load(_os.path.join(_lp, "sigx_lut.npy"))
        except Exception:  # noqa: BLE001
            _sigx = _np.round(_np.power(
                S.PHI, (_np.arange(65536, dtype=_np.float64) - S.BIAS) / S.K
                ) * 16384).astype(_np.int64)
        try:
            _sig = _np.load(_os.path.join(_lp, "sig_lut.npy"))
        except Exception:  # noqa: BLE001
            _k = _np.arange(2 * 16 * 16384 + 1, dtype=_np.float64)
            _x = (_k - 16 * 16384) / 16384.0
            _sig = _np.round(1.0 / (1.0 + _np.exp(-_x)) * 16384
                             ).astype(_np.int64)
        sm_extra.append(_bake_vec("NF_SIGX", _sigx, per_line=8))
        sm_extra.append(_bake_vec("NF_SIG", _sig, per_line=8))
        sm_helpers += r"""
static void nf_sigmoid(int8_t s, int32_t e, uint8_t z,
                       int8_t *os, int32_t *oe, uint8_t *oz) {
  /* sigmoid_int port: EXPACT gather (any range, no saturation) -> LUT
     over span +-16 (262144 in 2^-14) -> asymptotes outside. */
  int64_t ee = (int64_t)e;
  if (ee < 0) ee = 0;
  if (ee > 65535) ee = 65535;
  int64_t x14 = z ? 0 : (int64_t)s * NF_SIGX[ee];
  int64_t y14;
  if (x14 < -262144) y14 = 0;
  else if (x14 > 262144) y14 = 16384;
  else {
    int64_t idx = x14 + 262144;
    y14 = NF_SIG[idx];
  }
  nf_from_fixed(y14 * 16, 32768, os, oe, oz);
}
static void nf_silu(int8_t s, int32_t e, uint8_t z,
                    int8_t *os, int32_t *oe, uint8_t *oz) {
  int8_t gs; int32_t ge; uint8_t gz;
  nf_sigmoid(s, e, z, &gs, &ge, &gz);
  nf_tmul(gs, ge, gz, s, e, z, os, oe, oz);
}
"""
    if need_softmax:
        _lp = _os.path.join(_os.path.dirname(_os.path.abspath(_N.__file__)),
                            "..", "luts")
        try:
            _exp = _np.load(_os.path.join(_lp, "exp_lut.npy"))
        except Exception:  # noqa: BLE001 -- build frozen formula
            _d = _np.arange(262145, dtype=_np.float64)
            _exp = _np.round((2.0 ** 24) * _np.exp(-_d / 16384.0)
                             ).astype(_np.int64)
        try:
            _fhi = _np.load(_os.path.join(_lp, "frac_hi_lut.npy"))
        except Exception:  # noqa: BLE001
            _fhi = _np.round(_np.power(
                S.PHI, _np.arange(1, 8193, dtype=_np.float64) / S.K
                ) * (1 << 18)).astype(_np.int64)
        sm_extra = [_bake_vec("NF_EXP", _exp, per_line=8),
                    _bake_vec("NF_FRACHI", _fhi, per_line=8),
                    f"static const int64_t NF_EPSC = {int(eps_c)}LL;"]
        sm_helpers = r"""
static uint64_t nf_isqrt(uint64_t a) {
  /* Integer floor sqrt (Newton, pure integer, no libm). */
  if (a < 2) return a;
  uint64_t x = a;
  uint64_t y = (x + 1) >> 1;
  while (y < x) { x = y; y = (x + a / x) >> 1; }
  return x;
}
static int64_t nf_to_fixed_wide(int8_t s, int32_t e, uint8_t z, int m) {
  if (z) return 0;
  int64_t d = (int64_t)m - (int64_t)e;
  if (d < 0) {
    if (d < -8192) return (int64_t)s * NF_BASE;
    return (int64_t)s * NF_FRACHI[(int)(-d) - 1];
  }
  if (d > 13312) return 0;
  return (int64_t)s * NF_FRAC[d];
}
static void nf_softmax_tail(const int64_t *num, int64_t den, int64_t n,
                            int8_t *os, int32_t *oe, uint8_t *oz) {
  /* Authored normalization (matches op_softmax/softmax_wide): den
     broadcasts per row; tdiv truncates C-style; from_fixed @ BIAS. */
  for (int64_t j = 0; j < n; ++j) {
    int64_t dd = (den == 0) ? 1 : den;
    int64_t c = (num[j] * (int64_t)(1 << 18)) / dd; /* tdiv trunc */
    nf_from_fixed(c, 32768, &os[j], &oe[j], &oz[j]);
  }
}
static void nf_softmax_row(const int64_t *q, int64_t n,
                           int8_t *os, int32_t *oe, uint8_t *oz,
                           int64_t *num) {
  /* softmaxN_fixed @ BIAS: 2^-18 -> 2^-14, rowwise max, EXP LUT. */
  int64_t s0 = (q[0] >= 0) ? (q[0] / 16) : -(((-q[0]) + 15) / 16);
  int64_t vmax = s0;
  for (int64_t j = 1; j < n; ++j) {
    int64_t s14 = (q[j] >= 0) ? (q[j] / 16) : -(((-q[j]) + 15) / 16);
    if (s14 > vmax) vmax = s14;
  }
  int64_t den = 0;
  for (int64_t j = 0; j < n; ++j) {
    int64_t s14 = (q[j] >= 0) ? (q[j] / 16) : -(((-q[j]) + 15) / 16);
    int64_t dd = vmax - s14;
    if (dd < 0) dd = 0;
    if (dd > 262144) dd = 262144;
    num[j] = NF_EXP[dd];
    den += num[j];
  }
  if (den == 0) den = 1;
  nf_softmax_tail(num, den, n, os, oe, oz);
}
static void nf_rmsnorm_row(const int8_t *xs, const int32_t *xe,
                           const uint8_t *xz, const int8_t *ws,
                           const int32_t *we, const uint8_t *wz,
                           int64_t C, int m,
                           int8_t *os, int32_t *oe, uint8_t *oz,
                           int64_t *q, int64_t *wq) {
  for (int64_t j = 0; j < C; ++j) {
    q[j] = nf_to_fixed(xs[j], xe[j], xz[j], m);
    wq[j] = nf_to_fixed(ws[j], we[j], wz[j], m);
  }
  __int128 ss = 0;
  for (int64_t j = 0; j < C; ++j) ss += (__int128)q[j] * q[j];
  /* tdiv(sum, C): truncation toward zero on signed __int128. */
  int64_t ms = (ss >= 0) ? (int64_t)(ss / C) : -(int64_t)((-ss) / C);
  ms += NF_EPSC;
  uint64_t rms = nf_isqrt((ms >= 0) ? (uint64_t)ms : 0);
  for (int64_t j = 0; j < C; ++j) {
    int64_t norm;
    if (rms == 0) norm = 0;
    else {
      __int128 scaled = (__int128)q[j] * (int64_t)(1 << 18);
      norm = (scaled >= 0) ? (int64_t)(scaled / (int64_t)rms)
                           : -(int64_t)((-scaled) / (int64_t)rms);
    }
    __int128 aff = (__int128)norm * wq[j];
    int64_t av = (aff >= 0) ? (int64_t)(aff / (int64_t)(1 << 18))
                            : -(int64_t)((-aff) / (int64_t)(1 << 18));
    nf_from_fixed(av, m, &os[j], &oe[j], &oz[j]);
  }
}
"""
    L.extend(sm_extra)
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
SOFTMAX_HELPERS
""")
    L[-1] = L[-1].replace("SOFTMAX_HELPERS", sm_helpers)
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
        rope = ctx.get("rope")
        if rope is not None and any(mn == "ROTARY" for mn in ctx.get("mns", [])):
            import phi_core.lattice as _S
            import numpy as _np
            P, D = rope["P"], rope["D"]
            L.append(f"static const int NF_RP = {P};")
            for tag, mat in (("C", rope["cos"]), ("S", rope["sin"])):
                cs, ce, cz = _S.encode(_np.ascontiguousarray(mat))
                L.append(_bake_vec(f"NF_R{tag}S",
                                   _np.ascontiguousarray(cs, dtype=_np.int8)))
                L.append(_bake_vec(f"NF_R{tag}E",
                                   _np.ascontiguousarray(ce, dtype=_np.int32)))
                L.append(_bake_vec(f"NF_R{tag}Z",
                                   _np.ascontiguousarray(cz, dtype=_np.uint8)))
        if any(mn in FX_MN for mn in ctx.get("mns", [])):
            cfg = ctx.get("config", {})
            m_cov = int(cfg.get("m_cov", _FROZEN_COV))
            m_acc = int(cfg.get("m_acc", _FROZEN_ACC))
            mns = ctx.get("mns", [])
            need_sq = "SQUARE" in mns
            need_sm = any(mn in SM_MN or mn == "RMSNORM" for mn in mns)
            need_silu = "SILU" in mns
            eps_c = self._eps_counts(cfg, m_cov)
            L.append(fx_tables_and_helpers(m_cov, m_acc, need_sq,
                                           need_sm, eps_c, need_silu))
        return "\n".join(L) + "\n"

    @staticmethod
    def _eps_counts(cfg, m_cov):
        """eps_rms (true float) -> ambient counts (mirror op_rmsnorm).
        Legacy eps_rms_c honored; default 4514. Compile-time only."""
        import math as _math
        import phi_core.lattice as _S
        if "eps_rms" in cfg:
            _um = _S.PHI ** ((m_cov - _S.BIAS) / _S.K)
            return int(round(float(cfg["eps_rms"]) * float(1 << 36)
                             / (_um * _um)))
        return int(cfg.get("eps_rms_c", 4514))

    def pattern(self, mn, outs, args, sig, ctx):
        if mn not in INTEGER_MN:
            raise NoPattern(
                f"nonfpu: no integer pattern for {mn} (fixed-point core "
                f"covers moves + arithmetic + norms + softmax; "
                f"layernorm/silu/gelu/convs ahead)")
        S = ctx["streams"]
        for a in args:
            if a in S and S[a][0] == "F":
                raise NoPattern(
                    f"nonfpu: float stream '{a}' refused (no FPU on this "
                    f"target -- decode nothing, quantize first)")
        if mn in FX_MN:
            return self._fx_pattern(mn, outs, args, sig, ctx)
        return super().pattern(mn, outs, args, sig, ctx)

    @staticmethod
    def _tplane(stream, comp, idx, S, ctx):
        """Triple plane ref: array element, or baked literal when the
        stream folded to a constant."""
        consts = ctx.get("consts", {})
        if stream in consts:
            cs, ce, cz = consts[stream]
            lit = {"s": f"(int8_t){cs}", "e": f"(int32_t){ce}",
                   "z": f"(uint8_t){cz}"}[comp]
            return lit
        return f"{stream}_{comp}[{idx}]"

    def _fx_pattern(self, mn, outs, args, sig, ctx):
        S = ctx["streams"]
        o = outs[0]
        if mn == "ARGMAX":
            src = args[0]
            if src in ctx.get("gathered", {}):
                return super().pattern(mn, outs, args, sig, ctx)
            if S[src][0] != "T" or len(S[src][2]) != 2:
                raise NoPattern("nonfpu ARGMAX needs 2D T")
            w = S[src][2][1]
            we = str(int(w)) if w is not None else ctx["ids_n"]
            return (
                f"/* {o} = ARGMAX({src}) lattice class/exponent order, "
                f"ties first */\n"
                f"for (int64_t a_i = 0; a_i < {o}_N; ++a_i) {{\n"
                f"  int64_t best = 0; int bcls = -1; int64_t bkey = 0;\n"
                f"  for (int64_t a_j = 0; a_j < {we}; ++a_j) {{\n"
                f"    int8_t s = {src}_s[a_i * {we} + a_j];\n"
                f"    int32_t e = {src}_e[a_i * {we} + a_j];\n"
                f"    uint8_t z = {src}_z[a_i * {we} + a_j];\n"
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
        if mn in ("TBETA", "BETA"):
            ref = args[0]
            if S[ref][0] != "T":
                raise NoPattern(f"nonfpu {mn} needs T ref")
            import phi_core.lattice as _S
            import numpy as _np
            cfg = ctx["config"]
            val = float(cfg.get("beta_b", float(cfg.get("beta", 0.5)))) \
                if mn == "TBETA" else float(cfg.get("beta", 0.5))
            cs, ce, cz = _S.encode(_np.array([val]))
            co = ctx["count"][o]
            return (
                f"/* {o} = {mn} ({val}) baked triples */\n"
                f"for (int64_t k_i = 0; k_i < {co}; ++k_i) {{\n"
                f"  {o}_s[k_i] = (int8_t){int(cs[0])};\n"
                f"  {o}_e[k_i] = (int32_t){int(ce[0])};\n"
                f"  {o}_z[k_i] = (uint8_t){int(cz[0])};\n}}",
                None)
        if mn == "ROTARY":
            x, pos = args
            if S[x][0] != "T" or S[pos][0] != "I":
                raise NoPattern("nonfpu ROTARY needs T + I pos")
            d = S[x][2][-1]
            if d is None:
                raise NoPattern("nonfpu ROTARY needs baked head dim")
            _rope = ctx.get("rope") or {}
            if _rope.get("D") != d:
                raise NoPattern("nonfpu ROTARY needs baked tables "
                                "(compile-time rope)")
            return (
                f"/* {o} = ROTARY({x}) tmul/binop pairs, exact */\n"
                f"for (int64_t t_i = 0; t_i < {o}_N; ++t_i) {{\n"
                f"  int64_t pp = {pos}[t_i];\n"
                f"  if (pp < 0 || pp >= NF_RP) return 31;\n"
                f"  for (int64_t t_h = 0; t_h < {d // 2}; ++t_h) {{\n"
                f"    int8_t a0s = {x}_s[(t_i * {d}) + 2 * t_h];\n"
                f"    int32_t a0e = {x}_e[(t_i * {d}) + 2 * t_h];\n"
                f"    uint8_t a0z = {x}_z[(t_i * {d}) + 2 * t_h];\n"
                f"    int8_t a1s = {x}_s[(t_i * {d}) + 2 * t_h + 1];\n"
                f"    int32_t a1e = {x}_e[(t_i * {d}) + 2 * t_h + 1];\n"
                f"    uint8_t a1z = {x}_z[(t_i * {d}) + 2 * t_h + 1];\n"
                f"    int64_t co = pp * {d // 2} + t_h;\n"
                f"    int8_t y0s, y1s; int32_t y0e, y1e; uint8_t y0z, y1z;\n"
                f"    int8_t t0s, t1s;\n"
                f"    int32_t t0e, t1e; uint8_t t0z, t1z;\n"
                f"    nf_tmul(a0s, a0e, a0z, NF_RCS[co], NF_RCE[co], "
                f"NF_RCZ[co], &t0s, &t0e, &t0z);\n"
                f"    nf_tmul(a1s, a1e, a1z, NF_RSS[co], NF_RSE[co], "
                f"NF_RSZ[co], &t1s, &t1e, &t1z);\n"
                f"    nf_binop(t0s, t0e, t0z, t1s, t1e, t1z, 1,\n"
                f"      &y0s, &y0e, &y0z);\n"
                f"    nf_tmul(a0s, a0e, a0z, NF_RSS[co], NF_RSE[co], "
                f"NF_RSZ[co], &t0s, &t0e, &t0z);\n"
                f"    nf_tmul(a1s, a1e, a1z, NF_RCS[co], NF_RCE[co], "
                f"NF_RCZ[co], &t1s, &t1e, &t1z);\n"
                f"    nf_binop(t0s, t0e, t0z, t1s, t1e, t1z, 0,\n"
                f"      &y1s, &y1e, &y1z);\n"
                f"    {o}_s[(t_i * {d}) + 2 * t_h] = y0s;\n"
                f"    {o}_e[(t_i * {d}) + 2 * t_h] = y0e;\n"
                f"    {o}_z[(t_i * {d}) + 2 * t_h] = y0z;\n"
                f"    {o}_s[(t_i * {d}) + 2 * t_h + 1] = y1s;\n"
                f"    {o}_e[(t_i * {d}) + 2 * t_h + 1] = y1e;\n"
                f"    {o}_z[(t_i * {d}) + 2 * t_h + 1] = y1z;\n"
                f"  }}\n"
                f"}}",
                None)
        co = ctx["count"][o]
        if mn == "MUL":
            a, b = args
            if S[a][0] != "T" or S[b][0] != "T":
                raise NoPattern("nonfpu MUL needs T triples")
            return (
                f"/* {o} = MUL({a},{b}) tmul exact */\n"
                f"for (int64_t f_i = 0; f_i < {co}; ++f_i)\n"
                f"  nf_tmul({self._tplane(a, 's', 'f_i', S, ctx)}, "
                f"{self._tplane(a, 'e', 'f_i', S, ctx)}, "
                f"{self._tplane(a, 'z', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 's', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 'e', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 'z', 'f_i', S, ctx)}, "
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
                f"  nf_binop({self._tplane(a, 's', 'f_i', S, ctx)}, "
                f"{self._tplane(a, 'e', 'f_i', S, ctx)}, "
                f"{self._tplane(a, 'z', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 's', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 'e', 'f_i', S, ctx)}, "
                f"{self._tplane(b, 'z', 'f_i', S, ctx)}, {sub}, "
                f"&{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);",
                None)
        if mn == "SQUARE":
            a = args[0]
            if S[a][0] != "T":
                raise NoPattern("nonfpu SQUARE needs T triples")
            co = ctx["count"][o]
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
        if mn == "RMSNORM":
            x, w = args
            if S[x][0] != "T" or S[w][0] != "T":
                raise NoPattern("nonfpu RMSNORM needs T triples")
            o = outs[0]
            d = S[x][2][-1]
            if d is None:
                raise NoPattern("nonfpu RMSNORM needs baked width")
            return (
                f"/* {o} = RMSNORM({x},{w}) integer rows @ NF_M */\n"
                f"{{ int64_t *_q = malloc({d} * 8);\n"
                f"  int64_t *_wq = malloc({d} * 8);\n"
                f"  for (int64_t r_i = 0; r_i < {o}_N; ++r_i)\n"
                f"    nf_rmsnorm_row(&{x}_s[r_i * {d}], &{x}_e[r_i * {d}],\n"
                f"      &{x}_z[r_i * {d}], {w}_s, {w}_e, {w}_z, {d}, NF_M,\n"
                f"      &{o}_s[r_i * {d}], &{o}_e[r_i * {d}], "
                f"&{o}_z[r_i * {d}], _q, _wq);\n"
                f"  free(_q); free(_wq); }}",
                None)
        if mn == "TSHIFT":
            x = args[0]
            if S[x][0] != "T":
                raise NoPattern("nonfpu TSHIFT needs T triples")
            o = outs[0]
            _dw = S[x][2][-1]
            d = str(int(_dw)) if _dw is not None else ctx["ids_n"]
            return (
                f"/* {o} = TSHIFT({x}) wide bridge @ NF_MACC */\n"
                f"{{ int64_t *_q = malloc({d} * 8);\n"
                f"  for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {{\n"
                f"    for (int64_t f_j = 0; f_j < {d}; ++f_j)\n"
                f"      _q[f_j] = nf_to_fixed_wide({x}_s[f_i * {d} + f_j],\n"
                f"        {x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], "
                f"NF_MACC);\n"
                f"    int64_t _mx = _q[0];\n"
                f"    for (int64_t f_j = 1; f_j < {d}; ++f_j)\n"
                f"      if (_q[f_j] > _mx) _mx = _q[f_j];\n"
                f"    for (int64_t f_j = 0; f_j < {d}; ++f_j)\n"
                f"      nf_from_fixed(_q[f_j] - _mx, NF_MACC,\n"
                f"        &{o}_s[f_i * {d} + f_j], &{o}_e[f_i * {d} + f_j],\n"
                f"        &{o}_z[f_i * {d} + f_j]);\n"
                f"  }}\n"
                f"  free(_q); }}",
                None)
        if mn in ("SOFTMAX_WIDE", "SOFTMAX"):
            x = args[0]
            if S[x][0] != "T":
                raise NoPattern(f"nonfpu {mn} needs T triples")
            o = outs[0]
            _dw = S[x][2][-1]
            d = str(int(_dw)) if _dw is not None else ctx["ids_n"]
            bridge = ("nf_to_fixed_wide({x}_s[f_i * {d} + f_j], "
                      "{x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], "
                      "32768)" if mn == "SOFTMAX_WIDE"
                      else "nf_to_fixed({x}_s[f_i * {d} + f_j], "
                      "{x}_e[f_i * {d} + f_j], {x}_z[f_i * {d} + f_j], "
                      "32768)").format(x=x, d=d)
            return (
                f"/* {o} = {mn}({x}) row softmax @ BIAS */\n"
                f"{{ int64_t *_q = malloc({d} * 8);\n"
                f"  int64_t *_num = malloc({d} * 8);\n"
                f"  for (int64_t f_i = 0; f_i < {o}_N; ++f_i) {{\n"
                f"    for (int64_t f_j = 0; f_j < {d}; ++f_j)\n"
                f"      _q[f_j] = {bridge};\n"
                f"    nf_softmax_row(_q, {d}, &{o}_s[f_i * {d}],\n"
                f"      &{o}_e[f_i * {d}], &{o}_z[f_i * {d}], _num);\n"
                f"  }}\n"
                f"  free(_q); free(_num); }}",
                None)
        if mn == "SILU":
            a = args[0]
            if S[a][0] != "T":
                raise NoPattern("nonfpu SILU needs T triples")
            co = ctx["count"][o]
            return (
                f"/* {o} = SILU({a}) sigmoid-LUT + tmul, exact */\n"
                f"for (int64_t f_i = 0; f_i < {co}; ++f_i)\n"
                f"  nf_silu({a}_s[f_i], {a}_e[f_i], {a}_z[f_i], "
                f"&{o}_s[f_i], &{o}_e[f_i], &{o}_z[f_i]);",
                None)
        raise NoPattern(f"nonfpu: no fx pattern for {mn}")
