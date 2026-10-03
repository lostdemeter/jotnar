"""CUDA backend v0.1 (first cut): kernels + cuBLAS, FP32 on device.

Same bound IR + shape language as the C backend (chain.emit_c reused:
infer_shapes, literals, NoPattern contract). Value model: F -> float
(FP32 device; host samples are float64, dumped as float32), I ->
int64_t, T -> NoPattern (decode at data-prep, same discipline as C).

Kernels (correctness-first; block-reduce/streams/FP16 are listed
finishing touches, not this cut):
  k_gather_f32, k_argmax_f32 (ties-first), k_rmsnorm_f32,
  k_softmax_wide_f32, k_elem (op-switched template: add/sub/mul/div).
MATMUL/BATCH_MATMUL go to cuBLAS Sgemm/SgemmStridedBatched with the
row-major trick (C_row = A@B computed as C_col = B_col@A_col).
Memory v1: cudaMallocManaged throughout (host fread straight in,
implicit download after sync). Agreement: vs C backend tight eps,
vs lattice leveled (same story as C).
"""
import os
import subprocess

import numpy as np

from chain.asm import AsmError
from chain.emit_c import Backend, NoPattern, _dim_expr


KERNELS = r"""
#define TPB 256
__global__ void k_gather_f32(const float *t, const int64_t *ids, float *o,
                             int64_t C, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float *row = t + ids[i] * C;
  float *dst = o + i * C;
  for (int64_t j = 0; j < C; ++j) dst[j] = row[j];
}
__global__ void k_argmax_f32(const float *x, int64_t *o, int64_t C,
                             int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int64_t best = 0; float bv = x[i * C];
  for (int64_t j = 1; j < C; ++j)
    if (x[i * C + j] > bv) { bv = x[i * C + j]; best = j; }
  o[i] = best;
}
__global__ void k_rmsnorm_f32(const float *x, const float *w, float *o,
                              int64_t D, int64_t N, float eps) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double ss = 0;
  for (int64_t j = 0; j < D; ++j) { double v = x[i * D + j]; ss += v * v; }
  float rs = (float)(1.0 / sqrt(ss / (double)D + (double)eps));
  for (int64_t j = 0; j < D; ++j) o[i * D + j] = x[i * D + j] * rs * w[j];
}
__global__ void k_softmax_wide_f32(const float *x, float *o, int64_t D,
                                   int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float mx = x[i * D];
  for (int64_t j = 1; j < D; ++j) mx = fmaxf(mx, x[i * D + j]);
  double se = 0;
  for (int64_t j = 0; j < D; ++j) {
    float e = expf(x[i * D + j] - mx);
    o[i * D + j] = e; se += (double)e;
  }
  for (int64_t j = 0; j < D; ++j) o[i * D + j] /= (float)se;
}
/* op: 0 add, 1 sub, 2 mul, 3 div -- one template, many mnemonics. */
__global__ void k_elem(const float *a, const float *b, float *o, int64_t n,
                       int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float r = 0;
  if (op == 0) r = a[i] + b[i];
  else if (op == 1) r = a[i] - b[i];
  else if (op == 2) r = a[i] * b[i];
  else r = a[i] / b[i];
  o[i] = r;
}
__global__ void k_transpose_f32(const float *x, float *o, int64_t R,
                                int64_t C) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * C) return;
  int64_t r = i / C, c = i % C;
  o[c * R + r] = x[i];
}
__global__ void k_slice_ax1_f32(const float *x, float *o, int64_t W,
                                int64_t LO, int64_t WID, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * WID) return;
  int64_t r = i / WID, j = i % WID;
  o[i] = x[r * W + LO + j];
}
__global__ void k_concat_ax1_f32(const float *a, const float *b, float *o,
                                 int64_t DA, int64_t DB, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t W = DA + DB;
  if (i >= N * W) return;
  int64_t r = i / W, j = i % W;
  o[i] = (j < DA) ? a[r * DA + j] : b[r * DB + (j - DA)];
}
__global__ void k_select_f32(const int64_t *m, const float *a, const float *b,
                             float *o, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = m[i] ? a[i] : b[i];
}
__global__ void k_rotary_f32(const float *x, const int64_t *pos, float *o,
                             int64_t D, int64_t N, const float *theta) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t H = D / 2;
  if (i >= N * H) return;
  int64_t r = i / H, h = i % H;
  float ang = (float)pos[r] * theta[h];
  float c = cosf(ang), s = sinf(ang);
  float x0 = x[r * D + 2 * h], x1 = x[r * D + 2 * h + 1];
  o[r * D + 2 * h] = x0 * c - x1 * s;
  o[r * D + 2 * h + 1] = x0 * s + x1 * c;
}
__global__ void k_tshift_f32(const float *x, float *o, int64_t D, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float mx = x[i * D];
  for (int64_t j = 1; j < D; ++j) mx = fmaxf(mx, x[i * D + j]);
  for (int64_t j = 0; j < D; ++j) o[i * D + j] = x[i * D + j] - mx;
}
__global__ void k_fill_f32(float *o, int64_t n, float v) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = v;
}
"""


class CUDABackend(Backend):
    """CUDA-F backend. cuBLAS owns matmul; hand kernels own the rest."""

    name = "cuda"
    emits = True
    DEFAULT_NVCC = ("-O2", "-std=c++17")
    DEFAULT_ARCH = "sm_86"

    def value_repr(self, layout):
        kind = (layout or "").split(":")[0]
        return {"I": "int64_t", "F": "float"}.get(kind, "void/*T*/")

    def prologue(self, ctx):
        th = ctx.get("theta", [])
        thal = ("static const float THETA[] = {%s};" %
                ", ".join(repr(float(t)) + "f" for t in th)) if th else ""
        return ("\n".join([
            "#include <stdint.h>", "#include <stdio.h>",
            "#include <stdlib.h>", "#include <string.h>",
            "#include <cuda_runtime.h>", "#include <cublas_v2.h>", "",
            "#define CE(x) do { cudaError_t _e = (x); if (_e) { fprintf(stderr, \"cuda %d\\n\", (int)_e); return 10; } } while (0)",
            "#define BE(x) do { cublasStatus_t _e = (x); if (_e) { fprintf(stderr, \"cublas %d\\n\", (int)_e); return 11; } } while (0)",
            KERNELS,
            thal,
        ]) + "\n")

    @staticmethod
    def _we(S, x, ctx, where):
        w = S[x][2][-1]
        return str(int(w)) if w is not None else ctx["ids_n"]

    def pattern(self, mn, outs, args, sig, ctx):
        from chain.emit_c import CBackend as _CB
        S = ctx["streams"]
        cfg = ctx["config"]
        ids_n = ctx["ids_n"]
        launch = ("CE(cudaGetLastError()); CE(cudaDeviceSynchronize());")
        if mn == "GATHER":
            t, ids = args
            if S[t][0] != "F":
                raise NoPattern(f"cuda GATHER needs F table (v0.1; "
                                f"decode at data-prep)")
            w = self._we(S, t, ctx, f"GATHER {outs[0]}")
            return (
                f"/* {outs[0]} = GATHER({t}, {ids}) */\n"
                f"k_gather_f32<<<({ids}_N + TPB - 1) / TPB, TPB>>>"
                f"({t}, {ids}, {outs[0]}, {w}, {ids}_N);\n{launch}",
                (outs[0], w))
        if mn == "ARGMAX":
            src = args[0]
            if S[src][0] != "F" or len(S[src][2]) != 2:
                raise NoPattern("cuda ARGMAX needs 2D F (v0.1)")
            w = self._we(S, src, ctx, f"ARGMAX {outs[0]}")
            o = outs[0]
            return (
                f"/* {o} = ARGMAX({src}) */\n"
                f"k_argmax_f32<<<({o}_N + TPB - 1) / TPB, TPB>>>"
                f"({src}, {o}, {w}, {o}_N);\n{launch}",
                None)
        if mn == "RMSNORM":
            x, w = args
            if S[x][0] != "F" or S[w][0] != "F":
                raise NoPattern("cuda RMSNORM needs F (v0.1)")
            if "eps_rms" not in cfg:
                raise AsmError("cuda RMSNORM needs CONFIG eps_rms")
            o = outs[0]
            d = self._we(S, x, ctx, f"RMSNORM {o}")
            eps = float(cfg["eps_rms"])
            return (
                f"/* {o} = RMSNORM({x}) */\n"
                f"k_rmsnorm_f32<<<({o}_N + TPB - 1) / TPB, TPB>>>"
                f"({x}, {w}, {o}, {d}, {o}_N, {eps!r}f);\n{launch}",
                None)
        if mn == "SOFTMAX_WIDE":
            x = args[0]
            if S[x][0] != "F":
                raise NoPattern("cuda SOFTMAX_WIDE needs F (v0.1)")
            o = outs[0]
            d = self._we(S, x, ctx, f"SOFTMAX_WIDE {o}")
            return (
                f"/* {o} = SOFTMAX_WIDE({x}) */\n"
                f"k_softmax_wide_f32<<<({o}_N + TPB - 1) / TPB, TPB>>>"
                f"({x}, {o}, {d}, {o}_N);\n{launch}",
                None)
        if mn in ("MUL", "ADD", "SUB", "DIV"):
            a, b = args
            if S[a][0] != "F" or S[b][0] != "F" \
                    or tuple(S[a][2]) != tuple(S[b][2]):
                raise NoPattern(f"cuda {mn} needs same-shape F (v0.1)")
            o = outs[0]
            op = {"ADD": 0, "SUB": 1, "MUL": 2, "DIV": 3}[mn]
            co = ctx["count"][o]
            return (
                f"/* {o} = {mn} template op={op} */\n"
                f"k_elem<<<({co} + TPB - 1) / TPB, TPB>>>"
                f"({a}, {b}, {o}, {co}, {op});\n{launch}",
                None)
        if mn in ("MATMUL", "BATCH_MATMUL"):
            a, b = args
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"cuda {mn} needs F (v0.1)")
            o = outs[0]
            ash, bsh = S[a][2], S[b][2]
            # alpha lives in managed-symbol storage (no literal addresses).
            if len(ash) == 2:
                m, k, n = ash[0], ash[1], bsh[1]
                me = str(m) if m is not None else ids_n
                ke = str(k) if k is not None else ids_n
                ne = str(n) if n is not None else ids_n
                return (
                    f"/* {o} = {mn} cuBLAS */\n"
                    f"BE(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, {ne}, "
                    f"{me}, {ke}, &kOne, {b}, {ne}, {a}, {ke}, &kOne, "
                    f"{o}, {ne}));\n{launch}",
                    None)
            B = ash[0]
            if B is None:
                raise NoPattern("cuda batched matmul needs baked batch")
            m, k, n = ash[1], ash[2], bsh[2]
            me = str(m) if m is not None else ids_n
            ke = str(k) if k is not None else ids_n
            ne = str(n) if n is not None else ids_n
            return (
                f"/* {o} = {mn} cuBLAS strided-batched */\n"
                f"BE(cublasSgemmStridedBatched(h, CUBLAS_OP_N, "
                f"CUBLAS_OP_N, {ne}, {me}, {ke}, &kOne, {b}, {ne}, "
                f"{ke}*{ne}, {a}, {ke}, {me}*{ke}, &kOne, {o}, {ne}, "
                f"{me}*{ne}, {B}));\n{launch}",
                None)
        if mn == "TRANSPOSE":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F" or len(S[x][2]) != 2:
                raise NoPattern("cuda TRANSPOSE needs 2D F (v0.1)")
            rsh = S[x][2]
            re_ = str(rsh[0]) if rsh[0] is not None else ids_n
            ce = str(rsh[1]) if rsh[1] is not None else ids_n
            return (
                f"/* {o} = TRANSPOSE({x}) */\n"
                f"k_transpose_f32<<<(({re_}*{ce}) + TPB - 1) / TPB, TPB>>>"
                f"({x}, {o}, {re_}, {ce});\n{launch}",
                None)
        if mn == "SLICE":
            from chain.emit_c import _int_lit as _il
            x = args[0]
            o = outs[0]
            ax, lo, hi = (_il(args[1], "SLICE ax"), _il(args[2], "SLICE lo"),
                          _il(args[3], "SLICE hi"))
            if S[x][0] != "F" or len(S[x][2]) != 2 or ax != 1:
                raise NoPattern("cuda SLICE needs 2D ax1 F (v0.1)")
            w = S[x][2][1]
            we = str(w) if w is not None else ids_n
            return (
                f"/* {o} = SLICE({x},1,{lo},{hi}) */\n"
                f"k_slice_ax1_f32<<<(({o}_N * {hi - lo}) + TPB - 1) / TPB, "
                f"TPB>>>({x}, {o}, {we}, {lo}, {hi - lo}, {o}_N);\n{launch}",
                None)
        if mn == "CONCAT":
            from chain.emit_c import _int_lit as _il
            a, b = args[0], args[1]
            o = outs[0]
            ax = _il(args[2], "CONCAT axis") if args[2] not in S else None
            if S[a][0] != "F" or S[b][0] != "F" or ax != 1:
                raise NoPattern("cuda CONCAT needs 2D ax1 F (v0.1)")
            da, db = S[a][2][1], S[b][2][1]
            if da is None or db is None:
                raise NoPattern("cuda CONCAT needs baked widths (v0.1)")
            return (
                f"/* {o} = CONCAT({a},{b},1) */\n"
                f"k_concat_ax1_f32<<<(({o}_N * ({da} + {db})) + TPB - 1) / "
                f"TPB, TPB>>>({a}, {b}, {o}, {da}, {db}, {o}_N);\n{launch}",
                None)
        if mn == "SELECT":
            m, a, b = args
            o = outs[0]
            if S[m][0] != "I" or S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern("cuda SELECT needs I mask + F (v0.1)")
            co = ctx["count"][o]
            msh = S[m][2]
            guard = ""
            if all(d is not None for d in msh):
                import numpy as _np
                mc = int(_np.prod(msh, dtype=_np.int64))
                guard = f"if ((int64_t){mc} != {co}) return 21;\n  "
            return (
                f"/* {o} = SELECT({m}) */\n  {guard}"
                f"k_select_f32<<<(({co}) + TPB - 1) / TPB, TPB>>>"
                f"({m}, {a}, {b}, {o}, {co});\n{launch}",
                None)
        if mn == "ROTARY":
            x, pos = args
            o = outs[0]
            if S[x][0] != "F" or S[pos][0] != "I":
                raise NoPattern("cuda ROTARY needs F + I pos (v0.1)")
            d = S[x][2][1]
            if d is None:
                raise NoPattern("cuda ROTARY needs baked head dim")
            return (
                f"/* {o} = ROTARY({x}) */\n"
                f"k_rotary_f32<<<(({o}_N * ({d} / 2)) + TPB - 1) / TPB, "
                f"TPB>>>({x}, {pos}, {o}, {d}, {o}_N, THETA);\n{launch}",
                None)
        if mn == "TSHIFT":
            x = args[0]
            o = outs[0]
            if S[x][0] != "F":
                raise NoPattern("cuda TSHIFT needs F (v0.1)")
            d = S[x][2][1]
            de = str(d) if d is not None else ids_n
            return (
                f"/* {o} = TSHIFT({x}) */\n"
                f"k_tshift_f32<<<({o}_N + TPB - 1) / TPB, TPB>>>"
                f"({x}, {o}, {de}, {o}_N);\n{launch}",
                None)
        if mn in ("TBETA", "BETA"):
            ref = args[0]
            o = outs[0]
            cfg = ctx["config"]
            val = float(cfg.get("beta_b", float(cfg.get("beta", 0.5)))) \
                if mn == "TBETA" else float(cfg.get("beta", 0.5))
            co = ctx["count"][o]
            return (
                f"/* {o} = {mn} ({val}) */\n"
                f"k_fill_f32<<<(({co}) + TPB - 1) / TPB, TPB>>>"
                f"({o}, {co}, {val!r}f);\n{launch}",
                None)
        return super().pattern(mn, outs, args, sig, ctx)

    def epilogue(self, ctx):
        return ""


def compile_cuda(text, sample=None, outputs=None, registry=None,
                 sigs=None, basedir="."):
    """CUDA frontend entry: asm text -> .cu source."""
    from chain.asm import assemble
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    from chain.emit_c import infer_shapes
    config, inp, bound, _ = assemble(text, registry or _R, sigs or _S,
                                     basedir=basedir)
    if sample is None:
        raise AsmError("compile_cuda needs sample payload (shapes)")
    if not outputs:
        raise AsmError("compile_cuda needs outputs=[...]")
    be = CUDABackend()
    streams = infer_shapes((config, inp, bound), sample)
    in_names = [n for n, _ in inp]
    ids_n = None
    for n in in_names:
        k, _, sh = streams[n]
        if k == "I" and len(sh) == 1:
            ids_n = f"{n}_N"
            break
    if ids_n is None:
        # No (N,) id input: batch rides a dynamic-row F input (rows =
        # bytes / baked row width; fully-dynamic inputs refused).
        for n in in_names:
            k, _, sh = streams[n]
            if k == "F" and sh and sh[0] is None and sh[-1] is not None:
                ids_n = f"{n}_N"
                break
    if ids_n is None:
        ids_n = "1"
    theta = []
    for _on, _mn, _fn, _aa, _ln, _sg in bound:
        if _mn == "ROTARY":
            _d = streams[_aa[0]][2][1]
            _base = float(config.get("rope_base", 10000.0))
            _th = np.power(float(_base),
                           -2.0 * np.arange(_d // 2, dtype=np.float64) / _d)
            theta = [float(t) for t in _th]
            break
    ctx = {"streams": streams, "dims": {}, "dyn": {}, "ids_n": ids_n,
           "inputs": in_names, "outputs": outputs, "gathered": {},
           "config": dict(config), "theta": theta, "count": {}}
    for xn, (xk, _xs, xsh) in streams.items():
        if xk == "F" and xsh:
            ctx["count"][xn] = "(%s)" % _dim_expr(xsh, ids_n)
        elif xk == "I" and xn in outputs:
            if xsh and all(d is not None for d in xsh):
                ctx["count"][xn] = "(%s)" % int(
                    np.prod(list(xsh), dtype=np.int64))
            else:
                ctx["count"][xn] = "(%s)" % ids_n
    parts = [be.prologue(ctx)]
    decls = []
    for n in in_names:
        k, sub, sh = streams[n]
        if k == "F":
            decls.append(f"static float *{n}; static int64_t {n}_N;")
        elif k == "T":
            raise NoPattern(f"cuda: T input '{n}' -- decode at data-prep")
        else:
            decls.append(f"static int64_t *{n}; static int64_t {n}_N;")
    for o in outputs:
        ok, _, _ = streams[o]
        decls.append("static float *%s;" % o if ok == "F"
                     else f"static int64_t *{o}; static int64_t {o}_N;")
    inter = set()
    _, _, prog = (config, inp, bound)
    for outs, _m2, _f2, _a2, _l2, _s2 in prog:
        for x in outs:
            if x not in in_names and x not in outputs \
                    and streams[x][0] == "F":
                inter.add(x)
    if inter:
        parts.append("\n".join(decls))
        parts.append("/* intermediates (managed) */")
        parts.append("\n".join(f"float *{x};" for x in sorted(inter)))
    else:
        parts.append("\n".join(decls))
    parts.append("static cublasHandle_t h;")
    parts.append("static const float kOne = 1.0f;")
    parts.append("int main(int argc, char **argv) {")
    n_argv = len(in_names) + len(outputs)
    parts.append(f"  if (argc != {n_argv + 1}) return 9;")
    parts.append("  CE(cudaSetDevice(0)); BE(cublasCreate(&h));")
    parts.append("  BE(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));")
    idx = 1
    loads = []
    for n in in_names:
        k, _, _ = streams[n]
        if k == "F":
            wrow = sh[-1]
            if wrow is None:
                raise NoPattern(f"cuda: fully-dynamic F input '{n}' "
                                f"(batch x width both unknown)")
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); long {n}_B = ftell(f);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n}_N = {n}_B / ({int(wrow)} * 4);\n"
                f"    CE(cudaMallocManaged((void**)&{n}, {n}_B));\n"
                f"    if (fread({n}, 1, {n}_B, f) != (size_t){n}_B) return 2;"
                f" fclose(f); }}")
        else:
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); {n}_N = ftell(f)/8;"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    CE(cudaMallocManaged((void**)&{n}, {n}_N * 8));\n"
                f"    if (fread({n}, 8, {n}_N, f) != (size_t){n}_N) return 2;"
                f" fclose(f); }}")
        idx += 1
    parts.append("\n".join(loads))
    allocs = []
    for x in sorted(inter) + [o for o in outputs if streams[o][0] == "F"]:
        _k, _s, xsh = streams[x]
        cnt = _dim_expr(xsh, ids_n)
        allocs.append(f"  CE(cudaMallocManaged((void**)&{x}, ({cnt}) * 4));")
        # ROW-count var (first dim); element counts live in count[].
        from chain.emit_c import _rows_expr as _rr
        allocs.append(f"  int64_t {x}_N = {_rr(xsh, ids_n)};")
    for o in outputs:
        if streams[o][0] == "I":
            _osh = streams[o][2]
            _on = ("(%s)" % ids_n) if any(d is None for d in _osh) \
                else str(int(np.prod(list(_osh), dtype=np.int64)))
            allocs.append(f"  {o}_N = {_on}; "
                          f"CE(cudaMallocManaged((void**)&{o}, {o}_N * 8));")
    parts.append("\n".join(allocs))
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        parts.append(f"  /* L{ln}: {mn} */")
        parts.append("  " + pat.replace("\n", "\n  "))
    for o in outputs:
        ok = streams[o][0]
        el = "8" if ok == "I" else "4"
        co = ctx["count"][o]
        parts.append(f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                     f" return 3;\n"
                     f"    CE(cudaDeviceSynchronize());\n"
                     f"    fwrite({o}, {el}, {co}, f); fclose(f); }}")
        idx += 1
    parts.append("  BE(cublasDestroy(h));")
    parts.append("  return 0;\n}")
    parts.append(be.epilogue(ctx))
    return {"source": "\n".join(parts), "backend": be, "streams": streams,
            "dims": {}, "config": dict(config), "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1}


def build_cu(source, workdir, name="prog", nvcc="nvcc", nvflags=None,
             arch=None, libs=None):
    """Compile emitted .cu -> exe. Arch flag IS the target selection."""
    arch = arch or os.environ.get("CUDA_ARCH", CUDABackend.DEFAULT_ARCH)
    nvflags = list(nvflags or CUDABackend.DEFAULT_NVCC)
    libs = list(libs or ["-lcublas"])
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".cu")
    exe = os.path.join(workdir, name)
    with open(src, "w") as f:
        f.write(source)
    archf = f"-arch={arch}" if arch else None
    cmd = [nvcc] + nvflags + ([archf] if archf else []) + [src, "-o", exe] \
        + libs
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"nvcc build failed:\n{r.stderr}")
    return exe
