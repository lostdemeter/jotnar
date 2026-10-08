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
from chain.emit_c import Backend, NoPattern, _dim_expr, _short_site


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
__global__ void k_gather_f16(const __half *t, const int64_t *ids, float *o,
                             int64_t C, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int64_t base = ids[i] * C;
  float *dst = o + i * C;
  for (int64_t j = 0; j < C; ++j) dst[j] = __half2float(t[base + j]);
}
__global__ void k_rmsnorm_w16(const float *x, const __half *w, float *o,
                              int64_t D, int64_t N, float eps) {
  /* One BLOCK per row (not one thread): cooperative reduction over D.
  The old one-thread form cost ~400us/row in serial loop + launch. */
  int64_t r = (int64_t)blockIdx.x;
  if (r >= N) return;
  __shared__ float sh[TPB];
  double ss = 0;
  for (int64_t j = threadIdx.x; j < D; j += TPB) {
    double v = x[r * D + j];
    ss += v * v;
  }
  sh[threadIdx.x] = (float)ss;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float rs = (float)(1.0 / sqrt((double)sh[0] / (double)D + (double)eps));
  float wj;
  for (int64_t j = threadIdx.x; j < D; j += TPB) {
    wj = __half2float(w[j]);
    o[r * D + j] = x[r * D + j] * rs * wj;
  }
}
__global__ void k_cvt_f16(const __half *x, float *o, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = __half2float(x[i]);
}
__global__ void k_cvt_f16_from_f32(const float *x, __half *o, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  o[i] = __float2half(x[i]);
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
  /* One BLOCK per row: cooperative reduction (see w16 variant). */
  int64_t r = (int64_t)blockIdx.x;
  if (r >= N) return;
  __shared__ float sh[TPB];
  double ss = 0;
  for (int64_t j = threadIdx.x; j < D; j += TPB) {
    double v = x[r * D + j];
    ss += v * v;
  }
  sh[threadIdx.x] = (float)ss;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float rs = (float)(1.0 / sqrt((double)sh[0] / (double)D + (double)eps));
  for (int64_t j = threadIdx.x; j < D; j += TPB)
    o[r * D + j] = x[r * D + j] * rs * w[j];
}
__global__ void k_softmax_wide_f32(const float *x, float *o, int64_t D,
                                   int64_t N) {
  /* One BLOCK per row: parallel max + exp-sum reductions. */
  int64_t r = (int64_t)blockIdx.x;
  if (r >= N) return;
  __shared__ float sh[TPB];
  float mx = -1e30f;
  for (int64_t j = threadIdx.x; j < D; j += TPB)
    mx = fmaxf(mx, x[r * D + j]);
  sh[threadIdx.x] = mx;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
  mx = sh[0];
  double se = 0;
  for (int64_t j = threadIdx.x; j < D; j += TPB) {
    float e = expf(x[r * D + j] - mx);
    o[r * D + j] = e;
    se += (double)e;
  }
  sh[threadIdx.x] = (float)se;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  float inv = 1.0f / sh[0];
  for (int64_t j = threadIdx.x; j < D; j += TPB) o[r * D + j] *= inv;
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
/* unary template (op 0 = silu; room for gelu/sigmoid ids). */
__global__ void k_elemU(const float *a, float *o, int64_t n, int op) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float r = 0;
  if (op == 0) r = a[i] / (1.0f + expf(-a[i]));
  o[i] = r;
}
/* const-folded variant: one side is a compile-time scalar. */
__global__ void k_elemC(const float *a, float c, float *o, int64_t n, int op,
                        int c_first) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = a[i], r = 0;
  if (op == 0) r = c_first ? (c + x) : (x + c);
  else if (op == 1) r = c_first ? (c - x) : (x - c);
  else if (op == 2) r = c * x;
  else r = c_first ? (c / x) : (x / c);
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
__global__ void k_slice_ax0_f32(const float *x, float *o, int64_t W,
                                int64_t LO, int64_t N) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * W) return;
  o[i] = x[(i / W + LO) * W + (i % W)];
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
  /* One BLOCK per row: parallel max reduction, then shift. */
  int64_t r = (int64_t)blockIdx.x;
  if (r >= N) return;
  __shared__ float sh[TPB];
  float mx = -1e30f;
  for (int64_t j = threadIdx.x; j < D; j += TPB)
    mx = fmaxf(mx, x[r * D + j]);
  sh[threadIdx.x] = mx;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
    __syncthreads();
  }
  mx = sh[0];
  for (int64_t j = threadIdx.x; j < D; j += TPB) o[r * D + j] = x[r * D + j] - mx;
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
    use_fp16 = False  # F16 storage + FP32 compute (halves VRAM)
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
            "#include <cuda_runtime.h>", "#include <cublas_v2.h>",
            "#include <cuda_fp16.h>", "",
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
        # Per-op device-sync is the default (debug/timing friendly) but
        # costs ~10us x op-count per step. sync_each=False keeps error
        # checking and relies on same-stream ordering (correct: every
        # cross-step/device->host boundary already syncs explicitly --
        # stores sync before fwrite, time_ops events sync per op).
        if ctx.get("sync_each", True):
            launch = ("CE(cudaGetLastError()); CE(cudaDeviceSynchronize());")
        else:
            launch = "CE(cudaGetLastError());"
        st = ctx.get("stream", "")
        if mn == "GATHER":
            t, ids = args
            if S[t][0] != "F":
                raise NoPattern(f"cuda GATHER needs F table (v0.1; "
                                f"decode at data-prep)")
            w = self._we(S, t, ctx, f"GATHER {outs[0]}")
            if getattr(self, "use_fp16", False) and t in ctx.get("inputs", []):
                return (
                    f"/* {outs[0]} = GATHER({t}, {ids}) F16 table */\n"
                    f"k_gather_f16<<<({ids}_N + TPB - 1) / TPB, TPB{st}>>>"
                    f"({t}, {ids}, {outs[0]}, {w}, {ids}_N);\n{launch}",
                    (outs[0], w))
            return (
                f"/* {outs[0]} = GATHER({t}, {ids}) */\n"
                f"k_gather_f32<<<({ids}_N + TPB - 1) / TPB, TPB{st}>>>"
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
                f"k_argmax_f32<<<({o}_N + TPB - 1) / TPB, TPB{st}>>>"
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
            if getattr(self, "use_fp16", False) and w in ctx.get("inputs", []):
                return (
                    f"/* {o} = RMSNORM({x}) F16 weight */\n"
                    f"k_rmsnorm_w16<<<({o}_N), TPB{st}>>>"
                    f"({x}, {w}, {o}, {d}, {o}_N, {eps!r}f);\n{launch}",
                    None)
            return (
                f"/* {o} = RMSNORM({x}) */\n"
                f"k_rmsnorm_f32<<<({o}_N), TPB{st}>>>"
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
                f"k_softmax_wide_f32<<<({o}_N), TPB{st}>>>"
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
            consts = ctx.get("consts", {})
            if a in consts or b in consts:
                if a in consts and b in consts:
                    arr, cval, first = b, float(consts[a]), 1
                elif a in consts:
                    arr, cval, first = b, float(consts[a]), 1
                else:
                    arr, cval, first = a, float(consts[b]), 0
                return (
                    f"/* {o} = {mn} const-folded */\n"
                    f"k_elemC<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                    f"({arr}, {cval!r}f, {o}, {co}, {op}, {first});\n{launch}",
                    None)
            return (
                f"/* {o} = {mn} template op={op} */\n"
                f"k_elem<<<({co} + TPB - 1) / TPB, TPB{st}>>>"
                f"({a}, {b}, {o}, {co}, {op});\n{launch}",
                None)
        if mn == "SILU":
            x = args[0]
            if S[x][0] != "F":
                raise NoPattern("cuda SILU needs F")
            o = outs[0]
            co = ctx["count"][o]
            return (
                f"/* {o} = SILU({x}) */\n"
                f"k_elemU<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({x}, {o}, {co}, 0);\n{launch}",
                None)
        if mn == "CONVERT":
            x = args[0]
            o = outs[0]
            co = ctx["count"][o]
            return (
                f"/* {o} = CONVERT({x}) F16->F32 upcast */\n"
                f"k_cvt_f16<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({x}, {o}, {co});\n{launch}",
                None)
        if mn in ("MATMUL", "BATCH_MATMUL"):
            a, b = args
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern(f"cuda {mn} needs F (v0.1)")
            o = outs[0]
            ash, bsh = S[a][2], S[b][2]
            # alpha lives in managed-symbol storage (no literal addresses).
            _fp = getattr(self, "use_fp16", False)
            _inputs = ctx.get("inputs", [])
            # Operand storage: F16 inputs stay half; F32 streams convert
            # to half temps ONLY when paired with an F16 input (cuBLAS
            # needs uniform A/B types; activations are small, weights
            # stay half -- the VRAM point). Both-F32 pairs use Sgemm.
            def _is16(x):
                return _fp and x in _inputs and S[x][0] == "F"

            def _ty(x):
                return "CUDA_R_16F" if _is16(x) else "CUDA_R_32F"

            def _need_cvt(x, other):
                return (_is16(other) and not _is16(x)
                        and S[x][0] == "F")

            def _cvt(x, cnt):
                # F32 stream -> F16 temp (activation-side convert; weights
                # never convert). Returns (convert code, operand expr).
                # Storage is hoisted (one malloc per stream, outside any
                # loop: per-step mallocs would leak in serve mode).
                t = f"{x}_h16"
                done = ctx.setdefault("h16done", set())
                allocs = ctx.setdefault("h16allocs", {})
                if t in done:
                    return "", t
                done.add(t)
                allocs[t] = cnt
                code = (
                    f"k_cvt_f16_from_f32<<<(({cnt}) + TPB - 1) / TPB, TPB{st}>>>"
                    f"({x}, {t}, {cnt});\n"
                    f"  {launch}\n  ")
                return code, t

            if len(ash) == 2:
                m, k, n = ash[0], ash[1], bsh[1]
                me = str(m) if m is not None else ids_n
                ke = str(k) if k is not None else ids_n
                ne = str(n) if n is not None else ids_n
                if _fp and (_is16(a) or _is16(b)):
                    # Uniform F16 operands for GemmEx: convert any F32
                    # side to a half temp (activations only, small).
                    pre, ea, eb = "", a, b
                    if _need_cvt(a, b):
                        c, ea = _cvt(a, f"({me}) * ({ke})")
                        pre += c
                    if _need_cvt(b, a):
                        c, eb = _cvt(b, f"({ke}) * ({ne})")
                        pre += c
                    return (
                        f"/* {o} = {mn} cublasGemmEx F16->F32 */\n  " + pre +
                        f"BE(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, "
                        f"{ne}, {me}, {ke}, &kOne, {eb}, CUDA_R_16F, {ne}, "
                        f"{ea}, CUDA_R_16F, {ke}, &kZero, {o}, CUDA_R_32F, "
                        f"{ne}, CUBLAS_COMPUTE_32F, "
                        f"CUBLAS_GEMM_DEFAULT));\n{launch}",
                        None)
                return (
                    f"/* {o} = {mn} cuBLAS */\n"
                    f"BE(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, {ne}, "
                    f"{me}, {ke}, &kOne, {b}, {ne}, {a}, {ke}, &kZero, "
                    f"{o}, {ne}));\n{launch}",
                    None)
            B = ash[0]
            if B is None:
                raise NoPattern("cuda batched matmul needs baked batch")
            m, k, n = ash[1], ash[2], bsh[2]
            me = str(m) if m is not None else ids_n
            ke = str(k) if k is not None else ids_n
            ne = str(n) if n is not None else ids_n
            _bb = bsh[0]
            _strideB = "0" if _bb == 1 else f"{ke}*{ne}"
            if _fp and (_is16(a) or _is16(b)) and (_bb == 1 or _bb == ash[0]):
                pre, ea, eb = "", a, b
                if _need_cvt(a, b):
                    c, ea = _cvt(a, f"({B}) * ({me}) * ({ke})")
                    pre += c
                if _need_cvt(b, a):
                    _bcnt = f"({ke}) * ({ne})" if _bb == 1 \
                        else f"({B}) * ({ke}) * ({ne})"
                    c, eb = _cvt(b, _bcnt)
                    pre += c
                return (
                    f"/* {o} = {mn} GemmEx-strided F16->F32 */\n  " + pre +
                    f"BE(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, "
                    f"CUBLAS_OP_N, {ne}, {me}, {ke}, &kOne, {eb}, "
                    f"CUDA_R_16F, {ne}, {_strideB}, {ea}, CUDA_R_16F, {ke}, "
                    f"{me}*{ke}, &kZero, {o}, CUDA_R_32F, {ne}, {me}*{ne}, "
                    f"{B}, CUBLAS_COMPUTE_32F, "
                    f"CUBLAS_GEMM_DEFAULT));\n{launch}",
                    None)
            return (
                f"/* {o} = {mn} cuBLAS strided-batched */\n"
                f"BE(cublasSgemmStridedBatched(h, CUBLAS_OP_N, "
                f"CUBLAS_OP_N, {ne}, {me}, {ke}, &kOne, {b}, {ne}, "
                f"{_strideB}, {a}, {ke}, {me}*{ke}, &kZero, {o}, {ne}, "
                f"{me}*{ne}, {B}));\n{launch}",
                None)
        if mn == "BMMV":
            from chain.emit_c import _int_lit as _il
            a, b = args[0], args[1]
            o = outs[0]
            if S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern("cuda BMMV needs F bases (v0.1)")
            (BB, M, N_, K, LAA, SA, LAB, SB, LAC, SC, TB) = tuple(
                _il(args[i], "BMMV") for i in range(2, 13))
            if TB not in (0, 1):
                raise NoPattern("cuda BMMV TRANSB must be 0/1")
            # Row-major C(M,N) as col-major C(N,M) = B_col x A_col
            # (same trick as BATCH_MATMUL); the T flag rides transa
            # (the B-base operand) when TRANSB=1.
            tA = "CUBLAS_OP_T" if TB else "CUBLAS_OP_N"
            _fp = getattr(self, "use_fp16", False)
            _inputs = ctx.get("inputs", [])

            def _is16(x):
                return _fp and x in _inputs and S[x][0] == "F"
            if _fp and (_is16(a) or _is16(b)):
                # One side half (input), other float: convert the float
                # side over the whole base (views read subsets).
                pre, ea, eb = "", a, b
                from chain.emit_c import _dim_expr as _de
                if _is16(b) and not _is16(a):
                    cnt = _de(S[a][2], ctx["ids_n"])
                    c, ea = _cvt(a, cnt)
                    pre += c
                if _is16(a) and not _is16(b):
                    cnt = _de(S[b][2], ctx["ids_n"])
                    c, eb = _cvt(b, cnt)
                    pre += c
                return (
                    f"/* {o} = BMMV GemmEx-strided F16->F32 */\n  " + pre +
                    f"BE(cublasGemmStridedBatchedEx(h, {tA}, CUBLAS_OP_N, "
                    f"{N_}, {M}, {K}, &kOne, {eb}, CUDA_R_16F, {LAB}, "
                    f"{SB}, {ea}, CUDA_R_16F, {LAA}, {SA}, &kZero, {o}, "
                    f"CUDA_R_32F, {LAC}, {SC}, {BB}, CUBLAS_COMPUTE_32F, "
                    f"CUBLAS_GEMM_DEFAULT));\n{launch}",
                    None)
            return (
                f"/* {o} = BMMV cuBLAS strided-views */\n"
                f"BE(cublasSgemmStridedBatched(h, {tA}, CUBLAS_OP_N, "
                f"{N_}, {M}, {K}, &kOne, {b}, {LAB}, {SB}, {a}, {LAA}, "
                f"{SA}, &kZero, {o}, {LAC}, {SC}, {BB}));\n{launch}",
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
                f"k_transpose_f32<<<(({re_}*{ce}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({x}, {o}, {re_}, {ce});\n{launch}",
                None)
        if mn == "SLICE":
            from chain.emit_c import _int_lit as _il
            x = args[0]
            o = outs[0]
            ax, lo, hi = (_il(args[1], "SLICE ax"), _il(args[2], "SLICE lo"),
                          _il(args[3], "SLICE hi"))
            if S[x][0] != "F" or len(S[x][2]) != 2 or ax not in (0, 1):
                raise NoPattern("cuda SLICE needs 2D ax0/1 F")
            w = S[x][2][1]
            we = str(w) if w is not None else ids_n
            if ax == 1:
                return (
                    f"/* {o} = SLICE({x},1,{lo},{hi}) */\n"
                    f"k_slice_ax1_f32<<<(({o}_N * {hi - lo}) + TPB - 1) / TPB, "
                    f"TPB{st}>>>({x}, {o}, {we}, {lo}, {hi - lo}, {o}_N);\n{launch}",
                    None)
            nrows = hi - lo
            return (
                f"/* {o} = SLICE({x},0,{lo},{hi}) rows */\n"
                f"k_slice_ax0_f32<<<(({nrows} * {we}) + TPB - 1) / TPB, "
                f"TPB{st}>>>({x}, {o}, {we}, {lo}, {nrows});\n{launch}",
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
                f"TPB, TPB{st}>>>({a}, {b}, {o}, {da}, {db}, {o}_N);\n{launch}",
                None)
        if mn == "SELECT":
            m, a, b = args
            o = outs[0]
            if S[m][0] != "I" or S[a][0] != "F" or S[b][0] != "F":
                raise NoPattern("cuda SELECT needs I mask + F (v0.1)")
            co = ctx["count"][o]
            guard = f"if ({m}_N != {co}) return 21;\n  "
            return (
                f"/* {o} = SELECT({m}) */\n  {guard}"
                f"k_select_f32<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
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
                f"TPB{st}>>>({x}, {pos}, {o}, {d}, {o}_N, THETA);\n{launch}",
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
                f"k_tshift_f32<<<({o}_N), TPB{st}>>>"
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
                f"k_fill_f32<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({o}, {co}, {val!r}f);\n{launch}",
                None)
        return super().pattern(mn, outs, args, sig, ctx)

    def epilogue(self, ctx):
        return ""


def compile_cuda(text, sample=None, outputs=None, registry=None,
                 sigs=None, basedir=".", origin=None, use_fp16=False,
                 time_ops=False, live=None, sync_each=True, graph=False):
    """CUDA frontend entry: asm text -> .cu source.

    live: None (one-shot binary, current behavior) or a list of input
    stream names re-read per step in a persistent loop (weights-load-once
    server mode). Loop protocol over stdio: after each step (including
    the first) the binary prints READY; a "q" line on stdin quits,
    anything else runs another step over the live files (same argv
    paths, host rewrites them). Fixed-shape only: live buffers are
    malloc'd once from the first files. Live names that const-folded
    (no argv file) fail loud.
    graph: capture the step into a CUDA graph (single replay launch).
    Requires live (nothing to replay otherwise) and refuses time_ops
    (per-op event syncs are illegal inside capture). Forces
    sync_each=False (device syncs are illegal inside capture; stores
    still sync before host reads).
    """
    if graph and live is None:
        raise NoPattern("cuda: graph needs live=[...] (nothing to replay)")
    if graph and time_ops:
        raise NoPattern("cuda: graph + time_ops refused (event syncs "
                        "cannot be captured)")
    if graph:
        sync_each = False
    from chain.asm import assemble
    from chain.asm_ops import REGISTRY as _R, SIGS as _S
    from chain.emit_c import infer_shapes
    config, inp, bound, _ = assemble(text, registry or _R, sigs or _S,
                                     basedir=basedir, origin=origin)
    if sample is None:
        raise AsmError("compile_cuda needs sample payload (shapes)")
    if not outputs:
        raise AsmError("compile_cuda needs outputs=[...]")
    be = CUDABackend()
    be.use_fp16 = use_fp16
    from chain.emit_c import sanitize_cnames as _san
    inp, bound, sample, outputs, live, _alias = _san(
        inp, bound, sample, outputs, live)
    streams = infer_shapes((config, inp, bound), sample)
    in_names = [n for n, _ in inp]
    if use_fp16:
        # CONVERT pre-pass: F16 inputs consumed outside F16-capable
        # positions (matmul either side, GATHER table, RMSNORM weight)
        # get one F32 upcast; consumers rewritten. Weights feeding only
        # matmuls stay F16 (zero extra memory -- the whole point).
        CAP = {"MATMUL", "BATCH_MATMUL", "GATHER", "RMSNORM"}
        conv = {}
        for _outs, _mn, _fn, _args, _ln, _sg in bound:
            for _pos, _a in enumerate(_args):
                if _a not in in_names or streams[_a][0] != "F":
                    continue
                if _mn in CAP and not (_mn == "RMSNORM" and _pos == 0):
                    continue
                if _a not in conv:
                    conv[_a] = f"{_a}_f32"
                    streams[conv[_a]] = ("F", "double",
                                         streams[_a][2])
        if conv:
            new_bound = []
            for _a, _c in conv.items():
                new_bound.append(([conv[_a]], "CONVERT", None, [_a],
                                  "convert", None))
            for _outs, _mn, _fn, _args, _ln, _sg in bound:
                new_bound.append(
                    (_outs, _mn, _fn,
                     [conv.get(_a, _a) for _a in _args], _ln, _sg))
            bound = new_bound
    from chain.emit_c import find_consts as _fc
    consts = _fc((config, inp, bound), inp, streams, sample)
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
           "config": dict(config), "theta": theta, "count": {},
           "consts": consts, "sync_each": sync_each,
           "stream": ", 0, capStream" if graph else "",
           "mns": [mn for _, mn, _, _, _, _ in bound]}
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
    _fp16 = getattr(be, "use_fp16", False)
    decls = []
    for n in in_names:
        if n in consts:
            continue  # compile-time literal: no storage, no argv file
        k, sub, sh = streams[n]
        if k == "F":
            decls.append(f"static {'__half' if _fp16 else 'float'} *{n};"
                         f" static int64_t {n}_N;")
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
    if graph:
        parts.append("static cudaStream_t capStream;")
        parts.append("static cudaGraph_t capGr;")
        parts.append("static cudaGraphExec_t capGrx;")
    if time_ops:
        parts.append("static cudaEvent_t _te0, _te1;")
        parts.append("static float _msOT;")
    parts.append("static const float kOne = 1.0f;")
    parts.append("static const float kZero = 0.0f;")
    parts.append("int main(int argc, char **argv) {")
    live_ins = [n for n in in_names if n not in consts]
    if live is not None:
        live = list(live)
        for n in live:
            if n not in in_names:
                raise NoPattern(f"cuda: live input '{n}' is not an IN stream")
            if n in consts:
                raise NoPattern(f"cuda: live input '{n}' const-folded "
                                f"(no argv file to re-read)")
            if n in (outputs or []):
                raise NoPattern(f"cuda: live input '{n}' is an output")
    n_argv = len(live_ins) + len(outputs)
    parts.append(f"  if (argc != {n_argv + 1}) return 9;")
    parts.append("  CE(cudaSetDevice(0)); BE(cublasCreate(&h));")
    if time_ops:
        parts.append("  CE(cudaEventCreate(&_te0)); CE(cudaEventCreate(&_te1));")
    parts.append("  BE(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));")
    if graph:
        parts.append("  CE(cudaStreamCreate(&capStream));")
        parts.append("  BE(cublasSetStream(h, capStream));")
    idx = 1
    loads = []
    argidx = {}
    _elb = 2 if getattr(be, "use_fp16", False) else 4
    for n in in_names:
        if n in consts:
            continue
        argidx[n] = idx
        k, _, _ = streams[n]
        if k == "F":
            wrow = streams[n][2][-1]
            if wrow is None:
                raise NoPattern(f"cuda: fully-dynamic F input '{n}' "
                                f"(batch x width both unknown)")
            loads.append(
                f"  {{ FILE *f = fopen(argv[{idx}], \"rb\"); if (!f) return 1;\n"
                f"    fseek(f, 0, SEEK_END); long {n}_B = ftell(f);"
                f" fseek(f, 0, SEEK_SET);\n"
                f"    {n}_N = {n}_B / ({int(wrow)} * {_elb});\n"
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
    compute, stores = [], []
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        compute.append(f"  /* L{_short_site(ln)}: {mn} */")
        body = pat.replace("\n", "\n  ")
        if time_ops:
            # NOTE: braceless by design -- F16 temp decls (_cvt) must stay
            # function-scoped (a wrapping block hid them from later ops).
            body = ("CE(cudaEventRecord(_te0));\n  " + body +
                    f"\n  CE(cudaEventRecord(_te1)); "
                    f"CE(cudaEventSynchronize(_te1)); "
                    f"CE(cudaEventElapsedTime(&_msOT, _te0, _te1)); "
                    f"fprintf(stderr, \"op {mn} %f\\n\", _msOT);")
        compute.append("  " + body)
    for o in outputs:
        ok = streams[o][0]
        el = "8" if ok == "I" else "4"
        co = ctx["count"][o]
        stores.append(f"  {{ FILE *f = fopen(argv[{idx}], \"wb\"); if (!f)"
                     f" return 3;\n"
                     f"    CE(cudaDeviceSynchronize());\n"
                     f"    fwrite({o}, {el}, {co}, f); fclose(f); }}")
        idx += 1
    h16a = ctx.get("h16allocs", {})
    if h16a:
        hal = ["/* F16 temps (hoisted: malloc once ahead, converts run per step) */"]
        for t, cnt in h16a.items():
            hal.append(f"__half *{t} = 0;")
            hal.append(f"  CE(cudaMallocManaged((void**)&{t}, ({cnt}) * 2));")
        parts.append("\n".join(hal))
    if live is None:
        parts.extend(compute)
        parts.extend(stores)
    else:
        # Persistent loop: frozen inputs stay resident (malloc'd once);
        # live inputs re-fread per step (same argv paths, no re-malloc:
        # fixed-shape contract). Host drives via stdin lines.
        reloads = []
        for n in live:
            i = argidx[n]
            k, _, _ = streams[n]
            if k == "F":
                wrow = streams[n][2][-1]
                reloads.append((n, i, int(wrow)))
            else:
                reloads.append((n, i, None))
        if graph:
            parts.extend(compute)  # warmup: migrate pages + JIT kernels
            parts.append("  CE(cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal));")
            parts.extend(compute)  # captured run (executes AND records)
            parts.append("  CE(cudaStreamEndCapture(capStream, &capGr));")
            parts.append("  CE(cudaGraphInstantiate(&capGrx, capGr, NULL, NULL, 0));")
        else:
            parts.extend(compute)
        parts.extend(stores)
        parts.append('  printf("READY\\n"); fflush(stdout);')
        parts.append("  { char line[16];")
        parts.append("    while (fgets(line, sizeof(line), stdin)) {")
        parts.append("      if (line[0] == 'q') break;")
        for (n, i, wrow) in reloads:
            if wrow is not None:
                parts.append(
                    "    { FILE *f = fopen(argv[%d], \"rb\");" % i +
                    " if (!f) return 1;")
                parts.append(
                    "      fseek(f, 0, SEEK_END); long %s_B = ftell(f);" % n +
                    " fseek(f, 0, SEEK_SET);")
                parts.append(
                    "      %s_N = %s_B / (%d * %d);" % (n, n, wrow, _elb))
                parts.append(
                    "      if (fread(%s, 1, %s_B, f) != (size_t)%s_B)"
                    " return 2; fclose(f); }" % (n, n, n))
            else:
                parts.append(
                    "    { FILE *f = fopen(argv[%d], \"rb\");" % i +
                    " if (!f) return 1;")
                parts.append(
                    "      fseek(f, 0, SEEK_END); %s_N = ftell(f)/8;" % n +
                    " fseek(f, 0, SEEK_SET);")
                parts.append(
                    "      if (fread(%s, 8, %s_N, f) != (size_t)%s_N)"
                    " return 2; fclose(f); }" % (n, n, n))
        if graph:
            parts.append("      CE(cudaGraphLaunch(capGrx, capStream));")
        else:
            for c in compute:
                parts.append("    " + c.replace("\n", "\n    "))
        for s in stores:
            parts.append("    " + s.replace("\n", "\n    "))
        parts.append('      printf("READY\\n"); fflush(stdout);')
        parts.append("    } }")
    parts.append("  BE(cublasDestroy(h));")
    parts.append("  return 0;\n}")
    parts.append(be.epilogue(ctx))
    return {"source": "\n".join(parts), "backend": be, "streams": streams,
            "dims": {}, "config": dict(config), "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1, "consts": consts,
            "live": list(live) if live is not None else None,
            "graph": bool(graph), "cname": _alias}


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
