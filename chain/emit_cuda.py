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
import hashlib
import os
import re
import shutil
import subprocess

import numpy as np

from chain.asm import AsmError
from chain.emit_c import Backend, NoPattern, _dim_expr, _short_site

# Bump when the cache protocol itself changes (key inputs are the
# source + flags + arch + libs; emitter output changes are covered
# by the source hash).
CU_CACHE_VERSION = "v2"

# Multi-TU builds: split listings this big (op count) into ~200-op
# translation units compiled in parallel (JOTNAR_JOBS, default
# cpu_count). Single TU below the bar; JOTNAR_SINGLE_TU=1 forces it.
CU_SPLIT_OPS = 400
CU_CHUNK_OPS = 200


def cu_cache_dir():
    """Central build cache root (shared across workdirs)."""
    root = os.environ.get("JOTNAR_CACHEDIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "jotnar", "cu")
    os.makedirs(root, exist_ok=True)
    return root


def cu_kernel_prototypes():
    """Forward declarations for KERNELS (chunk TUs call, kernels TU defines)."""
    out = []
    for m in re.finditer(r"__global__\s+void\s+(\w+)\s*\((.*?)\)\s*\{",
                         KERNELS, re.S):
        args = " ".join(m.group(2).split())
        out.append(f"__global__ void {m.group(1)}({args});")
    return "\n".join(out)


def cu_jobs(n_units):
    """Parallel compile jobs (default: all cores)."""
    try:
        j = int(os.environ.get("JOTNAR_JOBS", "0"))
    except ValueError:
        j = 0
    return max(1, min(n_units, j or (os.cpu_count() or 8)))


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
__global__ void k_topk_f32(const float *x, float *v, int64_t *ix,
                         int64_t S, int64_t K, int64_t N) {
  /* One BLOCK per row (launch <<<N,1>>>): k-slot insertion over S.
     Ties-first: strict > displaces, so equal values keep the earlier
     index (matches lattice-ref stable descending argsort). K <= 64. */
  int64_t r = (int64_t)blockIdx.x;
  if (r >= N) return;
  float bv[64]; int64_t bi[64];
  for (int64_t j = 0; j < K; j++) { bv[j] = -1e30f; bi[j] = 0; }
  for (int64_t j = 0; j < S; j++) {
    float a = x[r * S + j];
    if (a > bv[K - 1]) {
      int64_t p = K - 1;
      while (p > 0 && a > bv[p - 1]) {
        bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--;
      }
      bv[p] = a; bi[p] = j;
    }
  }
  for (int64_t j = 0; j < K; j++) {
    v[r * K + j] = bv[j]; ix[r * K + j] = bi[j];
  }
}
__global__ void k_scatter_row_f32(const float *b, const float *row,
                                const int64_t *idx, float *o, int64_t C,
                                int64_t S, int64_t N) {
  /* out = base copy with row idx[0] := row. Grid-stride over S*C.
     OOB idx poisons o[0] with NaN (loud downstream: parity gates
     catch it; idx comes from pos, always valid by construction). */
  int64_t dst = idx[0];
  if (dst < 0 || dst >= S) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
      o[0] = __int_as_float(0x7fc00000);
    return;
  }
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int64_t r = i / C, c = i % C;
  o[i] = (r == dst) ? row[c] : b[i];
}
__global__ void k_scatter_row_f16(const float *b, const float *row,
                                const int64_t *idx, __half *o, int64_t C,
                                int64_t S, int64_t N) {
  /* Resident-cache form: base/output are fp16 files-upcast domain,
     update row is fp32 compute domain (rounded once on insert --
     standard KV-cache precision, same rounding as the host relay).
     OOB idx poisons o[0] with NaN (loud downstream). */
  int64_t dst = idx[0];
  if (dst < 0 || dst >= S) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
      o[0] = __float2half(nanf(""));
    return;
  }
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  int64_t r = i / C, c = i % C;
  /* base holds fp16-origin values (upcast is exact), so the downcast
     below round-trips exactly; only the inserted row rounds (once). */
  o[i] = (r == dst) ? __float2half(row[c]) : __float2half(b[i]);
}
__global__ void k_attn_gqa_f32(const float *q, const float *K, const float *V,
                             const int64_t *m, float *o, int64_t S,
                             int64_t DH, int64_t NKV, int64_t PER,
                             int64_t NH, float neg) {
  /* Fused decode GQA attention: one block per head, TPB threads
     cooperating. Scores + mask + shift + softmax + context in one pass
     over K/V (one read each, vs three passes + mask/select/shift
     kernels). All reductions use fixed tree order => bit-deterministic
     across runs and binaries (unlike strided cublas). S <= 512
     (emitter refuses more). Single query row (decode); mask row =
     head % PER. */
  int64_t h = (int64_t)blockIdx.x;
  if (h >= NH) return;
  int64_t g = h / PER, hr = h % PER;
  int tx = (int)threadIdx.x;
  __shared__ float sc[512];
  __shared__ float red[TPB];
  /* Phase 1: scores (independent per position, no sync needed). */
  for (int64_t j = tx; j < S; j += TPB) {
    double acc = 0.0;
    for (int64_t d = 0; d < DH; d++)
      acc += (double)q[h * DH + d] * (double)K[j * NKV * DH + g * DH + d];
    sc[j] = m[hr * S + j] ? (float)acc : neg;
  }
  __syncthreads();
  /* Phase 2: rowmax (parallel max + tree). */
  float mx = -1e30f;
  for (int64_t j = tx; j < S; j += TPB) mx = fmaxf(mx, sc[j]);
  red[tx] = mx;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if (tx < s) red[tx] = fmaxf(red[tx], red[tx + s]);
    __syncthreads();
  }
  mx = red[0];
  /* Phase 3: exp + sum (partial sums + tree; fixed order). */
  double se = 0.0;
  for (int64_t j = tx; j < S; j += TPB) {
    float e = expf(sc[j] - mx);
    sc[j] = e;
    se += (double)e;
  }
  red[tx] = (float)se;
  __syncthreads();
  for (int s = TPB / 2; s > 0; s >>= 1) {
    if (tx < s) red[tx] += red[tx + s];
    __syncthreads();
  }
  double inv = 1.0 / (double)red[0];
  /* Phase 4: context (one DH lane per thread, serial over S). */
  for (int64_t d = tx; d < DH; d += TPB) {
    double acc = 0.0;
    for (int64_t j = 0; j < S; j++)
      acc += (double)sc[j] * (double)V[j * NKV * DH + g * DH + d];
    o[h * DH + d] = (float)(acc * inv);
  }
}
__global__ void k_rope_heads_f32(const float *x, const int64_t *pos, float *o,
                               int64_t DH, int64_t NH, int64_t N,
                               const float *theta) {
  /* Strided per-head rotary over flat (R,NH*DH): head h, pair p reads
     (x[h*DH+2p], x[h*DH+2p+1]), rotates by pos[r]*theta[p]. Same math
     as NH per-head ROTARY ops + CONCAT rejoin, one pass, no copies.
     Elementwise => deterministic. THETA baked (period DH). */
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t H = DH / 2;
  if (i >= N * NH * H) return;
  int64_t r = i / (NH * H), rem = i % (NH * H);
  int64_t h = rem / H, p = rem % H;
  float ang = (float)pos[r] * theta[p];
  float c = cosf(ang), s = sinf(ang);
  float x0 = x[r * NH * DH + h * DH + 2 * p];
  float x1 = x[r * NH * DH + h * DH + 2 * p + 1];
  o[r * NH * DH + h * DH + 2 * p] = x0 * c - x1 * s;
  o[r * NH * DH + h * DH + 2 * p + 1] = x0 * s + x1 * c;
}
__global__ void k_swiglu_f32(const float *ug, float *o, int64_t I,
                           int64_t N) {
  /* SwiGLU gate: mid[i] = silu(ug[I+i]) * ug[i]. One pass over ug
     (vs split/split/silu/mul: 4 kernels + 3 extra passes).
     Elementwise => deterministic. */
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * I) return;
  int64_t r = i / I, c = i % I;
  float g = ug[r * 2 * I + I + c];
  float s = g / (1.0f + expf(-g));
  o[i] = s * ug[r * 2 * I + c];
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
    # -O1 default: measured 95s vs 124s build on qwen7b-dec (5771 ops)
    # with identical step ms (36ms/tok); host code is launch
    # boilerplate, runtime lives in cuBLAS/hand kernels. Override
    # with NVCC_FLAGS="-O2 -std=c++17" for record builds.
    DEFAULT_NVCC = ("-O1", "-std=c++17")
    DEFAULT_ARCH = "sm_86"

    def value_repr(self, layout):
        kind = (layout or "").split(":")[0]
        return {"I": "int64_t", "F": "float"}.get(kind, "void/*T*/")

    def prologue_head(self):
        return ("\n".join([
            "#include <stdint.h>", "#include <stdio.h>",
            "#include <stdlib.h>", "#include <string.h>",
            "#include <cuda_runtime.h>", "#include <cublas_v2.h>",
            "#include <cuda_fp16.h>", "",
            "#define CE(x) do { cudaError_t _e = (x); if (_e) { fprintf(stderr, \"cuda %d\\n\", (int)_e); return 10; } } while (0)",
            "#define BE(x) do { cublasStatus_t _e = (x); if (_e) { fprintf(stderr, \"cublas %d\\n\", (int)_e); return 11; } } while (0)",
        ]) + "\n")

    def prologue(self, ctx):
        th = ctx.get("theta", [])
        thal = ("static const float THETA[] = {%s};" %
                ", ".join(repr(float(t)) + "f" for t in th)) if th else ""
        return ("\n".join([
            self.prologue_head(),
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
        if mn == "SWIGLU":
            from chain.emit_c import _int_lit as _il
            x = args[0]
            o = outs[0]
            inter = _il(args[1], "SWIGLU inter")
            if S[x][0] != "F" or len(S[x][2]) != 2:
                raise NoPattern("cuda SWIGLU needs 2D F (v0.1)")
            if S[x][2][1] is not None and S[x][2][1] != 2 * inter:
                raise NoPattern("cuda SWIGLU needs static width 2*I "
                                f"(got {S[x][2]})")
            co = ctx["count"][o]
            return (
                f"/* {o} = SWIGLU({x},{inter}) */\n"
                f"k_swiglu_f32<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({x}, {o}, {inter}, {co});\n{launch}",
                None)
        if mn == "ROPE":
            from chain.emit_c import _int_lit as _il
            x, pos = args[0], args[1]
            o = outs[0]
            nh, dh = _il(args[2], "ROPE nh"), _il(args[3], "ROPE dh")
            if S[x][0] != "F" or S[pos][0] != "I":
                raise NoPattern("cuda ROPE needs F + I pos (v0.1)")
            if S[x][2][1] != nh * dh or dh % 2:
                raise NoPattern("cuda ROPE needs static width nh*dh, "
                                f"even dh (got {S[x][2]})")
            if len(ctx.get("theta", [])) != dh // 2:
                raise NoPattern("cuda ROPE needs baked THETA period dh "
                                "(mixed head dims in one program refused)")
            return (
                f"/* {o} = ROPE({x},{nh}x{dh}) */\n"
                f"k_rope_heads_f32<<<(({o}_N * ({nh}) * ({dh} / 2)) + TPB - 1) / TPB, "
                f"TPB{st}>>>({x}, {pos}, {o}, {dh}, {nh}, {o}_N, THETA);\n{launch}",
                None)
        if mn == "ATTN":
            from chain.emit_c import _int_lit as _il
            qr, kf, vf, m = args[0], args[1], args[2], args[3]
            o = outs[0]
            if S[qr][0] != "F" or S[kf][0] != "F" or S[vf][0] != "F" \
                    or S[m][0] != "I":
                raise NoPattern("cuda ATTN needs F qr/Kf/Vf + I mask "
                                "(v0.1)")
            (nh, nkv, dh, s) = tuple(_il(args[i], "ATTN") for i in range(4, 8))
            if s > 512:
                raise NoPattern(f"cuda ATTN S={s} > 512 (kernel slots)")
            per = nh // nkv
            neg = float(ctx["config"].get("beta", -1e9))
            return (
                f"/* {o} = ATTN({qr},{kf},{vf}) fused decode-GQA */\n"
                f"k_attn_gqa_f32<<<({nh}), TPB{st}>>>"
                f"({qr}, {kf}, {vf}, {m}, {o}, {s}, {dh}, {nkv}, "
                f"{per}, {nh}, {neg!r}f);\n{launch}",
                None)
        if mn == "SCATTER":
            from chain.emit_c import _dim_expr as _de
            base, row, idx = args
            o = outs[0]
            if S[base][0] != "F" or S[row][0] != "F" or S[idx][0] != "I":
                raise NoPattern("cuda SCATTER needs F base + F row + I "
                                "idx (v0.1)")
            bsh, rsh = S[base][2], S[row][2]
            if len(bsh) != 2 or tuple(rsh) != (1, bsh[1]):
                raise NoPattern("cuda SCATTER needs base (S,C) + row "
                                f"(1,C), got {bsh} {rsh}")
            if bsh[0] is None or bsh[1] is None:
                raise NoPattern("cuda SCATTER needs static shapes (v0.1)")
            co = ctx["count"][o]
            if getattr(self, "use_fp16", False) and o in ctx.get("inputs", []):
                return (
                    f"/* {o} = SCATTER({base},{row},{idx}) resident-f16 */\n"
                    f"k_scatter_row_f16<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                    f"({base}, {row}, {idx}, {o}, {bsh[1]}, {bsh[0]}, "
                    f"{co});\n{launch}",
                    None)
            return (
                f"/* {o} = SCATTER({base},{row},{idx}) */\n"
                f"k_scatter_row_f32<<<(({co}) + TPB - 1) / TPB, TPB{st}>>>"
                f"({base}, {row}, {idx}, {o}, {bsh[1]}, {bsh[0]}, "
                f"{co});\n{launch}",
                None)
        if mn == "TOPK":
            from chain.emit_c import _int_lit as _il
            x = args[0]
            vo, io = outs[0], outs[1]
            k = _il(args[1], "TOPK k")
            if S[x][0] != "F" or len(S[x][2]) != 2:
                raise NoPattern("cuda TOPK needs 2D F scores (v0.1)")
            rsh, ssh = S[x][2]
            if ssh is None or not (1 <= k <= ssh):
                raise NoPattern(f"cuda TOPK needs static S>=k (k={k})")
            if k > 64:
                raise NoPattern("cuda TOPK k<=64 (kernel slots)")
            r = str(rsh) if rsh is not None else ids_n
            return (
                f"/* {vo},{io} = TOPK({x},{k}) */\n"
                f"k_topk_f32<<<({r}), 1{st}>>>"
                f"({x}, {vo}, {io}, {ssh}, {k}, {r});\n{launch}",
                None)
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
        # SHADOW RULE (resident caches): the front snapshot goes stale
        # after an in-program write to the input (SCATTER in place);
        # every op that assigns a mirrored input is followed by a
        # CONVERT refresh, so later readers see post-write state.
        # Reads before the first write use the front snapshot (same
        # values). Non-shadowed programs emit zero refresh ops.
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
            refreshed = []
            for _entry in bound:
                refreshed.append(_entry)
                _outs, _mn, _fn, _args, _ln, _sg = _entry
                for _o in _outs:
                    if _o in conv:
                        refreshed.append(([conv[_o]], "CONVERT", None,
                                          [_o], "convert", None))
            bound = refreshed
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
        if _mn == "ROPE":
            # Strided rotary carries dh as a literal (same table form).
            from chain.emit_c import _int_lit as _il2
            _d = _il2(_aa[3], "ROPE dh")
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
    interI = set()
    _, _, prog = (config, inp, bound)
    for outs, _m2, _f2, _a2, _l2, _s2 in prog:
        for x in outs:
            if x not in in_names and x not in outputs:
                if streams[x][0] == "F":
                    inter.add(x)
                elif streams[x][0] == "I":
                    # Intermediate index streams (TOPK idx feeding
                    # GATHER): static shape required (dynamic rows
                    # come from input ids files, never intermediates).
                    if (len(streams[x][2]) != 2 or
                            any(d is None for d in streams[x][2])):
                        raise NoPattern(
                            f"cuda: intermediate I '{x}' needs static "
                            f"2D shape, got {streams[x][2]}")
                    interI.add(x)
    # Multi-TU split decision (needs op count only).
    chunked = (len(prog) >= CU_SPLIT_OPS
               and os.environ.get("JOTNAR_SINGLE_TU") != "1")
    # shared_vars: file-scope state (buffers/handles/temps) shared by
    # chunk TUs. Legacy single-TU keeps its `static` decls untouched;
    # multi-TU emits these as extern (header) + definitions (main TU).
    shared_vars = []
    for n in in_names:
        if n in consts:
            continue
        k, sub, sh = streams[n]
        if k == "F":
            shared_vars.append(f"{'__half' if _fp16 else 'float'} *{n};")
            shared_vars.append(f"int64_t {n}_N;")
        elif k != "T":
            shared_vars.append(f"int64_t *{n};")
            shared_vars.append(f"int64_t {n}_N;")
    for o in outputs:
        ok, _, _ = streams[o]
        if ok == "F":
            shared_vars.append(f"float *{o};")
        else:
            shared_vars.append(f"int64_t *{o};")
            shared_vars.append(f"int64_t {o}_N;")
    for x in sorted(inter):
        shared_vars.append(f"float *{x};")
        shared_vars.append(f"int64_t {x}_N;")
    for x in sorted(interI):
        shared_vars.append(f"int64_t *{x};")
        shared_vars.append(f"int64_t {x}_N;")
    for o in outputs:
        if streams[o][0] == "F":
            shared_vars.append(f"int64_t {o}_N;")
    shared_vars.append("cublasHandle_t h;")
    shared_vars.append("void *_jotws;")
    if graph:
        shared_vars += ["cudaStream_t capStream;", "cudaGraph_t capGr;",
                        "cudaGraphExec_t capGrx;"]
    if time_ops:
        shared_vars += ["cudaEvent_t _te0, _te1;", "float _msOT;"]
    if inter or interI:
        parts.append("\n".join(decls))
        parts.append("/* intermediates (managed) */")
        parts.append("\n".join(
            [f"float *{x};" for x in sorted(inter)] +
            [f"int64_t *{x};" for x in sorted(interI)]))
    else:
        parts.append("\n".join(decls))
    parts.append("static cublasHandle_t h;")
    parts.append("static void *_jotws = 0;")
    if graph:
        parts.append("static cudaStream_t capStream;")
        parts.append("static cudaGraph_t capGr;")
        parts.append("static cudaGraphExec_t capGrx;")
    if time_ops:
        parts.append("static cudaEvent_t _te0, _te1;")
        parts.append("static float _msOT;")
    parts.append("static const float kOne = 1.0f;")
    parts.append("static const float kZero = 0.0f;")
    main_head_at = len(parts)
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
    # Deterministic cuBLAS: a managed workspace + cublasSetWorkspace
    # selects deterministic algorithms (split-K/reduction order fixed).
    # Without it, summation order varies with allocation state: same
    # binary differs run-to-run (~0.5 values; top-k amplifies at razor
    # boundaries into decision flips). Geometry, not statistics: ON by
    # default; JOTNAR_NONDET=1 restores legacy behavior (debug/perf).
    # Size via JOTNAR_WS_MB (default 32). Must precede any GEMM call;
    # host setup only (never inside graph capture).
    parts.append("  { const char *_nd = getenv(\"JOTNAR_NONDET\");")
    parts.append("    if (!_nd || !_nd[0] || _nd[0] != '1') {")
    parts.append("      long _mb = 32;")
    parts.append("      { const char *_e = getenv(\"JOTNAR_WS_MB\");")
    parts.append("        if (_e && _e[0]) { long _v = atol(_e);")
    parts.append("          if (_v >= 1 && _v <= 512) _mb = _v; } }")
    parts.append("      CE(cudaMallocManaged((void**)&_jotws, "
                   "(size_t)_mb * 1048576));")
    parts.append("      BE(cublasSetWorkspace(h, _jotws, "
                   "(size_t)_mb * 1048576)); } }")
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
        # Chunked builds share it file-wide (extern); legacy keeps the
        # function-local decl (identical semantics: assigned pre-use).
        from chain.emit_c import _rows_expr as _rr
        if chunked:
            allocs.append(f"  {x}_N = {_rr(xsh, ids_n)};")
        else:
            allocs.append(f"  int64_t {x}_N = {_rr(xsh, ids_n)};")
    for o in outputs:
        if streams[o][0] == "I":
            _osh = streams[o][2]
            _on = ("(%s)" % ids_n) if any(d is None for d in _osh) \
                else str(int(np.prod(list(_osh), dtype=np.int64)))
            allocs.append(f"  {o}_N = {_on}; "
                          f"CE(cudaMallocManaged((void**)&{o}, {o}_N * 8));")
    for x in sorted(interI):
        # Intermediate index streams (static 2D by construction above).
        _xsh = streams[x][2]
        _xn = str(int(np.prod(list(_xsh), dtype=np.int64)))
        _xc = _dim_expr(_xsh, ids_n)
        allocs.append(f"  CE(cudaMallocManaged((void**)&{x}, ({_xc}) * 8));")
        if chunked:
            allocs.append(f"  {x}_N = {_xn};")
        else:
            allocs.append(f"  int64_t {x}_N = {_xn};")
    parts.append("\n".join(allocs))
    compute, stores = [], []
    for outs, mn, _fn, args, ln, sig in prog:
        pat, gathered = be.pattern(mn, outs, args, sig, ctx)
        if gathered:
            ctx["gathered"][gathered[0]] = gathered[1]
        # One entry per op (comment + body): legacy join is byte-identical
        # to the old two-entry form, and chunks partition cleanly by op.
        body = pat.replace("\n", "\n  ")
        if time_ops:
            # NOTE: braceless by design -- F16 temp decls (_cvt) must stay
            # function-scoped (a wrapping block hid them from later ops).
            # Chunk functions preserve this (plain function scope, no
            # extra blocks).
            body = ("CE(cudaEventRecord(_te0));\n  " + body +
                    f"\n  CE(cudaEventRecord(_te1)); "
                    f"CE(cudaEventSynchronize(_te1)); "
                    f"CE(cudaEventElapsedTime(&_msOT, _te0, _te1)); "
                    f"fprintf(stderr, \"op {mn} %f\\n\", _msOT);")
        compute.append(f"  /* L{_short_site(ln)}: {mn} */\n  " + body)
    # Chunk partition (multi-TU only): contiguous op groups.
    nchunks = 0
    chunks = []
    if chunked:
        per = max(CU_CHUNK_OPS, (len(compute) + 31) // 32)
        chunks = [compute[i:i + per] for i in range(0, len(compute), per)]
        nchunks = len(chunks)

    def _calls(ind):
        return [f"{ind}{{ int _jrc = jotnar_step_{i}(); "
                f"if (_jrc) return _jrc; }}"
                for i in range(nchunks)]
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
    h16_decls, h16_mallocs = [], []
    h16_comment = ("/* F16 temps (hoisted: malloc once ahead, converts run "
                   "per step) */")
    for t, cnt in h16a.items():
        # Shared form has no initializer (extern + initializer IS a
        # definition); legacy local keeps `= 0` (byte-identical).
        h16_decls.append(f"__half *{t};")
        h16_mallocs.append(
            f"  CE(cudaMallocManaged((void**)&{t}, ({cnt}) * 2));")
    if chunked:
        shared_vars += h16_decls

    # Shadowed state: IN streams reassigned as OUTs (in-place update by
    # naming, e.g. resident KV caches). Warmup/capture runs mutate them,
    # so they are reloaded from the startup files before READY (serve
    # starts from loaded files: zeros normally, or --prefill-dir copies;
    # unconditional zeroing would wipe preloaded caches). One-shot
    # programs skip this (no reuse).
    from chain.emit_c import _dim_expr as _de2
    _shadowed = []
    _seen_out = set()
    for _outs, _mn2, _f2, _a2, _l2, _s2 in prog:
        for _x in _outs:
            _seen_out.add(_x)
    for _n in in_names:
        if _n in _seen_out and _n not in consts:
            _k, _, _sh = streams[_n]
            if _k not in ("F", "I"):
                raise NoPattern(f"cuda: shadowed {_n} kind {_k} (v0.1)")
            _shadowed.append(_n)

    def _tail(use_calls):
        # Assemble main() body tail (h16 + steps + stores + loop).
        # use_calls=False: legacy inline ops (byte-identical to the old
        # single-TU source). use_calls=True: jotnar_step_N() calls for
        # the multi-TU main.cu. Pure string building: safe to run twice.
        t = []
        if h16a:
            if use_calls:
                t.append("\n".join([h16_comment] + h16_mallocs))
            else:
                hal = [h16_comment]
                for d, m in zip(h16_decls, h16_mallocs):
                    hal += [d[:-1] + " = 0;", m]
                t.append("\n".join(hal))
        step_main = _calls("  ") if use_calls else compute
        step_loop = (_calls("    ") if use_calls else
                     ["    " + c.replace("\n", "\n    ") for c in compute])
        if live is None:
            t.extend(step_main)
            t.extend(stores)
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
                t.extend(step_main)  # warmup: migrate pages + JIT kernels
                t.append("  CE(cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal));")
                t.extend(step_main)  # captured run (executes AND records)
                t.append("  CE(cudaStreamEndCapture(capStream, &capGr));")
                t.append("  CE(cudaGraphInstantiate(&capGrx, capGr, NULL, NULL, 0));")
            else:
                t.extend(step_main)
            t.extend(stores)
            if _shadowed:
                t.append("  /* state reload: warmup/capture mutated shadowed "
                         "streams; restore startup files before READY */")
                for _nm in _shadowed:
                    _k, _, _sh = streams[_nm]
                    _i = argidx[_nm]
                    if _k == "F":
                        _wrow = _sh[-1]
                        if _wrow is None:
                            raise NoPattern(
                                f"cuda: shadowed {_nm} needs baked width")
                        _elb_s = 2 if getattr(
                            be, "use_fp16", False) else 4
                        t.append(
                            "  { FILE *f = fopen(argv[%d], \"rb\");" % _i +
                            " if (!f) return 1;")
                        t.append(
                            "    fseek(f, 0, SEEK_END); long %s_B = ftell(f);"
                            % _nm + " fseek(f, 0, SEEK_SET);")
                        t.append(
                            "    %s_N = %s_B / (%d * %d);" %
                            (_nm, _nm, _wrow, _elb_s))
                        t.append(
                            "    if (fread(%s, 1, %s_B, f) != (size_t)%s_B)"
                            " return 2; fclose(f); }" % (_nm, _nm, _nm))
                    else:
                        t.append(
                            "  { FILE *f = fopen(argv[%d], \"rb\");" % _i +
                            " if (!f) return 1;")
                        t.append(
                            "    fseek(f, 0, SEEK_END); %s_N = ftell(f)/8;"
                            % _nm + " fseek(f, 0, SEEK_SET);")
                        t.append(
                            "    if (fread(%s, 8, %s_N, f) != (size_t)%s_N)"
                            " return 2; fclose(f); }" % (_nm, _nm, _nm))
            t.append('  printf("READY\\n"); fflush(stdout);')
            t.append("  { char line[16];")
            t.append("    while (fgets(line, sizeof(line), stdin)) {")
            t.append("      if (line[0] == 'q') break;")
            for (n, i, wrow) in reloads:
                if wrow is not None:
                    t.append(
                        "    { FILE *f = fopen(argv[%d], \"rb\");" % i +
                        " if (!f) return 1;")
                    t.append(
                        "      fseek(f, 0, SEEK_END); long %s_B = ftell(f);" % n +
                        " fseek(f, 0, SEEK_SET);")
                    t.append(
                        "      %s_N = %s_B / (%d * %d);" % (n, n, wrow, _elb))
                    t.append(
                        "      if (fread(%s, 1, %s_B, f) != (size_t)%s_B)"
                        " return 2; fclose(f); }" % (n, n, n))
                else:
                    t.append(
                        "    { FILE *f = fopen(argv[%d], \"rb\");" % i +
                        " if (!f) return 1;")
                    t.append(
                        "      fseek(f, 0, SEEK_END); %s_N = ftell(f)/8;" % n +
                        " fseek(f, 0, SEEK_SET);")
                    t.append(
                        "      if (fread(%s, 8, %s_N, f) != (size_t)%s_N)"
                        " return 2; fclose(f); }" % (n, n, n))
            if graph:
                t.append("      CE(cudaGraphLaunch(capGrx, capStream));")
            else:
                t.extend(step_loop)
            for s in stores:
                t.append("    " + s.replace("\n", "\n    "))
            t.append('      printf("READY\\n"); fflush(stdout);')
            t.append("    } }")
        t.append("  BE(cublasDestroy(h));")
        t.append("  return 0;\n}")
        return t
    main_head = list(parts[main_head_at:])
    parts.extend(_tail(False))
    parts.append(be.epilogue(ctx))
    source = "\n".join(parts)
    units = None
    if chunked:
        main_body = "\n".join(_tail(True))
        units = _split_units(be, ctx, shared_vars, chunks, nchunks,
                             main_head, main_body)
    return {"source": source, "backend": be, "streams": streams,
            "dims": {}, "config": dict(config), "inputs": in_names,
            "outputs": outputs, "n_argv": idx - 1, "consts": consts,
            "live": list(live) if live is not None else None,
            "graph": bool(graph), "cname": _alias, "units": units}


def _split_units(be, ctx, shared_vars, chunks, nchunks, main_head,
                 main_body):
    """Multi-TU sources: common header + kernels TU + chunk TUs + main TU.

    chunks are contiguous op groups; every emitted line references
    only file-scope state (buffers/handles in shared_vars), so any
    op boundary is a safe TU boundary. Returns [{"name", "content"}].
    """
    th = ctx.get("theta", [])
    thal = ("static const float THETA[] = {%s};" %
            ", ".join(repr(float(t)) + "f" for t in th)) if th else ""
    head = be.prologue_head()
    header = "\n".join([
        "#pragma once",
        head,
        "#define TPB 256",
        cu_kernel_prototypes(),
        thal,
        "/* shared state (defined once in jot_main.cu) */",
        "\n".join(f"extern {d}" for d in shared_vars),
        "static const float kOne = 1.0f;",
        "static const float kZero = 0.0f;",
        "",
    ])
    defs = "\n".join(shared_vars) + "\n"
    step_decls = "\n".join(f"int jotnar_step_{i}(void);"
                            for i in range(nchunks)) + "\n"
    units = [{"name": "jot_common.h", "content": header},
             {"name": "jot_kernels.cu",
              "content": head + "\n" + KERNELS + "\n"}]
    for i, blk in enumerate(chunks):
        body = "\n".join(blk)
        units.append({
            "name": f"jot_chunk_{i:02d}.cu",
            # int return: CE/BE macros return error codes; main
            # propagates them (fail-loud across the TU boundary).
            "content": (f'#include "jot_common.h"\n'
                        f"int jotnar_step_{i}(void) {{\n{body}\nreturn 0;\n}}\n"),
        })
    units.append({"name": "jot_main.cu",
                  "content": (f'#include "jot_common.h"\n{defs}\n'
                              f"{step_decls}\n" + "\n".join(main_head)
                              + f"\n{main_body}\n")})
    return units


def build_cu(source, workdir, name="prog", nvcc="nvcc", nvflags=None,
             arch=None, libs=None, units=None):
    """Compile emitted .cu -> exe. Arch flag IS the target selection.

    Flags default to CUDABackend.DEFAULT_NVCC (-O1); override with
    explicit nvflags or NVCC_FLAGS env ("-O2 -std=c++17" for records).

    units (from compile_cuda["units"]): multi-TU build -- translation
    units compile in parallel (JOTNAR_JOBS, default all cores) and
    link once. Best for 500+ op listings (single cc1plus on the whole
    program is the bottleneck); small programs stay single-TU.
    """
    arch = arch or os.environ.get("CUDA_ARCH", CUDABackend.DEFAULT_ARCH)
    envf = os.environ.get("NVCC_FLAGS", "").split() or None
    nvflags = list(nvflags or envf or CUDABackend.DEFAULT_NVCC)
    libs = list(libs or ["-lcublas"])
    os.makedirs(workdir, exist_ok=True)
    src = os.path.join(workdir, name + ".cu")
    exe = os.path.join(workdir, name)
    with open(src, "w") as f:
        f.write(source)
    # Content-keyed cache: identical source+flags+arch+libs skips nvcc
    # entirely (today's suites rebuild the same 5771-op program per
    # workdir: ~95s a throw). Disabled with JOTNAR_NOCACHE=1.
    h = hashlib.sha256()
    h.update(CU_CACHE_VERSION.encode())
    h.update(b"\x00".join([arch.encode()] + [f.encode() for f in nvflags]
                           + [f.encode() for f in libs]))
    if units:
        for u in units:
            h.update(b"\x00" + u["name"].encode() + b"\x00"
                     + u["content"].encode())
    else:
        h.update(b"\x00" + source.encode())
    key = h.hexdigest()[:32]
    if os.environ.get("JOTNAR_NOCACHE") != "1":
        try:
            cdir = cu_cache_dir()
            centry = os.path.join(cdir, key)
            cexe = os.path.join(centry, "exe")
            if os.path.isfile(cexe):
                shutil.copyfile(cexe, exe)
                os.chmod(exe, 0o755)
                return exe
        except OSError:
            pass  # cache best-effort; fall through to a real build
    else:
        centry, cexe = None, None
    archf = f"-arch={arch}" if arch else None
    if not units:
        cmd = [nvcc] + nvflags + ([archf] if archf else []) + [src, "-o", exe] \
            + libs
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise AsmError(f"nvcc build failed:\n{r.stderr}")
        _cache_store(exe, key, " ".join(cmd))
        return exe
    exe = _build_multi(nvcc, nvflags, archf, libs, workdir, name, units)
    _cache_store(exe, key, f"multi-TU ({len(units)} units)")
    return exe


def _cache_store(exe, key, info):
    if os.environ.get("JOTNAR_NOCACHE") == "1":
        return
    try:
        cdir = cu_cache_dir()
        centry = os.path.join(cdir, key)
        cexe = os.path.join(centry, "exe")
        if not os.path.isfile(cexe):
            tmp = os.path.join(cdir, f".tmp-{os.getpid()}-{key}")
            os.makedirs(centry, exist_ok=True)
            shutil.copyfile(exe, tmp)
            with open(os.path.join(centry, "info"), "w") as f:
                f.write(info + "\n")
            os.replace(tmp, cexe)
    except OSError:
        pass


def _build_multi(nvcc, nvflags, archf, libs, workdir, name, units):
    """Parallel-TU build: compile units concurrently, link once."""
    import concurrent.futures
    tudir = os.path.join(workdir, name + "_tu")
    os.makedirs(tudir, exist_ok=True)
    cus = []
    for u in units:
        if not u["name"].endswith(".cu"):
            continue
        p = os.path.join(tudir, u["name"])
        with open(p, "w") as f:
            f.write(u["content"])
        if u["name"] != "jot_common.h":
            cus.append(p)
    # Header goes alongside the .cu files (quoted include).
    for u in units:
        if u["name"].endswith(".h"):
            with open(os.path.join(tudir, u["name"]), "w") as f:
                f.write(u["content"])
    jobs = cu_jobs(len(cus))
    print(f"nvcc multi-TU: {len(cus)} units x {jobs} jobs", flush=True)

    def _cc(cu):
        o = cu[:-3] + ".o"
        cmd = [nvcc] + nvflags + ([archf] if archf else []) \
            + ["-c", cu, "-o", o]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (cu, o, r)

    objs, fails = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        for cu, o, r in ex.map(_cc, cus):
            if r.returncode != 0:
                fails.append((cu, r.stderr[-2000:]))
            else:
                objs.append(o)
    if fails:
        msg = "\n".join(f"{c}:\n{e}" for c, e in fails[:3])
        raise AsmError(f"nvcc multi-TU: {len(fails)}/{len(cus)} units "
                       f"failed:\n{msg}")
    exe = os.path.join(workdir, name)
    cmd = [nvcc] + ([archf] if archf else []) + sorted(objs) \
        + ["-o", exe] + libs
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise AsmError(f"nvcc link failed:\n{r.stderr[-2000:]}")
    return exe
