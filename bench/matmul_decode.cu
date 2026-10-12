/* MATMUL decode-shape microbench (round 1: isolation + determinism).
 *
 * Shapes = the 113 decode GEMMs (M=1; fused qkv/upgate already in):
 *   qkv     1 x 4608   x 3584
 *   oproj   1 x 3584   x 3584
 *   upgate  1 x 37888  x 3584
 *   down    1 x 3584   x 18944
 *   unembed 1 x 152064 x 3584
 * Plus one M=16 prefill-context row per shape (same N,K).
 *
 * Candidates:
 *   A  cublasGemmEx F16->F32, COMPUTE_32F, GEMM_DEFAULT (production)
 *   B  cublasGemmEx FAST_TF32 (VALUES DIFFER -- wall only, parity unjudged)
 *   C  hand GEMV: one block per output row-element, sequential K loop
 *      (fixed summation order => deterministic by construction)
 *
 * Fixed inputs (LCG fill, same every run), FNV-1a hash of C out.
 * Run N separate processes, diff hashes: bit-exact or it doesn't land.
 * Usage: nvcc -O2 -arch=native -o /tmp/mm_ub matmul_decode.cu -lcublas
 *        ./mm_ub [iters=200]   # prints ms/call, GB/s (weight bytes), hash
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

#define CE(x) do { cudaError_t _e = (x); if (_e) { \
  fprintf(stderr, "cuda %d @%d\n", (int)_e, __LINE__); return 1; } } while (0)
#define BE(x) do { cublasStatus_t _e = (x); if (_e) { \
  fprintf(stderr, "cublas %d @%d\n", (int)_e, __LINE__); return 1; } } while (0)

static uint32_t lcg = 0x12345678u;
static __half rndh() {
  lcg = lcg * 1664525u + 1013904223u;
  float v = ((lcg >> 9) & 0xffff) / 65535.0f - 0.5f; /* [-0.5, 0.5) */
  return __float2half(v);
}

/* candidate C: GEMV, one block per output element, grid-stride K loop */
__global__ void gemv_dot(const __half *A /*MxK*/, const __half *B /*NxK*/,
                         float *C /*MxN*/, int M, int N, int K) {
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;
  float acc = 0.0f;
  const __half *ar = A + (size_t)m * K;
  const __half *br = B + (size_t)n * K;
  for (int k = 0; k < K; k++) acc += __half2float(ar[k]) * __half2float(br[k]);
  C[(size_t)m * N + n] = acc;
}

static uint64_t fnv(const float *d, size_t n) {
  /* hash on device-memcopied host buffer */
  static float *h = 0; static size_t cap = 0;
  if (cap < n) { delete[] h; h = new float[n]; cap = n; }
  CE(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
  uint64_t x = 1469598103934665603ull;
  const uint32_t *w = (const uint32_t *)h; /* 1 float = 1 word */
  for (size_t i = 0; i < n; i++) {
    x ^= w[i]; x *= 1099511628211ull;
  }
  return x;
}

struct Shape { const char *tag; int M, N, K; };

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;
  /* production mirror: managed workspace + cublasSetWorkspace */
  cublasHandle_t h; BE(cublasCreate(&h));
  void *ws = 0; CE(cudaMallocManaged(&ws, 32 << 20));
  BE(cublasSetWorkspace(h, ws, 32 << 20));
  const float kOne = 1.0f, kZero = 0.0f;
  cudaEvent_t t0, t1; CE(cudaEventCreate(&t0)); CE(cudaEventCreate(&t1));

  Shape shapes[] = {
    {"qkv", 1, 4608, 3584}, {"oproj", 1, 3584, 3584},
    {"upgate", 1, 37888, 3584}, {"down", 1, 3584, 18944},
    {"unembed", 1, 152064, 3584},
  };
  /* biggest buffers once */
  const int MXK = 16 * 18944, MXN = 16 * 152064;
  const size_t BNEED = (size_t)152064 * 3584; /* max N*K over shapes */
  __half *A, *B; float *C;
  CE(cudaMalloc(&A, (size_t)MXK * 2));
  CE(cudaMalloc(&B, BNEED * 2));
  CE(cudaMalloc(&C, (size_t)MXN * 4));
  /* fixed fill */
  __half *hA = new __half[MXK], *hB = new __half[BNEED];
  for (int i = 0; i < MXK; i++) hA[i] = rndh();
  for (size_t i = 0; i < BNEED; i++) hB[i] = rndh();
  CE(cudaMemcpy(A, hA, (size_t)MXK * 2, cudaMemcpyHostToDevice));
  CE(cudaMemcpy(B, hB, BNEED * 2, cudaMemcpyHostToDevice));

  for (int s = 0; s < 5; s++) {
    for (int mi = 0; mi < 2; mi++) {
      int M = mi ? 16 : 1, N = shapes[s].N, K = shapes[s].K;
      size_t wbytes = (size_t)N * K * 2;
      float ms;
      /* A: cublas production */
      CE(cudaEventRecord(t0));
      for (int i = 0; i < iters; i++)
        BE(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &kOne, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                        &kZero, C, CUDA_R_32F, N,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      CE(cudaEventRecord(t1)); CE(cudaEventSynchronize(t1));
      CE(cudaEventElapsedTime(&ms, t0, t1));
      uint64_t ha = fnv(C, (size_t)M * N);
      /* B: TF32 (values differ) */
      CE(cudaEventRecord(t0));
      for (int i = 0; i < iters; i++)
        BE(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &kOne, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                        &kZero, C, CUDA_R_32F, N,
                        CUBLAS_COMPUTE_32F_FAST_TF32,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      CE(cudaEventRecord(t1)); CE(cudaEventSynchronize(t1));
      float msB; CE(cudaEventElapsedTime(&msB, t0, t1));
      /* C: hand GEMV (M=1 only; M=16 uses 16-block y grid) */
      dim3 blk(256), grd((N + 255) / 256, M);
      CE(cudaEventRecord(t0));
      for (int i = 0; i < iters; i++)
        gemv_dot<<<grd, blk>>>(A, B, C, M, N, K);
      CE(cudaEventRecord(t1)); CE(cudaEventSynchronize(t1));
      float msC; CE(cudaEventElapsedTime(&msC, t0, t1));
      CE(cudaGetLastError());
      uint64_t hc = fnv(C, (size_t)M * N);
      double gbs = wbytes / 1e9 / (ms / iters * 1e-3);
      printf("%-7s M=%-2d N=%-6d K=%-5d | A %8.3fms %7.1fGB/s h=%016llx | "
             "B %8.3fms | C %8.3fms h=%016llx\n",
             shapes[s].tag, M, N, K, ms / iters, gbs,
             (unsigned long long)ha, msB / iters, msC / iters,
             (unsigned long long)hc);
      fflush(stdout);
    }
  }
  return 0;
}
