/* k_matmul: CUDA lowering of phi-core matmul_int (v1.2 gate 1).
 *
 * Integer-only, FPU-free. One thread per output element (bi,i,j); each
 * thread loops k with the exact holo_matmul product law (sign-mul,
 * exp-add clipped [0,65535], zero-or, to_fixed @ m_acc), int64 register
 * accumulate, from_fixed on device. m_acc is a kernel parameter (the v1.1
 * scale path reaches CUDA too). LUTs arrive as int global pointers
 * (phi-core conv.cu precedent: read-only cache friendly; values fit 32
 * bits per cuda_dev.cuh). B broadcasts when NbB==1.
 *
 * Schedule (recorded per IR.md -- in-bounds int64 sums commute, so any
 * thread mapping is value-identical; this records what WE run):
 *   1D grid, 256 threads/block, flat = (bi*N+i)*M+j; k loop innermost.
 *   Tiling: none (v1 scalar). Reduction order: single-thread accumulate
 *   per output (no cross-thread reduction -- no order effect BY
 *   CONSTRUCTION, stronger than commuted; the roadmap's stated risk
 *   cannot fire in this mapping).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cuda_dev.cuh"

#define ACC_BOUND ((int64_t)1 << 62)

__global__ void k_matmul_elem(
    const int8_t *__restrict__ A_s, const int *__restrict__ A_e,
    const uint8_t *__restrict__ A_z,
    const int8_t *__restrict__ B_s, const int *__restrict__ B_e,
    const uint8_t *__restrict__ B_z,
    int Nb, int N, int K, int M, int NbB, int m_acc,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ O_s, int *__restrict__ O_e, uint8_t *__restrict__ O_z) {
    int64_t flat = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t no = (int64_t)Nb * N * M;
    if (flat >= no) return;
    int j = (int)(flat % M);
    int i = (int)((flat / M) % N);
    int bi = (int)(flat / ((int64_t)N * M));
    int64_t base = frac[0];
    int64_t acc = 0;
    for (int k = 0; k < K; k++) {
        int64_t ai = ((int64_t)bi * N + i) * K + k;
        int64_t bbi = (NbB == 1) ? (int64_t)k * M + j
                                 : (((int64_t)bi * K + k) * M + j);
        uint8_t pz = A_z[ai] | B_z[bbi];
        int pe = (int)A_e[ai] + (int)B_e[bbi] - PHI_BIAS;
        if (pe < 0) pe = 0;
        if (pe > 65535) pe = 65535;
        int ps = (int)A_s[ai] * (int)B_s[bbi];
        acc += to_fixed_elem(ps, pe, pz, m_acc, base, frac);
    }
    assert(acc > -ACC_BOUND && acc < ACC_BOUND);
    from_fixed_elem(acc, m_acc, coarse, fine, &O_s[flat], &O_e[flat], &O_z[flat]);
}

/* Host driver: same file-exchange format as matmul_exchange (same header,
 * same planes, same out layout) so one Python harness gates both sides. */
#include "luts.h"

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(_e)); exit(1); } } while (0)

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static int *to_int_lut(const int64_t *src, int n, const char *tag) {
    int *dst = (int *)malloc(sizeof(int) * (size_t)n);
    if (!dst) die("oom lut");
    for (int i = 0; i < n; i++) {
        if (src[i] > 2147483647LL || src[i] < -2147483648LL) {
            fprintf(stderr, "lut %s[%d] overflows int\n", tag, i);
            exit(1);
        }
        dst[i] = (int)src[i];
    }
    return dst;
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: matmul_cuda_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[6];
    if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
    int Nb = hd[0], N = hd[1], K = hd[2], M = hd[3], NbB = hd[4],
        m_acc = hd[5];
    if (Nb < 1 || N < 1 || K < 1 || M < 1 || (NbB != 1 && NbB != Nb)) die("bad header");
    size_t na = (size_t)Nb * N * K, nb = (size_t)NbB * K * M,
           no = (size_t)Nb * N * M;
    int8_t *h_as = (int8_t *)malloc(na), *h_bs = (int8_t *)malloc(nb);
    int *h_ae_i = (int *)malloc(sizeof(int) * na),
        *h_be_i = (int *)malloc(sizeof(int) * nb);
    uint8_t *h_az = (uint8_t *)malloc(na), *h_bz = (uint8_t *)malloc(nb);
    if (!h_as || !h_bs || !h_ae_i || !h_be_i || !h_az || !h_bz) die("oom");
    /* planes are stored s-block, e-block, z-block per matrix (int == int32) */
    if (fread(h_as, 1, na, f) != na) die("read A_s");
    if (fread(h_ae_i, sizeof(int), na, f) != na) die("read A_e");
    if (fread(h_az, 1, na, f) != na) die("read A_z");
    if (fread(h_bs, 1, nb, f) != nb) die("read B_s");
    if (fread(h_be_i, sizeof(int), nb, f) != nb) die("read B_e");
    if (fread(h_bz, 1, nb, f) != nb) die("read B_z");
    fclose(f);
    int *h_frac = to_int_lut(LUT_FRAC, 13313, "FRAC");
    int *h_coarse = to_int_lut(LUT_COARSE, 193, "COARSE");
    int *h_fine = to_int_lut(LUT_FINE, 16384, "FINE");
    int8_t *d_as, *d_bs, *d_os;
    int *d_ae, *d_be, *d_oe, *d_frac, *d_coarse, *d_fine;
    uint8_t *d_az, *d_bz, *d_oz;
    CUDA_CHECK(cudaMalloc(&d_as, na)); CUDA_CHECK(cudaMalloc(&d_ae, sizeof(int) * na));
    CUDA_CHECK(cudaMalloc(&d_az, na)); CUDA_CHECK(cudaMalloc(&d_bs, nb));
    CUDA_CHECK(cudaMalloc(&d_be, sizeof(int) * nb)); CUDA_CHECK(cudaMalloc(&d_bz, nb));
    CUDA_CHECK(cudaMalloc(&d_os, no)); CUDA_CHECK(cudaMalloc(&d_oe, sizeof(int) * no));
    CUDA_CHECK(cudaMalloc(&d_oz, no));
    CUDA_CHECK(cudaMalloc(&d_frac, sizeof(int) * 13313));
    CUDA_CHECK(cudaMalloc(&d_coarse, sizeof(int) * 193));
    CUDA_CHECK(cudaMalloc(&d_fine, sizeof(int) * 16384));
    CUDA_CHECK(cudaMemcpy(d_as, h_as, na, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ae, h_ae_i, sizeof(int) * na, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_az, h_az, na, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bs, h_bs, nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_be, h_be_i, sizeof(int) * nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bz, h_bz, nb, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frac, h_frac, sizeof(int) * 13313, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coarse, h_coarse, sizeof(int) * 193, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fine, h_fine, sizeof(int) * 16384, cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = (int)((no + (size_t)threads - 1) / (size_t)threads);
    k_matmul_elem<<<blocks, threads>>>(
        d_as, d_ae, d_az, d_bs, d_be, d_bz,
        Nb, N, K, M, NbB, m_acc, d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int8_t *h_os = (int8_t *)malloc(no);
    int *h_oe = (int *)malloc(sizeof(int) * no);
    uint8_t *h_oz = (uint8_t *)malloc(no);
    if (!h_os || !h_oe || !h_oz) die("oom");
    CUDA_CHECK(cudaMemcpy(h_os, d_os, no, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_oe, d_oe, sizeof(int) * no, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_oz, d_oz, no, cudaMemcpyDeviceToHost));
    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    if (fwrite(h_os, 1, no, g) != no) die("write s");
    if (fwrite(h_oe, sizeof(int), no, g) != (size_t)no) die("write e");
    if (fwrite(h_oz, 1, no, g) != no) die("write z");
    fclose(g);
    printf("matmul_cuda_exchange: Nb=%d (%dx%d)x(%dx%d) bcast=%d m_acc=%d done\n",
           Nb, N, K, K, M, NbB == 1, m_acc);
    return 0;
}
