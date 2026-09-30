/* holo_units: CUDA lowerings of holo_conv + holo_mux (v1.2 gate 2a).
 *
 * Integer-only, FPU-free. Each kernel mirrors its proven C twin
 * line-for-line (holo_conv.c / holo_ops.c, both file-exchange-gated
 * bit-exact vs numpy) -- the C layer is the reference for the CUDA port,
 * so any slip is isolated to parallelism. LUTs as int globals (conv.cu
 * precedent). m_acc/m_out parameterized (v1.1 scale path in CUDA too).
 *
 * Schedule (recorded per IR.md -- in-bounds int64 sums commute; this is
 * what WE run): k_conv_rep: one thread per output pixel, ky->kx taps
 * innermost, replicate-edge clamp; k_mux: one thread per element.
 * Tiling: none (v1 scalar). No cross-thread reductions anywhere.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cuda_dev.cuh"
#include "luts.h"

/* elementwise kernels live in holo_elem.cu (compiled in below) */
#include "holo_elem.cu"

#define ACC_BOUND ((int64_t)1 << 62)

__global__ void k_conv_rep(
    const int8_t *__restrict__ in_s, const int *__restrict__ in_e,
    const uint8_t *__restrict__ in_z, int H, int W,
    const int8_t *__restrict__ k_s, const int *__restrict__ k_e,
    const uint8_t *__restrict__ k_z, int KH,
    int m_acc, int m_out,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ O_s, int *__restrict__ O_e, uint8_t *__restrict__ O_z) {
    int64_t flat = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = (int64_t)H * W;
    if (flat >= N) return;
    int ox = (int)(flat % W), oy = (int)(flat / W);
    int r = KH / 2;
    int64_t base = frac[0];
    int64_t acc = 0;
    for (int ky = 0; ky < KH; ky++) {
        int iy = oy + ky - r;
        if (iy < 0) iy = 0;
        if (iy >= H) iy = H - 1;
        for (int kx = 0; kx < KH; kx++) {
            int ix = ox + kx - r;
            if (ix < 0) ix = 0;
            if (ix >= W) ix = W - 1;
            int64_t ai = (int64_t)iy * W + ix;
            int64_t wi = (int64_t)ky * KH + kx;
            uint8_t pz = in_z[ai] | k_z[wi];
            int pe = (int)in_e[ai] + (int)k_e[wi] - PHI_BIAS;
            if (pe < 0) pe = 0;
            if (pe > 65535) pe = 65535;
            int ps = (int)in_s[ai] * (int)k_s[wi];
            acc += to_fixed_elem((int8_t)ps, pe, pz, m_acc, base, frac);
        }
    }
    assert(acc > -ACC_BOUND && acc < ACC_BOUND);
    int8_t ms;
    int me;
    uint8_t mz;
    from_fixed_elem(acc, m_acc, coarse, fine, &ms, &me, &mz);
    if (m_acc == m_out) {
        O_s[flat] = ms;
        O_e[flat] = me;
        O_z[flat] = mz;
    } else {
        int64_t q2 = to_fixed_elem(ms, me, mz, m_out, base, frac);
        from_fixed_elem(q2, m_out, coarse, fine, &O_s[flat], &O_e[flat], &O_z[flat]);
    }
}

__global__ void k_mux(
    const int8_t *__restrict__ S, const int *__restrict__ E,
    const uint8_t *__restrict__ Z, int nstreams,
    const int8_t *__restrict__ bucket, int n,
    int8_t *__restrict__ O_s, int *__restrict__ O_e, uint8_t *__restrict__ O_z) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = (int)bucket[i];
    if (b < 0) b = 0;
    if (b >= nstreams) b = nstreams - 1;
    int64_t si = (int64_t)b * n + i;
    O_s[i] = S[si];
    O_e[i] = E[si];
    O_z[i] = Z[si];
}

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

static void *xmalloc_raw(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("oom");
    return p;
}
#define xmalloc(T, n) ((T *)xmalloc_raw(sizeof(T) * (size_t)(n)))

static void read_planar(FILE *f, int8_t *s, int *e, uint8_t *z, size_t n,
                        const char *tag) {
    if (fread(s, 1, n, f) != n) die(tag);
    if (fread(e, sizeof(int), n, f) != n) die(tag);
    if (fread(z, 1, n, f) != n) die(tag);
}

static void write_planar(FILE *g, int8_t *s, int *e, uint8_t *z, size_t n,
                         const char *tag) {
    if (fwrite(s, 1, n, g) != n) die(tag);
    if (fwrite(e, sizeof(int), n, g) != n) die(tag);
    if (fwrite(z, 1, n, g) != n) die(tag);
}

static int *dev_int(const int *h, size_t n) {
    int *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int) * n));
    CUDA_CHECK(cudaMemcpy(d, h, sizeof(int) * n, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: units_cuda_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t task;
    if (fread(&task, sizeof(task), 1, f) != 1) die("read task");
    int *d_frac = dev_int(to_int_lut(LUT_FRAC, 13313, "FRAC"), 13313);
    int *d_coarse = dev_int(to_int_lut(LUT_COARSE, 193, "COARSE"), 193);
    int *d_fine = dev_int(to_int_lut(LUT_FINE, 16384, "FINE"), 16384);
    int threads = 256;
    if (task == 0) {
        int32_t hd[5];
        if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
        int H = hd[0], W = hd[1], KH = hd[2], m_acc = hd[3], m_out = hd[4];
        size_t N = (size_t)H * W, K = (size_t)KH * KH;
        int8_t *h_is = xmalloc(int8_t, N), *h_ks = xmalloc(int8_t, K);
        int *h_ie = xmalloc(int, N), *h_ke = xmalloc(int, K);
        uint8_t *h_iz = xmalloc(uint8_t, N), *h_kz = xmalloc(uint8_t, K);
        read_planar(f, h_is, h_ie, h_iz, N, "read in");
        read_planar(f, h_ks, h_ke, h_kz, K, "read k");
        fclose(f);
        int8_t *d_is, *d_ks, *d_os;
        int *d_ie, *d_ke, *d_oe;
        uint8_t *d_iz, *d_kz, *d_oz;
        CUDA_CHECK(cudaMalloc(&d_is, N)); CUDA_CHECK(cudaMalloc(&d_ie, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_iz, N)); CUDA_CHECK(cudaMalloc(&d_ks, K));
        CUDA_CHECK(cudaMalloc(&d_ke, sizeof(int) * K)); CUDA_CHECK(cudaMalloc(&d_kz, K));
        CUDA_CHECK(cudaMalloc(&d_os, N)); CUDA_CHECK(cudaMalloc(&d_oe, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&d_oz, N));
        CUDA_CHECK(cudaMemcpy(d_is, h_is, N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ie, h_ie, sizeof(int) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_iz, h_iz, N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ks, h_ks, K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ke, h_ke, sizeof(int) * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kz, h_kz, K, cudaMemcpyHostToDevice));
        int blocks = (int)((N + (size_t)threads - 1) / (size_t)threads);
        k_conv_rep<<<blocks, threads>>>(
            d_is, d_ie, d_iz, H, W, d_ks, d_ke, d_kz, KH,
            m_acc, m_out, d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int8_t *h_os = xmalloc(int8_t, N);
        int *h_oe = xmalloc(int, N);
        uint8_t *h_oz = xmalloc(uint8_t, N);
        CUDA_CHECK(cudaMemcpy(h_os, d_os, N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oe, d_oe, sizeof(int) * N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oz, d_oz, N, cudaMemcpyDeviceToHost));
        FILE *g = fopen(argv[2], "wb");
        if (!g) die("open out");
        write_planar(g, h_os, h_oe, h_oz, N, "write out");
        fclose(g);
        printf("units_cuda_exchange: conv %dx%d KH=%d m_acc=%d m_out=%d done\n",
               H, W, KH, m_acc, m_out);
    } else if (task == 1) {        int32_t hd[2];
        if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
        int n = hd[0], ns = hd[1];
        if (n < 1 || ns < 1) die("bad header");
        size_t nt = (size_t)n * (size_t)ns;
        int8_t *h_s = xmalloc(int8_t, nt), *h_b = xmalloc(int8_t, (size_t)n);
        int *h_e = xmalloc(int, nt);
        uint8_t *h_z = xmalloc(uint8_t, nt);
        read_planar(f, h_s, h_e, h_z, nt, "read streams");
        if (fread(h_b, 1, (size_t)n, f) != (size_t)n) die("read bucket");
        fclose(f);
        int8_t *d_s, *d_b, *d_os;
        int *d_e, *d_oe;
        uint8_t *d_z, *d_oz;
        CUDA_CHECK(cudaMalloc(&d_s, nt)); CUDA_CHECK(cudaMalloc(&d_e, sizeof(int) * nt));
        CUDA_CHECK(cudaMalloc(&d_z, nt)); CUDA_CHECK(cudaMalloc(&d_b, (size_t)n));
        CUDA_CHECK(cudaMalloc(&d_os, (size_t)n)); CUDA_CHECK(cudaMalloc(&d_oe, sizeof(int) * (size_t)n));
        CUDA_CHECK(cudaMalloc(&d_oz, (size_t)n));
        CUDA_CHECK(cudaMemcpy(d_s, h_s, nt, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_e, h_e, sizeof(int) * nt, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_z, h_z, nt, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b, (size_t)n, cudaMemcpyHostToDevice));
        int blocks = (int)(((size_t)n + (size_t)threads - 1) / (size_t)threads);
        k_mux<<<blocks, threads>>>(d_s, d_e, d_z, ns, d_b, n, d_os, d_oe, d_oz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int8_t *h_os = xmalloc(int8_t, (size_t)n);
        int *h_oe = xmalloc(int, (size_t)n);
        uint8_t *h_oz = xmalloc(uint8_t, (size_t)n);
        CUDA_CHECK(cudaMemcpy(h_os, d_os, (size_t)n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oe, d_oe, sizeof(int) * (size_t)n, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oz, d_oz, (size_t)n, cudaMemcpyDeviceToHost));
        FILE *g = fopen(argv[2], "wb");
        if (!g) die("open out");
        write_planar(g, h_os, h_oe, h_oz, (size_t)n, "write out");
        fclose(g);
        printf("units_cuda_exchange: mux n=%d ns=%d done\n", n, ns);
    } else if (task >= 2 && task <= 10) {
        /* elementwise tasks (kernels in holo_elem.cu):
         * 2=sqrt[n]+in 3=tmul[n]+A+B 10=tdiv[n]+A+B 4=binop[n,m,is_sub]+A+B
         * 5=square_clip[n,m]+int64 qhi+in 6=sigmoid[n]+in 7=abs[n]+in
         * 8=relu[n,m]+in 9=clip[n,m]+int64 qlo,qhi+in */
        int32_t hd[3];
        int need = (task == 4) ? 3 : (task == 5 || task == 8) ? 2
            : (task == 9) ? 2 : 1;
        if (fread(hd, sizeof(int32_t), (size_t)need, f) != (size_t)need) die("read header");
        int n = hd[0];
        if (n < 1) die("bad header");
        size_t un = (size_t)n;
        int64_t qlo = 0, qhi = 0;
        if (task == 5) {
            if (fread(&qhi, sizeof(qhi), 1, f) != 1) die("read qhi");
        } else if (task == 9) {
            if (fread(&qlo, sizeof(qlo), 1, f) != 1) die("read qlo");
            if (fread(&qhi, sizeof(qhi), 1, f) != 1) die("read qhi");
        }
        int two = (task == 3 || task == 4 || task == 10);
        int8_t *h_as = xmalloc(int8_t, un), *h_bs = two ? xmalloc(int8_t, un) : NULL;
        int *h_ae = xmalloc(int, un), *h_be = two ? xmalloc(int, un) : NULL;
        uint8_t *h_az = xmalloc(uint8_t, un), *h_bz = two ? xmalloc(uint8_t, un) : NULL;
        read_planar(f, h_as, h_ae, h_az, un, "read A");
        if (two) read_planar(f, h_bs, h_be, h_bz, un, "read B");
        fclose(f);
        int8_t *d_as, *d_bs = NULL, *d_os;
        int *d_ae, *d_be = NULL, *d_oe;
        uint8_t *d_az, *d_bz = NULL, *d_oz;
        CUDA_CHECK(cudaMalloc(&d_as, un)); CUDA_CHECK(cudaMalloc(&d_ae, sizeof(int) * un));
        CUDA_CHECK(cudaMalloc(&d_az, un));
        CUDA_CHECK(cudaMalloc(&d_os, un)); CUDA_CHECK(cudaMalloc(&d_oe, sizeof(int) * un));
        CUDA_CHECK(cudaMalloc(&d_oz, un));
        CUDA_CHECK(cudaMemcpy(d_as, h_as, un, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ae, h_ae, sizeof(int) * un, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_az, h_az, un, cudaMemcpyHostToDevice));
        if (two) {
            CUDA_CHECK(cudaMalloc(&d_bs, un)); CUDA_CHECK(cudaMalloc(&d_be, sizeof(int) * un));
            CUDA_CHECK(cudaMalloc(&d_bz, un));
            CUDA_CHECK(cudaMemcpy(d_bs, h_bs, un, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_be, h_be, sizeof(int) * un, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_bz, h_bz, un, cudaMemcpyHostToDevice));
        }
        int *d_sigx32 = NULL, *d_sig = NULL;
        int64_t *d_sigx = NULL;
        if (task == 6) {
            CUDA_CHECK(cudaMalloc(&d_sigx, sizeof(int64_t) * 65536));
            CUDA_CHECK(cudaMemcpy(d_sigx, LUT_SIGX, sizeof(int64_t) * 65536,
                                  cudaMemcpyHostToDevice));
            d_sig = dev_int(to_int_lut(LUT_SIG, 524289, "SIG"), 524289);
        }
        int blocks = (int)((un + (size_t)threads - 1) / (size_t)threads);
        int m = (need >= 2) ? hd[1] : 0;
        if (task == 2) k_sqrt<<<blocks, threads>>>(d_as, d_ae, d_az, n, d_os, d_oe, d_oz);
        else if (task == 3) k_tmul<<<blocks, threads>>>(d_as, d_ae, d_az, d_bs, d_be, d_bz, n, d_os, d_oe, d_oz);
        else if (task == 10) k_tdiv<<<blocks, threads>>>(d_as, d_ae, d_az, d_bs, d_be, d_bz, n, d_os, d_oe, d_oz);
        else if (task == 4) k_binop<<<blocks, threads>>>(d_as, d_ae, d_az, d_bs, d_be, d_bz, n, m, hd[2], d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
        else if (task == 5) k_square_clip<<<blocks, threads>>>(d_as, d_ae, d_az, n, m, qhi, d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
        else if (task == 6) k_sigmoid<<<blocks, threads>>>(d_as, d_ae, d_az, n, d_sigx, d_sig, d_coarse, d_fine, d_os, d_oe, d_oz);
        else if (task == 7) k_abs<<<blocks, threads>>>(d_as, d_ae, d_az, n, d_os, d_oe, d_oz);
        else if (task == 8) k_relu<<<blocks, threads>>>(d_as, d_ae, d_az, n, m, d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
        else k_clip<<<blocks, threads>>>(d_as, d_ae, d_az, n, m, qlo, qhi, d_frac, d_coarse, d_fine, d_os, d_oe, d_oz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        int8_t *h_os = xmalloc(int8_t, un);
        int *h_oe = xmalloc(int, un);
        uint8_t *h_oz = xmalloc(uint8_t, un);
        CUDA_CHECK(cudaMemcpy(h_os, d_os, un, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oe, d_oe, sizeof(int) * un, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_oz, d_oz, un, cudaMemcpyDeviceToHost));
        FILE *g = fopen(argv[2], "wb");
        if (!g) die("open out");
        write_planar(g, h_os, h_oe, h_oz, un, "write out");
        fclose(g);
        printf("units_cuda_exchange: elem task=%d n=%d done\n", task, n);
    } else {
        die("bad task (want 0=conv, 1=mux, 2..10=elem)");
    }
    return 0;
}
