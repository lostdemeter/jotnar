/* flagship_cuda_exchange: full flagship (splat_soft + v5) on CUDA (v1.2 gate 2c).
 *
 * Host (float, boundary by doctrine): decode/luma/encode-A live in the
 * Python harness with the SAME numpy calls the int path uses; this binary
 * takes A triples + pre-encoded kernels + pre-encoded scalar planes and
 * runs A -> YENH triples entirely on device. Host does gain/chroma after.
 * in.bin: int32 H,W,m_acc,m_cov | int64 tq,qhi1 | A | KX KY KS (9 taps
 * each) | KB0..4 (25 each) | consts[13] (beta att dm ds thr hi k30 w0 w1
 * w2 four half zero) | out.bin: YENH planes. All planes s-block/e-block/
 * z-block. Launch order mirrors chain/splat.py + luminance_finish exactly
 * (composition on the host side, no new math); every kernel is unit-gated.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cuda_dev.cuh"
#include "holo_elem.cu"

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(_e)); exit(1); } } while (0)

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

typedef struct { int8_t *s; int *e; uint8_t *z; } DT;

static int threads_g;
static int blocks_g;

#define LAUNCH(k, ...) do { k<<<blocks_g, threads_g>>>(__VA_ARGS__); \
    CUDA_CHECK(cudaGetLastError()); } while (0)

static DT dalloc(size_t n) {
    DT d;
    CUDA_CHECK(cudaMalloc(&d.s, n));
    CUDA_CHECK(cudaMalloc(&d.e, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d.z, n));
    return d;
}

static void up(DT d, int8_t *hs, int *he, uint8_t *hz, size_t n) {
    CUDA_CHECK(cudaMemcpy(d.s, hs, n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.e, he, sizeof(int) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.z, hz, n, cudaMemcpyHostToDevice));
}

static void down(DT d, int8_t *hs, int *he, uint8_t *hz, size_t n) {
    CUDA_CHECK(cudaMemcpy(hs, d.s, n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(he, d.e, sizeof(int) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hz, d.z, n, cudaMemcpyDeviceToHost));
}

static void read_planar(FILE *f, int8_t *s, int *e, uint8_t *z, size_t n,
                        const char *tag) {
    if (fread(s, 1, n, f) != n) die(tag);
    if (fread(e, sizeof(int), n, f) != n) die(tag);
    if (fread(z, 1, n, f) != n) die(tag);
}

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

static int *dev_int(const int *h, size_t n) {
    int *d;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int) * n));
    CUDA_CHECK(cudaMemcpy(d, h, sizeof(int) * n, cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: flagship_cuda_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[4];
    if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
    int H = hd[0], W = hd[1], m_acc = hd[2], m_cov = hd[3];
    int64_t tq = 0, qhi1 = 0;
    if (fread(&tq, sizeof(tq), 1, f) != 1) die("read tq");
    if (fread(&qhi1, sizeof(qhi1), 1, f) != 1) die("read qhi1");
    size_t N = (size_t)H * W;
    int8_t *h_s = (int8_t *)malloc(N);
    int *h_e = (int *)malloc(sizeof(int) * N);
    uint8_t *h_z = (uint8_t *)malloc(N);
    if (!h_s || !h_e || !h_z) die("oom");
#define RDHOST() read_planar(f, h_s, h_e, h_z, N, "read")
    RDHOST(); /* A */
    DT A = dalloc(N);
    up(A, h_s, h_e, h_z, N);
    int8_t *h_ks = (int8_t *)malloc(25);
    int *h_ke = (int *)malloc(sizeof(int) * 25);
    uint8_t *h_kz = (uint8_t *)malloc(25);
    if (!h_ks || !h_ke || !h_kz) die("oom");
    int ksz[8] = {9, 9, 9, 25, 25, 25, 25, 25};
    DT K[8];
    for (int ki = 0; ki < 8; ki++) {
        size_t kn = (size_t)ksz[ki];
        int8_t *ks = (int8_t *)malloc(kn);
        int *ke = (int *)malloc(sizeof(int) * kn);
        uint8_t *kz = (uint8_t *)malloc(kn);
        if (!ks || !ke || !kz) die("oom");
        read_planar(f, ks, ke, kz, kn, "read kernel");
        K[ki] = dalloc(kn);
        up(K[ki], ks, ke, kz, kn);
        free(ks); free(ke); free(kz);
    }
    enum { CBETA, CATT, CDM, CDS, CTHR, CHI, CK30, CW0, CW1, CW2, CFOUR, CHALF, CZERO, NCONST };
    DT C[NCONST];
    for (int ci = 0; ci < NCONST; ci++) {
        read_planar(f, h_s, h_e, h_z, N, "read const");
        C[ci] = dalloc(N);
        up(C[ci], h_s, h_e, h_z, N);
    }
    fclose(f);
    free(h_s); free(h_e); free(h_z); free(h_ks); free(h_ke); free(h_kz);
    int *d_frac = dev_int(to_int_lut(LUT_FRAC, 13313, "FRAC"), 13313);
    int *d_coarse = dev_int(to_int_lut(LUT_COARSE, 193, "COARSE"), 193);
    int *d_fine = dev_int(to_int_lut(LUT_FINE, 16384, "FINE"), 16384);
    int64_t *d_sigx;
    CUDA_CHECK(cudaMalloc(&d_sigx, sizeof(int64_t) * 65536));
    CUDA_CHECK(cudaMemcpy(d_sigx, LUT_SIGX, sizeof(int64_t) * 65536,
                          cudaMemcpyHostToDevice));
    int *d_sig = dev_int(to_int_lut(LUT_SIG, 524289, "SIG"), 524289);
    threads_g = 256;
    blocks_g = (int)((N + (size_t)threads_g - 1) / (size_t)threads_g);
    int n = (int)N;
#define T3(d) d.s, d.e, d.z
    /* --- structure tensor --- */
    DT gx = dalloc(N), gy = dalloc(N);
    LAUNCH(k_conv_rep, T3(A), H, W, T3(K[0]), 3, m_acc, m_cov,
           d_frac, d_coarse, d_fine, T3(gx));
    LAUNCH(k_conv_rep, T3(A), H, W, T3(K[1]), 3, m_acc, m_cov,
           d_frac, d_coarse, d_fine, T3(gy));
    DT jxx = dalloc(N), jyy = dalloc(N), jxy = dalloc(N);
    LAUNCH(k_tmul, T3(gx), T3(gx), n, T3(jxx));
    LAUNCH(k_tmul, T3(gy), T3(gy), n, T3(jyy));
    LAUNCH(k_tmul, T3(gx), T3(gy), n, T3(jxy));
    DT Jxx = dalloc(N), Jyy = dalloc(N), Jxy = dalloc(N);
    LAUNCH(k_conv_rep, T3(jxx), H, W, T3(K[2]), 3, m_acc, m_cov,
           d_frac, d_coarse, d_fine, T3(Jxx));
    LAUNCH(k_conv_rep, T3(jyy), H, W, T3(K[2]), 3, m_acc, m_cov,
           d_frac, d_coarse, d_fine, T3(Jyy));
    LAUNCH(k_conv_rep, T3(jxy), H, W, T3(K[2]), 3, m_acc, m_cov,
           d_frac, d_coarse, d_fine, T3(Jxy));
    /* --- coherence --- */
    DT d = dalloc(N);
    LAUNCH(k_binop, T3(Jxx), T3(Jyy), n, m_cov, 1, d_frac, d_coarse, d_fine, T3(d));
    DT d2 = dalloc(N), xy2 = dalloc(N), t4 = dalloc(N);
    LAUNCH(k_tmul, T3(d), T3(d), n, T3(d2));
    LAUNCH(k_tmul, T3(Jxy), T3(Jxy), n, T3(xy2));
    LAUNCH(k_tmul, T3(xy2), T3(C[CFOUR]), n, T3(t4));
    DT discA = dalloc(N), disc = dalloc(N), aniso = dalloc(N);
    LAUNCH(k_binop, T3(d2), T3(t4), n, m_acc, 0, d_frac, d_coarse, d_fine, T3(discA));
    LAUNCH(k_rescale, T3(discA), n, m_acc, m_cov, d_frac, d_coarse, d_fine, T3(disc));
    LAUNCH(k_sqrt, T3(disc), n, T3(aniso));
    DT trace = dalloc(N), raw = dalloc(N), coh0 = dalloc(N), coh = dalloc(N);
    LAUNCH(k_binop, T3(Jxx), T3(Jyy), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(trace));
    LAUNCH(k_tdiv, T3(aniso), T3(trace), n, T3(raw));
    LAUNCH(k_wherez, trace.z, T3(raw), T3(C[CZERO]), n, T3(coh0));
    LAUNCH(k_clip, T3(coh0), n, m_cov, 0, qhi1, d_frac, d_coarse, d_fine, T3(coh));
    DT s2 = dalloc(N);
    LAUNCH(k_binop, T3(Jxy), T3(Jxy), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(s2));
    int8_t *d_bucket;
    CUDA_CHECK(cudaMalloc(&d_bucket, N));
    LAUNCH(k_bucket, T3(d), T3(s2), T3(coh), n, m_cov, tq, d_frac, d_bucket);
    /* --- bank (soft) --- */
    DT outs[5];
    for (int bi = 0; bi < 5; bi++) {
        outs[bi] = dalloc(N);
        LAUNCH(k_conv_rep, T3(A), H, W, T3(K[3 + bi]), 5, m_acc, m_cov,
               d_frac, d_coarse, d_fine, T3(outs[bi]));
    }
    DT negd = dalloc(N), negs2 = dalloc(N);
    LAUNCH(k_neg, T3(d), n, T3(negd));
    LAUNCH(k_neg, T3(s2), n, T3(negs2));
    DT rV = dalloc(N), rH = dalloc(N), rD1 = dalloc(N), rD2 = dalloc(N);
    LAUNCH(k_relu, T3(d), n, m_cov, d_frac, d_coarse, d_fine, T3(rV));
    LAUNCH(k_relu, T3(negd), n, m_cov, d_frac, d_coarse, d_fine, T3(rH));
    LAUNCH(k_relu, T3(s2), n, m_cov, d_frac, d_coarse, d_fine, T3(rD1));
    LAUNCH(k_relu, T3(negs2), n, m_cov, d_frac, d_coarse, d_fine, T3(rD2));
    DT tA = dalloc(N), tB = dalloc(N), den = dalloc(N);
    LAUNCH(k_binop, T3(rV), T3(rH), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(tA));
    LAUNCH(k_binop, T3(rD1), T3(rD2), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(tB));
    LAUNCH(k_binop, T3(tA), T3(tB), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(den));
    DT wV = dalloc(N), wH = dalloc(N), wD1 = dalloc(N), wD2 = dalloc(N);
    LAUNCH(k_tdiv, T3(rV), T3(den), n, T3(wV));
    LAUNCH(k_tdiv, T3(rH), T3(den), n, T3(wH));
    LAUNCH(k_tdiv, T3(rD1), T3(den), n, T3(wD1));
    LAUNCH(k_tdiv, T3(rD2), T3(den), n, T3(wD2));
    DT p0 = dalloc(N), p1 = dalloc(N), q0 = dalloc(N);
    DT p3 = dalloc(N), q1 = dalloc(N), p2 = dalloc(N), acc = dalloc(N);
    LAUNCH(k_tmul, T3(wV), T3(outs[0]), n, T3(p0));
    LAUNCH(k_tmul, T3(wH), T3(outs[1]), n, T3(p1));
    LAUNCH(k_binop, T3(p0), T3(p1), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(q0));
    LAUNCH(k_tmul, T3(wD1), T3(outs[3]), n, T3(p3));
    LAUNCH(k_binop, T3(q0), T3(p3), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(q1));
    LAUNCH(k_tmul, T3(wD2), T3(outs[2]), n, T3(p2));
    LAUNCH(k_binop, T3(q1), T3(p2), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(acc));
    DT AS = dalloc(N);
    LAUNCH(k_wherez, den.z, T3(acc), T3(outs[4]), n, T3(AS));
    /* --- beta v5 --- */
    DT D = dalloc(N);
    LAUNCH(k_binop, T3(A), T3(AS), n, m_cov, 1, d_frac, d_coarse, d_fine, T3(D));
    DT c1 = dalloc(N), e1 = dalloc(N), g1 = dalloc(N);
    DT c2 = dalloc(N), e2 = dalloc(N), g2 = dalloc(N);
    LAUNCH(k_binop, T3(coh), T3(C[CTHR]), n, m_cov, 1, d_frac, d_coarse, d_fine, T3(c1));
    LAUNCH(k_tmul, T3(c1), T3(C[CK30]), n, T3(e1));
    LAUNCH(k_sigmoid, T3(e1), n, d_sigx, d_sig, d_coarse, d_fine, T3(g1));
    LAUNCH(k_binop, T3(coh), T3(C[CHI]), n, m_cov, 1, d_frac, d_coarse, d_fine, T3(c2));
    LAUNCH(k_tmul, T3(c2), T3(C[CK30]), n, T3(e2));
    LAUNCH(k_sigmoid, T3(e2), n, d_sigx, d_sig, d_coarse, d_fine, T3(g2));
    DT wA = dalloc(N), w = dalloc(N), base = dalloc(N);
    DT td1 = dalloc(N), td2 = dalloc(N);
    LAUNCH(k_tmul, T3(C[CDM]), T3(g1), n, T3(td1));
    LAUNCH(k_binop, T3(C[CATT]), T3(td1), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(wA));
    LAUNCH(k_tmul, T3(C[CDS]), T3(g2), n, T3(td2));
    LAUNCH(k_binop, T3(wA), T3(td2), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(w));
    LAUNCH(k_tmul, T3(C[CBETA]), T3(w), n, T3(base));
    DT ad = dalloc(N), x4 = dalloc(N), dhat = dalloc(N);
    LAUNCH(k_abs, T3(D), n, T3(ad));
    LAUNCH(k_tmul, T3(ad), T3(C[CFOUR]), n, T3(x4));
    LAUNCH(k_clip, T3(x4), n, m_cov, 0, qhi1, d_frac, d_coarse, d_fine, T3(dhat));
    DT t1 = dalloc(N), t2 = dalloc(N), lt = dalloc(N), logit = dalloc(N);
    DT gate = dalloc(N), scale = dalloc(N), BEFF = dalloc(N);
    LAUNCH(k_tmul, T3(C[CW1]), T3(coh), n, T3(t1));
    LAUNCH(k_tmul, T3(C[CW2]), T3(dhat), n, T3(t2));
    LAUNCH(k_binop, T3(C[CW0]), T3(t1), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(lt));
    LAUNCH(k_binop, T3(lt), T3(t2), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(logit));
    LAUNCH(k_sigmoid, T3(logit), n, d_sigx, d_sig, d_coarse, d_fine, T3(gate));
    LAUNCH(k_binop, T3(C[CHALF]), T3(gate), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(scale));
    LAUNCH(k_tmul, T3(base), T3(scale), n, T3(BEFF));
    /* --- finish --- */
    DT BD = dalloc(N), AE = dalloc(N), YENH = dalloc(N);
    LAUNCH(k_tmul, T3(D), T3(BEFF), n, T3(BD));
    LAUNCH(k_binop, T3(A), T3(BD), n, m_cov, 0, d_frac, d_coarse, d_fine, T3(AE));
    LAUNCH(k_square_clip, T3(AE), n, m_cov, qhi1, d_frac, d_coarse, d_fine, T3(YENH));
    CUDA_CHECK(cudaDeviceSynchronize());
    int8_t *h_os = (int8_t *)malloc(N);
    int *h_oe = (int *)malloc(sizeof(int) * N);
    uint8_t *h_oz = (uint8_t *)malloc(N);
    if (!h_os || !h_oe || !h_oz) die("oom");
    down(YENH, h_os, h_oe, h_oz, N);
    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    if (fwrite(h_os, 1, N, g) != N) die("write s");
    if (fwrite(h_oe, sizeof(int), N, g) != (size_t)N) die("write e");
    if (fwrite(h_oz, 1, N, g) != N) die("write z");
    fclose(g);
    printf("flagship_cuda_exchange: %dx%d m_acc=%d m_cov=%d done\n", H, W, m_acc, m_cov);
    return 0;
}
