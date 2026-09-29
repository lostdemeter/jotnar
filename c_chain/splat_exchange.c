/* splat_exchange: file-exchange driver for full splat_blur (step 1+2 gates).
 *
 * Mirrors chain/splat.py op-for-op on TRIPLE arrays (roundtrips included:
 * every binop is to_fixed -> int64 add/sub -> from_fixed, exactly like
 * numpy's binop_fixed; compares read fixed counts like numpy's to_fixed).
 * All weight/scalar triples arrive pre-encoded from Python (identical values
 * to kernel_triples/const_trip: no FP encode divergence). FPU-free C.
 *
 * Layout (no structs on the wire):
 *   int32 H,W,m_acc,m_cov | int64 tq,qlo,qhi | int8 four_s | int32 four_e |
 *   uint8 four_z | A planes (s[N],e[N],z[N]) | 6 kernels: int32 KH + planes |
 *   out: out planes (s,e,z) + int8 bucket[N].
 * Kernel order: sobelX, sobelY, smooth, bankV, bankH, bankD1, bankD2, bankIso.
 * Bucket rule (mirrors splat.py v1.1): gate shut -> 4; |2Jxy|>|diff| ->
 * diagonal by sign(sq) (sq<0 -> 2, else 3); diff>0 -> 0 else 1.
 * Usage: splat_exchange in.bin out.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "holo_conv.h"
#include "holo_ops.h"
#include "bridge.h"
#include "luts.h"

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static uint8_t *rd1(FILE *f, size_t n) {
    uint8_t *b = malloc(n ? n : 1);
    if (!b) die("oom");
    if (n && fread(b, 1, n, f) != n) die("read u8");
    return b;
}

static int8_t *rdS(FILE *f, size_t n) { return (int8_t *)rd1(f, n); }

static int32_t *rdE(FILE *f, size_t n) {
    int32_t *b = malloc((n ? n : 1) * sizeof(int32_t));
    if (!b) die("oom");
    if (n && fread(b, sizeof(int32_t), n, f) != n) die("read e");
    return b;
}

static trip_t *totrip(const int8_t *s, const int32_t *e, const uint8_t *z, int n) {
    trip_t *t = malloc(sizeof(trip_t) * (size_t)(n ? n : 1));
    if (!t) die("oom");
    for (int i = 0; i < n; i++) { t[i].s = s[i]; t[i].e = e[i]; t[i].z = z[i]; }
    return t;
}

static trip_t *read_kernel(FILE *f, int *kh) {
    int32_t k;
    if (fread(&k, sizeof k, 1, f) != 1) die("read KH");
    *kh = k;
    int n = k * k;
    int8_t *s = rdS(f, (size_t)n);
    int32_t *e = rdE(f, (size_t)n);
    uint8_t *z = rd1(f, (size_t)n);
    trip_t *t = totrip(s, e, z, n);
    free(s); free(e); free(z);
    return t;
}

/* binop on triples @ m (mirrors binop_fixed: bridge + assert via fq_add/sub) */
static trip_t *binop_t(const trip_t *a, const trip_t *b, int n, int m, int sub) {
    int64_t *qa = malloc(sizeof(int64_t) * (size_t)n);
    int64_t *qb = malloc(sizeof(int64_t) * (size_t)n);
    int64_t *qo = malloc(sizeof(int64_t) * (size_t)n);
    trip_t *ac = malloc(sizeof(trip_t) * (size_t)n);
    trip_t *bc = malloc(sizeof(trip_t) * (size_t)n);
    trip_t *o = malloc(sizeof(trip_t) * (size_t)n);
    if (!qa || !qb || !qo || !ac || !bc || !o) die("oom");
    for (int i = 0; i < n; i++) { ac[i] = a[i]; bc[i] = b[i]; }
    fq_t fa = { qa, n, 0 }, fb = { qb, n, 0 }, fo = { qo, n, 0 };
    to_fixed(ac, &fa, m);
    to_fixed(bc, &fb, m);
    if (sub) fq_sub(&fa, &fb, &fo);
    else fq_add(&fa, &fb, &fo);
    triples_from_fixed(&fo, o);
    free(qa); free(qb); free(qo); free(ac); free(bc);
    return o;
}

/* triples -> fixed counts @ m (fresh buffer; mirrors S.to_fixed) */
static int64_t *t2q(const trip_t *t, int n, int m) {
    int64_t *q = malloc(sizeof(int64_t) * (size_t)(n ? n : 1));
    trip_t *tc = malloc(sizeof(trip_t) * (size_t)(n ? n : 1));
    if (!q || !tc) die("oom");
    for (int i = 0; i < n; i++) tc[i] = t[i];
    fq_t f = { q, n, 0 };
    to_fixed(tc, &f, m);
    free(tc);
    return q;
}

/* fixed counts @ m -> triples (mirrors S.from_fixed) */
static trip_t *q2t(const int64_t *q, int n, int m) {
    int64_t *qc = malloc(sizeof(int64_t) * (size_t)(n ? n : 1));
    trip_t *t = malloc(sizeof(trip_t) * (size_t)(n ? n : 1));
    if (!qc || !t) die("oom");
    for (int i = 0; i < n; i++) qc[i] = q[i];
    fq_t f = { qc, n, m };
    triples_from_fixed(&f, t);
    free(qc);
    return t;
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: splat_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[5];
    if (fread(hd, sizeof hd, 1, f) != 1) die("read header");
    int H = hd[0], W = hd[1], m_acc = hd[2], m_cov = hd[3];
    int fused = hd[4];
    int N = H * W;
    int64_t tq, qlo, qhi;
    if (fread(&tq, sizeof tq, 1, f) != 1) die("read tq");
    if (fread(&qlo, sizeof qlo, 1, f) != 1) die("read qlo");
    if (fread(&qhi, sizeof qhi, 1, f) != 1) die("read qhi");
    trip_t four;
    if (fread(&four.s, 1, 1, f) != 1) die("read four_s");
    if (fread(&four.e, sizeof four.e, 1, f) != 1) die("read four_e");
    if (fread(&four.z, 1, 1, f) != 1) die("read four_z");

    int8_t *as = rdS(f, (size_t)N);
    int32_t *ae = rdE(f, (size_t)N);
    uint8_t *az = rd1(f, (size_t)N);
    trip_t *A = totrip(as, ae, az, N);
    free(as); free(ae); free(az);

    int k1, k2, k3, k4, k5, k6, k7, k8;
    trip_t *sX = read_kernel(f, &k1);
    trip_t *sY = read_kernel(f, &k2);
    trip_t *smo = read_kernel(f, &k3);
    trip_t *bV = read_kernel(f, &k4);
    trip_t *bH = read_kernel(f, &k5);
    trip_t *bD1 = read_kernel(f, &k6);
    trip_t *bD2 = read_kernel(f, &k7);
    trip_t *bI = read_kernel(f, &k8);
    fclose(f);

    trip_t *gx = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *gy = malloc(sizeof(trip_t) * (size_t)N);
    if (!gx || !gy) die("oom");

    /* gradients + tensor products + smoothing (all triples) */
    holo_conv(A, H, W, sX, k1, m_acc, m_cov, gx);
    holo_conv(A, H, W, sY, k2, m_acc, m_cov, gy);
    trip_t *pxx = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *pyy = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *pxy = malloc(sizeof(trip_t) * (size_t)N);
    if (!pxx || !pyy || !pxy) die("oom");
    holo_mul(gx, gx, pxx, N);
    holo_mul(gy, gy, pyy, N);
    holo_mul(gx, gy, pxy, N);
    trip_t *sx = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *sy = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *sz = malloc(sizeof(trip_t) * (size_t)N);
    if (!sx || !sy || !sz) die("oom");
    holo_conv(pxx, H, W, smo, k3, m_acc, m_cov, sx);
    holo_conv(pyy, H, W, smo, k3, m_acc, m_cov, sy);
    holo_conv(pxy, H, W, smo, k3, m_acc, m_cov, sz);
    free(pxx); free(pyy); free(pxy);

    /* d = Jxx-Jyy; s2 = 2*Jxy (triples @ m_cov, like numpy) */
    trip_t *dt = binop_t(sx, sy, N, m_cov, 1);
    trip_t *s2t = binop_t(sz, sz, N, m_cov, 0);

    /* discriminant squares exact; sum @ m_acc; rescale_ to m_cov */
    trip_t *d2 = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *x2 = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *t4 = malloc(sizeof(trip_t) * (size_t)N);
    if (!d2 || !x2 || !t4) die("oom");
    holo_mul(dt, dt, d2, N);
    holo_mul(sz, sz, x2, N);
    for (int i = 0; i < N; i++) holo_mul(&x2[i], &four, &t4[i], 1);
    int64_t *qa = t2q(d2, N, m_acc), *qb = t2q(t4, N, m_acc);
    int64_t *qacc = malloc(sizeof(int64_t) * (size_t)N);
    int64_t *qcov = malloc(sizeof(int64_t) * (size_t)N);
    if (!qacc || !qcov) die("oom");
    for (int i = 0; i < N; i++) qacc[i] = qa[i] + qb[i];
    fq_t fa = { qacc, N, m_acc }, fb = { qcov, N, 0 };
    fq_rescale(&fa, m_cov, &fb);
    trip_t *disc = q2t(qcov, N, m_cov);
    free(qa); free(qb); free(qacc); free(qcov); free(d2); free(x2); free(t4);

    /* aniso = sqrt(disc); trace; raw = div; 0/0 -> 0; clip [qlo,qhi] */
    trip_t *aniso = malloc(sizeof(trip_t) * (size_t)N);
    if (!aniso) die("oom");
    holo_sqrt(disc, aniso, N);
    free(disc);
    trip_t *trace = binop_t(sx, sy, N, m_cov, 0);
    trip_t *raw = malloc(sizeof(trip_t) * (size_t)N);
    if (!raw) die("oom");
    holo_div(aniso, trace, raw, N);
    free(aniso);
    int64_t *qc = calloc((size_t)(N ? N : 1), sizeof(int64_t));
    if (!qc) die("oom");
    for (int i = 0; i < N; i++) {
        if (trace[i].z) {
            qc[i] = 0;
        } else {
            int64_t qq;
            if (raw[i].z) qq = 0;
            else {
                int64_t dd = (int64_t)m_cov - (int64_t)raw[i].e;
                qq = dd < 0 ? (int64_t)raw[i].s * LUT_FRAC[0]
                    : (dd > FRAC_CAP ? 0 : (int64_t)raw[i].s * LUT_FRAC[dd]);
            }
            if (qq < qlo) qq = qlo;
            if (qq > qhi) qq = qhi;
            qc[i] = qq;
        }
    }
    free(raw);
    trip_t *coh = q2t(qc, N, m_cov);
    free(qc);

    /* bucket from integer compares (mirrors numpy exactly) */
    int64_t *cohq = t2q(coh, N, m_cov);
    int64_t *dqq = t2q(dt, N, m_cov);
    int64_t *sqq = t2q(s2t, N, m_cov);
    int8_t *bucket = malloc((size_t)N);
    if (!bucket) die("oom");
    for (int i = 0; i < N; i++) {
        int gate = cohq[i] >= tq;
        int64_t ad = dqq[i] < 0 ? -dqq[i] : dqq[i];
        int64_t as2 = sqq[i] < 0 ? -sqq[i] : sqq[i];
        int diag = as2 > ad;
        int vert = dqq[i] > 0;
        int dpos = sqq[i] >= 0;
        bucket[i] = (int8_t)(!gate ? 4 : (!diag ? (vert ? 0 : 1) : (dpos ? 3 : 2)));
    }
    free(cohq); free(dqq); free(sqq);
    free(coh); free(dt); free(s2t); free(trace);
    free(sx); free(sy); free(sz); free(gx); free(gy);

    /* bank blurs + mux (5 streams), or the fused gathered pass */
    trip_t *oV = NULL, *oH = NULL, *oD1 = NULL, *oD2 = NULL, *oI = NULL;
    trip_t *out = malloc(sizeof(trip_t) * (size_t)N);
    if (!out) die("oom");
    if (fused) {
        const trip_t *bank[5] = { bV, bH, bD1, bD2, bI };
        holo_bank_fused(A, H, W, bank, k4, bucket, m_acc, m_cov, out);
    } else {
        oV = malloc(sizeof(trip_t) * (size_t)N);
        oH = malloc(sizeof(trip_t) * (size_t)N);
        oD1 = malloc(sizeof(trip_t) * (size_t)N);
        oD2 = malloc(sizeof(trip_t) * (size_t)N);
        oI = malloc(sizeof(trip_t) * (size_t)N);
        if (!oV || !oH || !oD1 || !oD2 || !oI) die("oom");
        holo_conv(A, H, W, bV, k4, m_acc, m_cov, oV);
        holo_conv(A, H, W, bH, k5, m_acc, m_cov, oH);
        holo_conv(A, H, W, bD1, k6, m_acc, m_cov, oD1);
        holo_conv(A, H, W, bD2, k7, m_acc, m_cov, oD2);
        holo_conv(A, H, W, bI, k8, m_acc, m_cov, oI);
        const trip_t *streams[5] = { oV, oH, oD1, oD2, oI };
        holo_mux(streams, 5, bucket, out, N);
        free(oV); free(oH); free(oD1); free(oD2); free(oI);
    }
    free(A);
    free(sX); free(sY); free(smo); free(bV); free(bH); free(bD1); free(bD2); free(bI);

    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    for (int i = 0; i < N; i++) fputc(out[i].s, g);
    for (int i = 0; i < N; i++) {
        int32_t e = out[i].e;
        if (fwrite(&e, sizeof e, 1, g) != 1) die("write e");
    }
    for (int i = 0; i < N; i++) fputc(out[i].z, g);
    if (fwrite(bucket, 1, (size_t)N, g) != (size_t)N) die("write bucket");
    fclose(g);
    free(out); free(bucket);
    printf("splat_exchange: %dx%d done\n", H, W);
    return 0;
}
