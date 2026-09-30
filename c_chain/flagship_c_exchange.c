/* flagship_c_exchange: full flagship (splat_soft + v5) in C (v1.2 gate 3).
 *
 * Same in.bin/out.bin format as flagship_cuda_exchange (one builder gates
 * both sides, no format fork). Orchestration mirrors the CUDA driver
 * launch-for-launch (composition on the host side, no new math); every op
 * is a proven C lowering (holo_conv.c / holo_ops.c / bridge.c). Small
 * helpers below (neg/wherez/relu/clip/square/rescale/bucket) are static
 * to this file, gated compositionally by flagship bit-exactness -- the
 * same standing as their CUDA twins (which likewise have no C twins).
 * No LUT, no FPU.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "holo_conv.h"
#include "holo_ops.h"
#include "bridge.h"

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static trip_t *talloc(size_t n) {
    trip_t *p = (trip_t *)malloc(sizeof(trip_t) * (n ? n : 1));
    if (!p) die("oom");
    return p;
}

static int64_t *qalloc(size_t n) {
    int64_t *p = (int64_t *)malloc(sizeof(int64_t) * (n ? n : 1));
    if (!p) die("oom");
    return p;
}

static void to_fx(const trip_t *t, int64_t *q, int m, size_t n) {
    fq_t f = { q, (int)n, m };
    to_fixed(t, &f, m);
}

static void from_fx(const int64_t *q, int m, trip_t *out, size_t n) {
    fq_t f = { (int64_t *)q, (int)n, m };
    triples_from_fixed(&f, out);
}

static void binop_arr(const trip_t *a, const trip_t *b, int sub, int m,
                      trip_t *out, size_t n) {
    int64_t *qa = qalloc(n), *qb = qalloc(n);
    to_fx(a, qa, m, n);
    to_fx(b, qb, m, n);
    for (size_t i = 0; i < n; i++) qa[i] += sub ? -qb[i] : qb[i];
    from_fx(qa, m, out, n);
    free(qa);
    free(qb);
}

static void neg_arr(const trip_t *a, trip_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i].s = (int8_t)(-(int)a[i].s);
        out[i].e = a[i].e;
        out[i].z = a[i].z;
    }
}

static void abs_arr(const trip_t *a, trip_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i].s = 1;
        out[i].e = a[i].e;
        out[i].z = a[i].z;
    }
}

static void wherez_arr(const trip_t *f, const trip_t *a, const trip_t *b,
                       trip_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = f[i].z ? b[i] : a[i];
}

static void relu_arr(const trip_t *a, int m, trip_t *out, size_t n) {
    int64_t *q = qalloc(n);
    to_fx(a, q, m, n);
    for (size_t i = 0; i < n; i++) if (q[i] < 0) q[i] = 0;
    from_fx(q, m, out, n);
    free(q);
}

static void clip_arr(const trip_t *a, int m, int64_t qlo, int64_t qhi,
                     trip_t *out, size_t n) {
    int64_t *q = qalloc(n);
    to_fx(a, q, m, n);
    for (size_t i = 0; i < n; i++) {
        if (q[i] < qlo) q[i] = qlo;
        if (q[i] > qhi) q[i] = qhi;
    }
    from_fx(q, m, out, n);
    free(q);
}

static void square_arr(const trip_t *a, int m, int64_t qhi, trip_t *out,
                       size_t n) {
    trip_t *p = talloc(n);
    holo_mul(a, a, p, n);
    clip_arr(p, m, 0, qhi, out, n);
    free(p);
}

static void rescale_arr(const trip_t *a, int m1, int m2, trip_t *out,
                        size_t n) {
    int64_t *q1 = qalloc(n);
    to_fx(a, q1, m1, n);
    trip_t *mid = talloc(n);
    from_fx(q1, m1, mid, n);
    if (m1 == m2) {
        for (size_t i = 0; i < n; i++) out[i] = mid[i];
    } else {
        int64_t *q2 = qalloc(n);
        to_fx(mid, q2, m2, n);
        from_fx(q2, m2, out, n);
        free(q2);
    }
    free(q1);
    free(mid);
}

static void bucket_arr(const trip_t *d, const trip_t *s2, const trip_t *coh,
                       int m, int64_t tq, int8_t *bucket, size_t n) {
    int64_t *qd = qalloc(n), *qs = qalloc(n), *qc = qalloc(n);
    to_fx(d, qd, m, n);
    to_fx(s2, qs, m, n);
    to_fx(coh, qc, m, n);
    for (size_t i = 0; i < n; i++) {
        int64_t ad = qd[i] < 0 ? -qd[i] : qd[i];
        int64_t as = qs[i] < 0 ? -qs[i] : qs[i];
        int diag_dom = as > ad;
        int vert = qd[i] > 0;
        int diag_pos = qs[i] >= 0;
        int dir = !diag_dom ? (vert ? 0 : 1) : (diag_pos ? 3 : 2);
        bucket[i] = (int8_t)((qc[i] >= tq) ? dir : 4);
    }
    free(qd);
    free(qs);
    free(qc);
}

static void read_planar(FILE *f, trip_t *t, size_t n, const char *tag) {
    int8_t *s = (int8_t *)malloc(n ? n : 1);
    int32_t *e = (int32_t *)malloc(sizeof(int32_t) * (n ? n : 1));
    uint8_t *z = (uint8_t *)malloc(n ? n : 1);
    if (!s || !e || !z) die("oom");
    if (fread(s, 1, n, f) != n) die(tag);
    if (fread(e, sizeof(int32_t), n, f) != n) die(tag);
    if (fread(z, 1, n, f) != n) die(tag);
    for (size_t i = 0; i < n; i++) { t[i].s = s[i]; t[i].e = e[i]; t[i].z = z[i]; }
    free(s);
    free(e);
    free(z);
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: flagship_c_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[4];
    if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
    int H = hd[0], W = hd[1], m_acc = hd[2], m_cov = hd[3];
    int64_t tq = 0, qhi1 = 0;
    if (fread(&tq, sizeof(tq), 1, f) != 1) die("read tq");
    if (fread(&qhi1, sizeof(qhi1), 1, f) != 1) die("read qhi1");
    size_t N = (size_t)H * W;
    trip_t *A = talloc(N);
    read_planar(f, A, N, "read A");
    int ksz[8] = {9, 9, 9, 25, 25, 25, 25, 25};
    trip_t *K[8];
    for (int ki = 0; ki < 8; ki++) {
        K[ki] = talloc((size_t)ksz[ki]);
        read_planar(f, K[ki], (size_t)ksz[ki], "read kernel");
    }
    enum { CBETA, CATT, CDM, CDS, CTHR, CHI, CK30, CW0, CW1, CW2, CFOUR, CHALF, CZERO, NCONST };
    trip_t *C[NCONST];
    for (int ci = 0; ci < NCONST; ci++) {
        C[ci] = talloc(N);
        read_planar(f, C[ci], N, "read const");
    }
    fclose(f);
    int KH3 = 3, KH5 = 5;
    /* --- structure tensor --- */
    trip_t *gx = talloc(N), *gy = talloc(N);
    holo_conv(A, H, W, K[0], KH3, m_acc, m_cov, gx);
    holo_conv(A, H, W, K[1], KH3, m_acc, m_cov, gy);
    trip_t *jxx = talloc(N), *jyy = talloc(N), *jxy = talloc(N);
    holo_mul(gx, gx, jxx, N);
    holo_mul(gy, gy, jyy, N);
    holo_mul(gx, gy, jxy, N);
    trip_t *Jxx = talloc(N), *Jyy = talloc(N), *Jxy = talloc(N);
    holo_conv(jxx, H, W, K[2], KH3, m_acc, m_cov, Jxx);
    holo_conv(jyy, H, W, K[2], KH3, m_acc, m_cov, Jyy);
    holo_conv(jxy, H, W, K[2], KH3, m_acc, m_cov, Jxy);
    /* --- coherence --- */
    trip_t *d = talloc(N);
    binop_arr(Jxx, Jyy, 1, m_cov, d, N);
    trip_t *d2 = talloc(N), *xy2 = talloc(N), *t4 = talloc(N);
    holo_mul(d, d, d2, N);
    holo_mul(Jxy, Jxy, xy2, N);
    holo_mul(xy2, C[CFOUR], t4, N);
    trip_t *discA = talloc(N), *disc = talloc(N), *aniso = talloc(N);
    binop_arr(d2, t4, 0, m_acc, discA, N);
    rescale_arr(discA, m_acc, m_cov, disc, N);
    holo_sqrt(disc, aniso, N);
    trip_t *trace = talloc(N), *raw = talloc(N), *coh0 = talloc(N), *coh = talloc(N);
    binop_arr(Jxx, Jyy, 0, m_cov, trace, N);
    holo_div(aniso, trace, raw, N);
    wherez_arr(trace, raw, C[CZERO], coh0, N);
    clip_arr(coh0, m_cov, 0, qhi1, coh, N);
    trip_t *s2 = talloc(N);
    binop_arr(Jxy, Jxy, 0, m_cov, s2, N);
    int8_t *bucket = (int8_t *)malloc(N ? N : 1);
    if (!bucket) die("oom");
    bucket_arr(d, s2, coh, m_cov, tq, bucket, N);
    (void)bucket; /* hard path unused by soft flagship; buckets audited upstream */
    /* --- bank (soft) --- */
    trip_t *outs[5];
    for (int bi = 0; bi < 5; bi++) {
        outs[bi] = talloc(N);
        holo_conv(A, H, W, K[3 + bi], KH5, m_acc, m_cov, outs[bi]);
    }
    trip_t *negd = talloc(N), *negs2 = talloc(N);
    neg_arr(d, negd, N);
    neg_arr(s2, negs2, N);
    trip_t *rV = talloc(N), *rH = talloc(N), *rD1 = talloc(N), *rD2 = talloc(N);
    relu_arr(d, m_cov, rV, N);
    relu_arr(negd, m_cov, rH, N);
    relu_arr(s2, m_cov, rD1, N);
    relu_arr(negs2, m_cov, rD2, N);
    trip_t *tA = talloc(N), *tB = talloc(N), *den = talloc(N);
    binop_arr(rV, rH, 0, m_cov, tA, N);
    binop_arr(rD1, rD2, 0, m_cov, tB, N);
    binop_arr(tA, tB, 0, m_cov, den, N);
    trip_t *wV = talloc(N), *wH = talloc(N), *wD1 = talloc(N), *wD2 = talloc(N);
    holo_div(rV, den, wV, N);
    holo_div(rH, den, wH, N);
    holo_div(rD1, den, wD1, N);
    holo_div(rD2, den, wD2, N);
    trip_t *p0 = talloc(N), *p1 = talloc(N), *q0 = talloc(N);
    trip_t *p3 = talloc(N), *q1 = talloc(N), *p2 = talloc(N), *acc = talloc(N);
    holo_mul(wV, outs[0], p0, N);
    holo_mul(wH, outs[1], p1, N);
    binop_arr(p0, p1, 0, m_cov, q0, N);
    holo_mul(wD1, outs[3], p3, N);
    binop_arr(q0, p3, 0, m_cov, q1, N);
    holo_mul(wD2, outs[2], p2, N);
    binop_arr(q1, p2, 0, m_cov, acc, N);
    trip_t *AS = talloc(N);
    wherez_arr(den, acc, outs[4], AS, N);
    /* --- beta v5 --- */
    trip_t *D = talloc(N);
    binop_arr(A, AS, 1, m_cov, D, N);
    trip_t *c1 = talloc(N), *e1 = talloc(N), *g1 = talloc(N);
    trip_t *c2 = talloc(N), *e2 = talloc(N), *g2 = talloc(N);
    binop_arr(coh, C[CTHR], 1, m_cov, c1, N);
    holo_mul(c1, C[CK30], e1, N);
    holo_sigmoid(e1, g1, N);
    binop_arr(coh, C[CHI], 1, m_cov, c2, N);
    holo_mul(c2, C[CK30], e2, N);
    holo_sigmoid(e2, g2, N);
    trip_t *wA = talloc(N), *w = talloc(N), *base = talloc(N);
    trip_t *td1 = talloc(N), *td2 = talloc(N);
    holo_mul(C[CDM], g1, td1, N);
    binop_arr(C[CATT], td1, 0, m_cov, wA, N);
    holo_mul(C[CDS], g2, td2, N);
    binop_arr(wA, td2, 0, m_cov, w, N);
    holo_mul(C[CBETA], w, base, N);
    trip_t *ad = talloc(N), *x4 = talloc(N), *dhat = talloc(N);
    abs_arr(D, ad, N);
    holo_mul(ad, C[CFOUR], x4, N);
    clip_arr(x4, m_cov, 0, qhi1, dhat, N);
    trip_t *t1 = talloc(N), *t2 = talloc(N), *lt = talloc(N), *logit = talloc(N);
    trip_t *gate = talloc(N), *scale = talloc(N), *BEFF = talloc(N);
    holo_mul(C[CW1], coh, t1, N);
    holo_mul(C[CW2], dhat, t2, N);
    binop_arr(C[CW0], t1, 0, m_cov, lt, N);
    binop_arr(lt, t2, 0, m_cov, logit, N);
    holo_sigmoid(logit, gate, N);
    binop_arr(C[CHALF], gate, 0, m_cov, scale, N);
    holo_mul(base, scale, BEFF, N);
    /* --- finish --- */
    trip_t *BD = talloc(N), *AE = talloc(N), *YENH = talloc(N);
    holo_mul(D, BEFF, BD, N);
    binop_arr(A, BD, 0, m_cov, AE, N);
    square_arr(AE, m_cov, qhi1, YENH, N);
    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    int8_t *os = (int8_t *)malloc(N);
    int32_t *oe = (int32_t *)malloc(sizeof(int32_t) * N);
    uint8_t *oz = (uint8_t *)malloc(N);
    if (!os || !oe || !oz) die("oom");
    for (size_t i = 0; i < N; i++) { os[i] = YENH[i].s; oe[i] = YENH[i].e; oz[i] = YENH[i].z; }
    if (fwrite(os, 1, N, g) != N) die("write s");
    if (fwrite(oe, sizeof(int32_t), N, g) != (size_t)N) die("write e");
    if (fwrite(oz, 1, N, g) != (size_t)N) die("write z");
    fclose(g);
    printf("flagship_c_exchange: %dx%d m_acc=%d m_cov=%d done\n", H, W, m_acc, m_cov);
    return 0;
}
