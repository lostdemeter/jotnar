/* C lowering of conv_trip -- bit-exact vs numpy (file-exchange gated). */
#include <stdlib.h>

#include "holo_conv.h"
#include "bridge.h"
#include "luts.h"

void holo_conv(const trip_t *in, int H, int W, const trip_t *ktap, int KH,
               int m_acc, int m_out, trip_t *out) {
    int r = KH / 2;
    int64_t base = LUT_FRAC[0];
    for (int oy = 0; oy < H; oy++) {
        for (int ox = 0; ox < W; ox++) {
            int64_t acc = 0;
            for (int ky = 0; ky < KH; ky++) {
                int iy = oy + ky - r;
                if (iy < 0) iy = 0;
                if (iy >= H) iy = H - 1;
                for (int kx = 0; kx < KH; kx++) {
                    int ix = ox + kx - r;
                    if (ix < 0) ix = 0;
                    if (ix >= W) ix = W - 1;
                    const trip_t *a = &in[iy * W + ix];
                    const trip_t *w = &ktap[ky * KH + kx];
                    int8_t ps = (int8_t)((int)a->s * (int)w->s);
                    int64_t pe = (int64_t)a->e + (int64_t)w->e - PHI_BIAS;
                    if (pe < 0) pe = 0;
                    if (pe > 65535) pe = 65535;
                    uint8_t pz = (uint8_t)(a->z | w->z);
                    int64_t q;
                    if (pz) {
                        q = 0;
                    } else {
                        int64_t d = (int64_t)m_acc - pe;
                        q = d < 0 ? (int64_t)ps * base
                            : (d > FRAC_CAP ? 0 : (int64_t)ps * LUT_FRAC[d]);
                    }
                    acc += q;
                }
            }
            /* rescale_: fixed@m_acc -> triples -> fixed@m_out -> triples */
            if (m_acc == m_out) {
                fq_t f = { &acc, 1, m_acc };
                triples_from_fixed(&f, &out[oy * W + ox]);
            } else {
                fq_t fa = { &acc, 1, m_acc };
                trip_t mid;
                triples_from_fixed(&fa, &mid);
                int64_t q2;
                if (mid.z) {
                    q2 = 0;
                } else {
                    int64_t d = (int64_t)m_out - (int64_t)mid.e;
                    q2 = d < 0 ? (int64_t)mid.s * base
                        : (d > FRAC_CAP ? 0 : (int64_t)mid.s * LUT_FRAC[d]);
                }
                fq_t fb = { &q2, 1, m_out };
                triples_from_fixed(&fb, &out[oy * W + ox]);
            }
        }
    }
}

/* Fused gathered bank pass (#LIB-006): per output pixel, accumulate taps of
 * ONLY the bucket-selected kernel. Same products and sums as 5x holo_conv +
 * mux -> bit-identical (integer addition commutes). */
void holo_bank_fused(const trip_t *in, int H, int W,
                     const trip_t *const *bank, int KH,
                     const int8_t *bucket, int m_acc, int m_out, trip_t *out) {
    int r = KH / 2;
    int64_t base = LUT_FRAC[0];
    for (int oy = 0; oy < H; oy++) {
        for (int ox = 0; ox < W; ox++) {
            int b = bucket[oy * W + ox];
            if (b < 0) b = 0;
            if (b > 4) b = 4;
            const trip_t *ktap = bank[b];
            int64_t acc = 0;
            for (int ky = 0; ky < KH; ky++) {
                int iy = oy + ky - r;
                if (iy < 0) iy = 0;
                if (iy >= H) iy = H - 1;
                for (int kx = 0; kx < KH; kx++) {
                    int ix = ox + kx - r;
                    if (ix < 0) ix = 0;
                    if (ix >= W) ix = W - 1;
                    const trip_t *a = &in[iy * W + ix];
                    const trip_t *w = &ktap[ky * KH + kx];
                    int8_t ps = (int8_t)((int)a->s * (int)w->s);
                    int64_t pe = (int64_t)a->e + (int64_t)w->e - PHI_BIAS;
                    if (pe < 0) pe = 0;
                    if (pe > 65535) pe = 65535;
                    uint8_t pz = (uint8_t)(a->z | w->z);
                    int64_t q;
                    if (pz) q = 0;
                    else {
                        int64_t d = (int64_t)m_acc - pe;
                        q = d < 0 ? (int64_t)ps * base
                            : (d > FRAC_CAP ? 0 : (int64_t)ps * LUT_FRAC[d]);
                    }
                    acc += q;
                }
            }
            fq_t fa = { &acc, 1, m_acc };
            trip_t mid;
            triples_from_fixed(&fa, &mid);
            int64_t q2;
            if (mid.z) q2 = 0;
            else {
                int64_t d = (int64_t)m_out - (int64_t)mid.e;
                q2 = d < 0 ? (int64_t)mid.s * base
                    : (d > FRAC_CAP ? 0 : (int64_t)mid.s * LUT_FRAC[d]);
            }
            fq_t fb = { &q2, 1, m_out };
            triples_from_fixed(&fb, &out[oy * W + ox]);
        }
    }
}
