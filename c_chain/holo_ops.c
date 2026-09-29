/* C lowering of sqrt_trip/tmul -- bit-exact vs numpy (see test_holo_c.c). */
#include "holo_ops.h"

/* floor((e - BIAS) / 2): numpy // floors; C / truncates toward zero, so
 * negative odd dividends differ by one without this correction. */
static int32_t floor_halve(int32_t e) {
    int64_t d = (int64_t)e - (int64_t)PHI_BIAS;
    int64_t q = d / 2;
    int64_t r = d % 2;
    if (r != 0 && ((r < 0) != (2 < 0))) {
        /* remainder nonzero and signs differ (divisor 2 > 0, so r < 0) */
        q -= 1;
    }
    int64_t out = q + (int64_t)PHI_BIAS;
    if (out < 0) out = 0;
    if (out > 65535) out = 65535;
    return (int32_t)out;
}

void holo_sqrt(const trip_t *in, trip_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i].s = 1;
        out[i].e = floor_halve(in[i].e);
        out[i].z = in[i].z;
    }
}

void holo_mul(const trip_t *a, const trip_t *b, trip_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i].s = (int8_t)((int16_t)a[i].s * (int16_t)b[i].s);
        int64_t e = (int64_t)a[i].e + (int64_t)b[i].e - (int64_t)PHI_BIAS;
        if (e < 0) e = 0;
        if (e > 65535) e = 65535;
        out[i].e = (int32_t)e;
        out[i].z = (uint8_t)(a[i].z | b[i].z);
    }
}

void holo_div(const trip_t *a, const trip_t *b, trip_t *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i].s = (int8_t)((int16_t)a[i].s * (int16_t)b[i].s);
        int64_t e = (int64_t)a[i].e - (int64_t)b[i].e + (int64_t)PHI_BIAS;
        if (e < 0) e = 0;
        if (e > 65535) e = 65535;
        out[i].e = (int32_t)e;
        out[i].z = (uint8_t)(a[i].z | b[i].z);
    }
}

void holo_mux(const trip_t * const *streams, int nstreams,
              const int8_t *bucket, trip_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int b = bucket[i];
        if (b < 0) b = 0;
        if (b >= nstreams) b = nstreams - 1;
        out[i] = streams[b][i];
    }
}
