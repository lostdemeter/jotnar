/* C lowering of phi-core matmul_int -- bit-exact vs numpy (file-exchange
 * gated). Integer-only, FPU-free. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "holo_matmul.h"
#include "bridge.h"
#include "luts.h"

#define ACC_BOUND ((int64_t)1 << 62)

void holo_matmul(const trip_t *A, const trip_t *B,
                 int Nb, int N, int K, int M, int NbB, int m_acc,
                 trip_t *out) {
    int64_t base = LUT_FRAC[0];
    for (int bi = 0; bi < Nb; bi++) {
        const trip_t *Bb = (NbB == 1) ? B : &B[(size_t)bi * K * M];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                int64_t acc = 0;
                for (int k = 0; k < K; k++) {
                    const trip_t *a = &A[((size_t)bi * N + i) * K + k];
                    const trip_t *b = &Bb[(size_t)k * M + j];
                    int8_t ps = (int8_t)((int)a->s * (int)b->s);
                    int64_t pe = (int64_t)a->e + (int64_t)b->e - PHI_BIAS;
                    if (pe < 0) pe = 0;
                    if (pe > 65535) pe = 65535;
                    if (!(a->z | b->z)) {
                        int64_t d = (int64_t)m_acc - pe;
                        acc += d < 0 ? (int64_t)ps * base
                             : (d > FRAC_CAP ? 0 : (int64_t)ps * LUT_FRAC[d]);
                    }
                }
                if (acc >= ACC_BOUND || acc <= -ACC_BOUND) {
                    fprintf(stderr, "holo_matmul: acc bound 2^62 hit\n");
                    exit(1);
                }
                fq_t f = { &acc, 1, m_acc };
                triples_from_fixed(&f, &out[((size_t)bi * N + i) * M + j]);
            }
        }
    }
}
