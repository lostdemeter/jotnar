/* holo_elem: CUDA lowerings of the holo elementwise triple ops (v1.2 gate 2b).
 *
 * Integer-only, FPU-free. Each kernel mirrors its proven numpy/C twin
 * (chain/holo_phi.py + c_chain/holo_ops.c, both gated): sqrt (exponent
 * halve, floor semantics), tmul/tdiv (sign-XOR + exp-add/sub, zero-or),
 * binop add/sub @ m (fixed-domain), square+clip [0,1], sigmoid (SIGX+SOS
 * tables, asymptotes exact), abs (sign force), relu (fixed max with 0),
 * clip (fixed clamp between host-computed bounds). No new math anywhere.
 * LUTs as int globals (conv.cu precedent).
 *
 * Schedule: one thread per element; no cross-thread reductions anywhere.
 * Tiling: none (v1 scalar).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cuda_dev.cuh"
#include "luts.h"

#define SIG_SPAN (16 * 16384)

__device__ __forceinline__ int floor_halve_dev(int e) {
    int64_t d = (int64_t)e - (int64_t)PHI_BIAS;
    int64_t q = d / 2;
    int64_t r = d % 2;
    if (r != 0 && r < 0) q -= 1;
    int64_t out = q + (int64_t)PHI_BIAS;
    if (out < 0) out = 0;
    if (out > 65535) out = 65535;
    return (int)out;
}

__global__ void k_sqrt(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    os[i] = 1;
    oe[i] = floor_halve_dev(e[i]);
    oz[i] = z[i];
}

__global__ void k_tmul(
    const int8_t *__restrict__ sa, const int *__restrict__ ea,
    const uint8_t *__restrict__ za,
    const int8_t *__restrict__ sb, const int *__restrict__ eb,
    const uint8_t *__restrict__ zb, int n,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    os[i] = (int8_t)((int16_t)sa[i] * (int16_t)sb[i]);
    int64_t ev = (int64_t)ea[i] + (int64_t)eb[i] - (int64_t)PHI_BIAS;
    if (ev < 0) ev = 0;
    if (ev > 65535) ev = 65535;
    oe[i] = (int)ev;
    oz[i] = (uint8_t)(za[i] | zb[i]);
}

__global__ void k_tdiv(
    const int8_t *__restrict__ sa, const int *__restrict__ ea,
    const uint8_t *__restrict__ za,
    const int8_t *__restrict__ sb, const int *__restrict__ eb,
    const uint8_t *__restrict__ zb, int n,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    os[i] = (int8_t)((int16_t)sa[i] * (int16_t)sb[i]);
    int64_t ev = (int64_t)ea[i] - (int64_t)eb[i] + (int64_t)PHI_BIAS;
    if (ev < 0) ev = 0;
    if (ev > 65535) ev = 65535;
    oe[i] = (int)ev;
    oz[i] = (uint8_t)(za[i] | zb[i]);
}

__global__ void k_binop(
    const int8_t *__restrict__ sa, const int *__restrict__ ea,
    const uint8_t *__restrict__ za,
    const int8_t *__restrict__ sb, const int *__restrict__ eb,
    const uint8_t *__restrict__ zb, int n, int m, int is_sub,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t base = frac[0];
    int64_t q = to_fixed_elem(sa[i], ea[i], za[i], m, base, frac)
        + (is_sub ? -to_fixed_elem(sb[i], eb[i], zb[i], m, base, frac)
                  : to_fixed_elem(sb[i], eb[i], zb[i], m, base, frac));
    int8_t s;
    int ev;
    uint8_t z;
    from_fixed_elem(q, m, coarse, fine, &s, &ev, &z);
    os[i] = s;
    oe[i] = ev;
    oz[i] = z;
}

__global__ void k_square_clip(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n, int m, int64_t qhi,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int8_t ps = (int8_t)((int16_t)s[i] * (int16_t)s[i]);
    int64_t pe = (int64_t)e[i] + (int64_t)e[i] - (int64_t)PHI_BIAS;
    if (pe < 0) pe = 0;
    if (pe > 65535) pe = 65535;
    uint8_t pz = z[i];
    int64_t base = frac[0];
    int64_t q = to_fixed_elem(ps, (int)pe, pz, m, base, frac);
    if (q < 0) q = 0;
    if (q > qhi) q = qhi;
    int8_t so;
    int eo;
    uint8_t zo;
    from_fixed_elem(q, m, coarse, fine, &so, &eo, &zo);
    os[i] = so;
    oe[i] = eo;
    oz[i] = zo;
}

__global__ void k_sigmoid(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n,
    const int64_t *__restrict__ sigx, const int *__restrict__ sig,
    const int *__restrict__ coarse, const int *__restrict__ fine,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ev = e[i];
    if (ev < 0) ev = 0;
    if (ev > 65535) ev = 65535;
    int64_t x14 = z[i] ? 0 : (int64_t)s[i] * sigx[ev];
    int64_t y14;
    if (x14 < -SIG_SPAN) y14 = 0;
    else if (x14 > SIG_SPAN) y14 = 16384;
    else {
        int64_t idx = x14 + SIG_SPAN;
        if (idx < 0) idx = 0;
        if (idx > 524288) idx = 524288;
        y14 = (int64_t)sig[idx];
    }
    int64_t q = y14 * 16;
    int8_t so;
    int eo;
    uint8_t zo;
    from_fixed_elem(q, PHI_BIAS, coarse, fine, &so, &eo, &zo);
    os[i] = so;
    oe[i] = eo;
    oz[i] = zo;
}

__global__ void k_abs(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    os[i] = 1;
    oe[i] = e[i];
    oz[i] = z[i];
}

__global__ void k_relu(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n, int m,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t base = frac[0];
    int64_t q = to_fixed_elem(s[i], e[i], z[i], m, base, frac);
    if (q < 0) q = 0;
    int8_t so;
    int eo;
    uint8_t zo;
    from_fixed_elem(q, m, coarse, fine, &so, &eo, &zo);
    os[i] = so;
    oe[i] = eo;
    oz[i] = zo;
}

__global__ void k_clip(
    const int8_t *__restrict__ s, const int *__restrict__ e,
    const uint8_t *__restrict__ z, int n, int m, int64_t qlo, int64_t qhi,
    const int *__restrict__ frac, const int *__restrict__ coarse,
    const int *__restrict__ fine,
    int8_t *__restrict__ os, int *__restrict__ oe, uint8_t *__restrict__ oz) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t base = frac[0];
    int64_t q = to_fixed_elem(s[i], e[i], z[i], m, base, frac);
    if (q < qlo) q = qlo;
    if (q > qhi) q = qhi;
    int8_t so;
    int eo;
    uint8_t zo;
    from_fixed_elem(q, m, coarse, fine, &so, &eo, &zo);
    os[i] = so;
    oe[i] = eo;
    oz[i] = zo;
}
