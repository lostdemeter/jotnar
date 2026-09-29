/* holo_conv: C lowering of chain/holo_phi.py conv_trip (Debt 4, full op).
 *
 * Contract (mirrors numpy exactly):
 *   per-tap tmul (sign-mul, exp-add clipped, zero-or) + to_fixed @ m_acc,
 *   int64 accumulate (order-free), from_fixed @ m_acc, rescale_ to m_out
 *   (fixed@m_acc -> triples -> fixed@m_out -> triples, i.e. fq_rescale),
 *   edge replicate (clamped indices, NOT zero-pad -- differs from phi_conv).
 *
 * Schedule (recorded per IR.md -- values identical under any order because
 * integer sums commute; this records what WE run):
 *   loop order oy -> ox -> ky -> kx, single thread, per-pixel accumulator.
 *   Tiling: none (v1 scalar). Thread mapping: none.
 */
#pragma once
#include <stdint.h>

#include "phi_types.h"

void holo_conv(const trip_t *in, int H, int W, const trip_t *ktap, int KH,
               int m_acc, int m_out, trip_t *out);

/* Fused gathered bank pass (#LIB-006): per output pixel, accumulate taps of
 * ONLY the bucket-selected kernel (bank[5], all KHxKH). Same products and
 * same integer sums as 5x holo_conv + mux -> bit-identical. Schedule:
 * oy -> ox -> ky -> kx, single thread, per-pixel accumulator. */
void holo_bank_fused(const trip_t *in, int H, int W,
                     const trip_t *const *bank, int KH,
                     const int8_t *bucket, int m_acc, int m_out, trip_t *out);
