/* holo_matmul: C lowering of phi-core matmul_int (v1.1 gate 2).
 *
 * Contract (mirrors numpy exactly):
 *   per-product tmul (sign-mul, exp-add clipped to [0,65535], zero-or) +
 *   to_fixed @ m_acc, int64 accumulate (order-free), from_fixed @ m_acc.
 *   B broadcasts when its batch is 1 (NbB==1, Nb arbitrary).
 *   Accumulator bound: |acc| < 2^62 fails loud (mirrors _assert_bound).
 *   m_acc arrives as a PARAMETER (the v1.1 scale path works in C too).
 *
 * Schedule (recorded per IR.md -- values identical under any order because
 * integer sums commute; this records what WE run):
 *   loop order bi -> i -> j -> k, single thread, per-output accumulator.
 *   Row chunking (numpy n_chunk=32): tiling only, same sums, not mirrored.
 *   Tiling: none (v1 scalar). Thread mapping: none.
 */
#pragma once
#include <stdint.h>

#include "phi_types.h"

void holo_matmul(const trip_t *A, const trip_t *B,
                 int Nb, int N, int K, int M, int NbB, int m_acc,
                 trip_t *out);
