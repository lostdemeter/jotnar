/* holo_ops: C lowering of the load-bearing triple ops (Debt 4).
 *
 * Covers sqrt (exponent halve, floor semantics), tmul (sign-XOR + exp-add,
 * zero-or), tdiv_pure (sign-XOR + exp-sub, zero-or, NO guards -- callers add
 * zero semantics explicitly, mirroring chain/holo_phi.py tdiv_pure), and
 * tmux (exact select among N triple-streams by an int8 bucket mask,
 * mirroring select_mux, #LIB-009). No LUT, no FPU, no branches on values.
 * Floor-div edge: numpy // floors, C / truncates (see floor_halve).
 */
#pragma once
#include <stdint.h>

#include "phi_types.h"

void holo_sqrt(const trip_t *in, trip_t *out, int n);
void holo_mul(const trip_t *a, const trip_t *b, trip_t *out, int n);
void holo_div(const trip_t *a, const trip_t *b, trip_t *out, int n);
void holo_mux(const trip_t * const *streams, int nstreams,
              const int8_t *bucket, trip_t *out, int n);
/* IR sigmoid (EXPACT gather + LUT, asymptotes exact). Needs the SIG/SIGX
 * tables: link with bridge.c (luts.h) like the exchange drivers. */
void holo_sigmoid(const trip_t *in, trip_t *out, int n);
