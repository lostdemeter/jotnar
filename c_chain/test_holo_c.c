/* test_holo_c: bit-exact gate for the C lowering (Debt 4).
 *
 * Shared vector table with test_core.py::test_c_vectors (same inputs, same
 * expected outputs, documented here and there). Any divergence between the
 * numpy reference and this lowering fails one of the two sides loudly.
 * Covers: sqrt exact-even, sqrt floor-odd-negative (the numpy-//-vs-C-/
 * edge), sqrt zero-flag, mul signs/zero-or, mul exponent clip both ends.
 */
#include <stdio.h>

#include "holo_ops.h"
#include "phi_types.h"

static int fails = 0;

#define CHECK(tag, cond) do { \
    printf("%-22s: %s\n", tag, (cond) ? "OK" : "FAIL"); \
    if (!(cond)) fails++; \
} while (0)

int main(void) {
    /* sqrt vectors: {in_e, in_z, exp_e, exp_z} (sign forced +1) */
    struct { int32_t e; uint8_t z; int32_t ee; uint8_t ez; } sv[] = {
        { 32768, 0, 32768, 0 },   /* d=0 -> 0 */
        { 33792, 0, 33280, 0 },   /* d=+1024 -> +512 */
        { 32765, 0, 32766, 0 },   /* d=-3 -> floor -2 (C / gives -1: the edge) */
        { 32764, 0, 32766, 0 },   /* d=-4 -> -2 exact */
        { 0,     0, 16384, 0 },   /* d=-32768 -> -16384 */
        { 33000, 1, 32884, 1 },   /* d=+232 -> +116, zero flag preserved */
    };
    for (unsigned i = 0; i < sizeof(sv) / sizeof(sv[0]); i++) {
        trip_t in = { -1, sv[i].e, sv[i].z };  /* sign -1 in: must force +1 */
        trip_t out = { 0, 0, 0 };
        holo_sqrt(&in, &out, 1);
        char tag[64];
        snprintf(tag, sizeof(tag), "sqrt[%u]", i);
        CHECK(tag, out.s == 1 && out.e == sv[i].ee && out.z == sv[i].ez);
        if (!(out.s == 1 && out.e == sv[i].ee && out.z == sv[i].ez))
            printf("    got s=%d e=%d z=%u want e=%d z=%u\n",
                   out.s, out.e, out.z, sv[i].ee, sv[i].ez);
    }

    /* mul vectors: {a_s,a_e,a_z, b_s,b_e,b_z, exp_s,exp_e,exp_z} */
    struct { int8_t as; int32_t ae; uint8_t az;
             int8_t bs; int32_t be; uint8_t bz;
             int8_t es; int32_t ee; uint8_t ez; } mv[] = {
        { 1, 33280, 0,   1, 33024, 0,   1, 33536, 0 },  /* exps add */
        { -1, 33280, 0, -1, 33024, 0,   1, 33536, 0 },  /* sign XOR */
        { 1, 33280, 0,  -1, 33024, 0,  -1, 33536, 0 },
        { 1, 33280, 1,   1, 33024, 0,   1, 33536, 1 },  /* zero-or */
        { 1, 65535, 0,   1, 65535, 0,   1, 65535, 0 },  /* high clip */
        { 1, 0, 0,       1, 0, 0,       1, 0, 0 },      /* low clip */
    };
    for (unsigned i = 0; i < sizeof(mv) / sizeof(mv[0]); i++) {
        trip_t a = { mv[i].as, mv[i].ae, mv[i].az };
        trip_t b = { mv[i].bs, mv[i].be, mv[i].bz };
        trip_t out = { 0, 0, 0 };
        holo_mul(&a, &b, &out, 1);
        char tag[64];
        snprintf(tag, sizeof(tag), "mul[%u]", i);
        CHECK(tag, out.s == mv[i].es && out.e == mv[i].ee && out.z == mv[i].ez);
        if (!(out.s == mv[i].es && out.e == mv[i].ee && out.z == mv[i].ez))
            printf("    got s=%d e=%d z=%u want s=%d e=%d z=%u\n",
                   out.s, out.e, out.z, mv[i].es, mv[i].ee, mv[i].ez);
    }

    /* div vectors: pure divide, zero-or, clips (no guards -- by design) */
    struct { int8_t as; int32_t ae; uint8_t az;
             int8_t bs; int32_t be; uint8_t bz;
             int8_t es; int32_t ee; uint8_t ez; } dv[] = {
        { 1, 33536, 0,   1, 33024, 0,   1, 33280, 0 },  /* exps subtract */
        { -1, 33536, 0,  1, 33024, 0,  -1, 33280, 0 },  /* sign XOR */
        { 1, 33536, 1,   1, 33024, 0,   1, 33280, 1 },  /* zero-or */
        { 1, 0, 0,       1, 65535, 0,   1, 0, 0 },      /* low clip */
        { 1, 65535, 0,   1, 0, 0,       1, 65535, 0 },  /* high clip */
    };
    for (unsigned i = 0; i < sizeof(dv) / sizeof(dv[0]); i++) {
        trip_t a = { dv[i].as, dv[i].ae, dv[i].az };
        trip_t b = { dv[i].bs, dv[i].be, dv[i].bz };
        trip_t out = { 0, 0, 0 };
        holo_div(&a, &b, &out, 1);
        char tag[64];
        snprintf(tag, sizeof(tag), "div[%u]", i);
        CHECK(tag, out.s == dv[i].es && out.e == dv[i].ee && out.z == dv[i].ez);
    }

    /* mux vectors: 3 streams, buckets 0/1/2 select exactly */
    {
        trip_t s0 = { 1, 33000, 0 }, s1 = { -1, 34000, 0 }, s2 = { 1, 32000, 1 };
        const trip_t *st[3] = { &s0, &s1, &s2 };
        int8_t bk[4] = { 0, 1, 2, 1 };
        trip_t so[1], mo[4];
        for (int i = 0; i < 4; i++) holo_mux(st, 3, &bk[i], so, 1), mo[i] = so[0];
        CHECK("mux[0]", mo[0].s == 1 && mo[0].e == 33000 && mo[0].z == 0);
        CHECK("mux[1]", mo[1].s == -1 && mo[1].e == 34000 && mo[1].z == 0);
        CHECK("mux[2]", mo[2].s == 1 && mo[2].e == 32000 && mo[2].z == 1);
        CHECK("mux[3]", mo[3].s == -1 && mo[3].e == 34000 && mo[3].z == 0);
    }

    /* sigmoid vectors: (s,e,z) inputs produced by numpy S.encode on
     * [-20,-2,-0.5,0,0.5,2,20] (provenance: sigmoid_trip gate run; values
     * 0.11932/0.37733/0.50023/0.62270/0.88084 + rails). C consumes integers
     * only (no libm, trap-clean); expectations exact. */
    struct { int8_t s; int32_t e; uint8_t z;
             int8_t es; int32_t ee; uint8_t ez; } gv[] = {
        { -1, 35955, 0,  -1, 18756, 1 },  /* saturating rail -> zero triple */
        { -1, 33505, 0,   1, 30506, 0 },
        { -1, 32031, 0,   1, 31731, 0 },
        {  1,     0, 0,   1, 32031, 0 },  /* encode(0.0): e=0, not z */
        {  1, 32031, 0,   1, 32264, 0 },
        {  1, 33505, 0,   1, 32633, 0 },
        {  1, 35955, 0,   1, 32768, 0 },  /* saturating rail -> one triple */
    };
    for (unsigned i = 0; i < sizeof(gv) / sizeof(gv[0]); i++) {
        trip_t in = { gv[i].s, gv[i].e, gv[i].z };
        trip_t out = { 0, 0, 0 };
        holo_sigmoid(&in, &out, 1);
        char tag[64];
        snprintf(tag, sizeof(tag), "sig[%u]", i);
        CHECK(tag, out.s == gv[i].es && out.e == gv[i].ee && out.z == gv[i].ez);
        if (!(out.s == gv[i].es && out.e == gv[i].ee && out.z == gv[i].ez))
            printf("    got s=%d e=%d z=%u want s=%d e=%d z=%u\n",
                   out.s, out.e, out.z, gv[i].es, gv[i].ee, gv[i].ez);
    }

    printf("RESULT: %s\n", fails ? "FAIL" : "ALL OK");
    return fails ? 1 : 0;
}