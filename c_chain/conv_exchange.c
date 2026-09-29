/* conv_exchange: file-exchange driver for bit-exact parity (Debt 4).
 *
 * Reads triples + kernel + scales written by test_c.py, runs holo_conv,
 * writes output triples. Planes are separate arrays (no struct padding).
 *   in:  int32 H,W,KH,m_acc,m_out | int8 in_s | int32 in_e | uint8 in_z |
 *        int8 k_s | int32 k_e | uint8 k_z
 *   out: int8 out_s | int32 out_e | uint8 out_z
 * Usage: conv_exchange in.bin out.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "holo_conv.h"
#include "phi_types.h"

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

int main(int argc, char **argv) {
    if (argc != 3) die("usage: conv_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[5];
    if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
    int H = hd[0], W = hd[1], KH = hd[2], m_acc = hd[3], m_out = hd[4];
    int N = H * W, K = KH * KH;
    trip_t *in = malloc(sizeof(trip_t) * (size_t)N);
    trip_t *kt = malloc(sizeof(trip_t) * (size_t)K);
    trip_t *out = malloc(sizeof(trip_t) * (size_t)N);
    int8_t *bs = malloc((size_t)N);
    int32_t *be = malloc(sizeof(int32_t) * (size_t)N);
    uint8_t *bz = malloc((size_t)N);
    if (!in || !kt || !out || !bs || !be || !bz) die("oom");
    if (fread(bs, 1, (size_t)N, f) != (size_t)N) die("read in_s");
    if (fread(be, sizeof(int32_t), (size_t)N, f) != (size_t)N) die("read in_e");
    if (fread(bz, 1, (size_t)N, f) != (size_t)N) die("read in_z");
    for (int i = 0; i < N; i++) { in[i].s = bs[i]; in[i].e = be[i]; in[i].z = bz[i]; }
    free(bs); free(be); free(bz);
    int8_t *ks = malloc((size_t)K);
    int32_t *ke = malloc(sizeof(int32_t) * (size_t)K);
    uint8_t *kz = malloc((size_t)K);
    if (!ks || !ke || !kz) die("oom");
    if (fread(ks, 1, (size_t)K, f) != (size_t)K) die("read k_s");
    if (fread(ke, sizeof(int32_t), (size_t)K, f) != (size_t)K) die("read k_e");
    if (fread(kz, 1, (size_t)K, f) != (size_t)K) die("read k_z");
    fclose(f);
    for (int i = 0; i < K; i++) { kt[i].s = ks[i]; kt[i].e = ke[i]; kt[i].z = kz[i]; }
    free(ks); free(ke); free(kz);

    holo_conv(in, H, W, kt, KH, m_acc, m_out, out);

    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    int8_t *os = malloc((size_t)N);
    int32_t *oe = malloc(sizeof(int32_t) * (size_t)N);
    uint8_t *oz = malloc((size_t)N);
    if (!os || !oe || !oz) die("oom");
    for (int i = 0; i < N; i++) { os[i] = out[i].s; oe[i] = out[i].e; oz[i] = out[i].z; }
    if (fwrite(os, 1, (size_t)N, g) != (size_t)N) die("write out_s");
    if (fwrite(oe, sizeof(int32_t), (size_t)N, g) != (size_t)N) die("write out_e");
    if (fwrite(oz, 1, (size_t)N, g) != (size_t)N) die("write out_z");
    fclose(g);
    free(in); free(kt); free(out); free(os); free(oe); free(oz);
    printf("conv_exchange: %dx%d KH=%d m_acc=%d m_out=%d done\n", H, W, KH, m_acc, m_out);
    return 0;
}
