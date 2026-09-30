/* matmul_exchange: file-exchange driver for bit-exact parity (v1.1 gate 2).
 *
 * Reads A/B triples + m_acc written by test_matmul_c.py, runs holo_matmul,
 * writes output triples. Planes are separate arrays (no struct padding).
 *   in:  int32 Nb,N,K,M,NbB,m_acc | int8 A_s | int32 A_e | uint8 A_z |
 *        int8 B_s | int32 B_e | uint8 B_z
 *   out: int8 out_s | int32 out_e | uint8 out_z
 * Usage: matmul_exchange in.bin out.bin
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "holo_matmul.h"
#include "phi_types.h"

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static void read_plane(FILE *f, int8_t *s, int32_t *e, uint8_t *z, size_t n,
                       const char *tag) {
    if (fread(s, 1, n, f) != n) die(tag);
    if (fread(e, sizeof(int32_t), n, f) != n) die(tag);
    if (fread(z, 1, n, f) != n) die(tag);
}

int main(int argc, char **argv) {
    if (argc != 3) die("usage: matmul_exchange in.bin out.bin");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("open in");
    int32_t hd[6];
    if (fread(hd, sizeof(hd), 1, f) != 1) die("read header");
    int Nb = hd[0], N = hd[1], K = hd[2], M = hd[3], NbB = hd[4],
        m_acc = hd[5];
    if (Nb < 1 || N < 1 || K < 1 || M < 1 || (NbB != 1 && NbB != Nb)) die("bad header");
    size_t na = (size_t)Nb * N * K, nb = (size_t)NbB * K * M,
           no = (size_t)Nb * N * M;
    int8_t *as = malloc(na), *bs = malloc(nb);
    int32_t *ae = malloc(sizeof(int32_t) * na), *be = malloc(sizeof(int32_t) * nb);
    uint8_t *az = malloc(na), *bz = malloc(nb);
    trip_t *A = malloc(sizeof(trip_t) * na), *B = malloc(sizeof(trip_t) * nb),
           *out = malloc(sizeof(trip_t) * no);
    if (!as || !bs || !ae || !be || !az || !bz || !A || !B || !out) die("oom");
    read_plane(f, as, ae, az, na, "read A");
    read_plane(f, bs, be, bz, nb, "read B");
    fclose(f);
    for (size_t i = 0; i < na; i++) { A[i].s = as[i]; A[i].e = ae[i]; A[i].z = az[i]; }
    for (size_t i = 0; i < nb; i++) { B[i].s = bs[i]; B[i].e = be[i]; B[i].z = bz[i]; }
    free(as); free(ae); free(az); free(bs); free(be); free(bz);

    holo_matmul(A, B, Nb, N, K, M, NbB, m_acc, out);

    FILE *g = fopen(argv[2], "wb");
    if (!g) die("open out");
    int8_t *os = malloc(no);
    int32_t *oe = malloc(sizeof(int32_t) * no);
    uint8_t *oz = malloc(no);
    if (!os || !oe || !oz) die("oom");
    for (size_t i = 0; i < no; i++) { os[i] = out[i].s; oe[i] = out[i].e; oz[i] = out[i].z; }
    if (fwrite(os, 1, no, g) != no) die("write out_s");
    if (fwrite(oe, sizeof(int32_t), no, g) != no) die("write out_e");
    if (fwrite(oz, 1, no, g) != no) die("write out_z");
    fclose(g);
    free(A); free(B); free(out); free(os); free(oe); free(oz);
    printf("matmul_exchange: Nb=%d (%dx%d)x(%dx%d) bcast=%d m_acc=%d done\n",
           Nb, N, K, K, M, NbB == 1, m_acc);
    return 0;
}
