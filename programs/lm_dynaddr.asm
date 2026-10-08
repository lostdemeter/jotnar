# LM depth-2 weight-tied (2x H=2 blocks, S<=8, V=513 frozen).
#
# First depth run (compounding decider): layer 1 from toks via GATHER,
# layer 2 from H2 directly (no gather), SAME weights both layers
# (tied -- isolates compounding from cross-regime effects; distinct
# L0/L1 banks are the follow-up). Per-head Dh=8, WIDE throughout.
CONFIG eps_rms 1e-6
CONFIG beta -30.0
IMPORT "mlp.asm"
IMPORT "dynaddr.asm"

IN tok
IN pos
IN cmask
IN emb
IN wq
IN wk
IN wv
IN wo
IN rms_w1
IN rms_w2
IN ukt
IN ckey

E = GATHER(emb, tok)
XN = RMSNORM(E, rms_w1)
Q = MATMUL(XN, wq)
K = MATMUL(XN, wk)
V = MATMUL(XN, wv)
Q1 = SLICE(Q, 1, 0, 8)
Q2 = SLICE(Q, 1, 8, 16)
K1 = SLICE(K, 1, 0, 8)
K2 = SLICE(K, 1, 8, 16)
V1 = SLICE(V, 1, 0, 8)
V2 = SLICE(V, 1, 8, 16)
QR1 = ROTARY(Q1, pos)
QR2 = ROTARY(Q2, pos)
KR1 = ROTARY(K1, pos)
KR2 = ROTARY(K2, pos)
KT1 = TRANSPOSE(KR1)
KT2 = TRANSPOSE(KR2)
SC1 = BATCH_MATMUL(QR1, KT1)
SC2 = BATCH_MATMUL(QR2, KT2)
NEG1 = BETA(SC1)
MS1 = SELECT(cmask, SC1, NEG1)
MS2 = SELECT(cmask, SC2, NEG1)
S1 = TSHIFT(MS1)
S2 = TSHIFT(MS2)
P1 = SOFTMAX_WIDE(S1)
P2 = SOFTMAX_WIDE(S2)
C1 = BATCH_MATMUL(P1, V1)
C2 = BATCH_MATMUL(P2, V2)
CTX = CONCAT(C1, C2, 1)
O = MATMUL(CTX, wo)
H = ADD(E, O)
HN = RMSNORM(H, rms_w2)
XA = CALL dynaddr_match(HN, ckey)
BC = MATMUL(XA, ukt)
BS = TSHIFT(BC)
BP = SOFTMAX_WIDE(BS)
RET = ARGMAX(BP, 1)
