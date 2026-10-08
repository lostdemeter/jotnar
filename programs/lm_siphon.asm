# LM depth-2 weight-tied (2x H=2 blocks, S<=8, V=513 frozen).
#
# First depth run (compounding decider): layer 1 from toks via GATHER,
# layer 2 from H2 directly (no gather), SAME weights both layers
# (tied -- isolates compounding from cross-regime effects; distinct
# L0/L1 banks are the follow-up). Per-head Dh=8, WIDE throughout.
CONFIG eps_rms 1e-6
CONFIG beta -30.0
IMPORT "mlp.asm"
IMPORT "siphon.asm"

IN tok
IN pos
IN cmask
IN emb
IN wq
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2
IN wlog
IN ukt
IN evb
IN wlogU
IN Uhn
IN Ue
IN Vc
IN onesS1
IN ones1V
IN onesSV

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
BC1 = MATMUL(HN, ukt)
BS1 = TSHIFT(BC1)
BP1 = SOFTMAX_WIDE(BS1)
BDOWN = MATMUL(BP1, evb)
H2 = ADD(H, BDOWN)
XN2 = RMSNORM(H2, rms_w1)
Q2A = MATMUL(XN2, wq)
K2A = MATMUL(XN2, wk)
V2A = MATMUL(XN2, wv)
Q12 = SLICE(Q2A, 1, 0, 8)
Q22 = SLICE(Q2A, 1, 8, 16)
K12 = SLICE(K2A, 1, 0, 8)
K22 = SLICE(K2A, 1, 8, 16)
V12 = SLICE(V2A, 1, 0, 8)
V22 = SLICE(V2A, 1, 8, 16)
QR12 = ROTARY(Q12, pos)
QR22 = ROTARY(Q22, pos)
KR12 = ROTARY(K12, pos)
KR22 = ROTARY(K22, pos)
KT12 = TRANSPOSE(KR12)
KT22 = TRANSPOSE(KR22)
SC12 = BATCH_MATMUL(QR12, KT12)
SC22 = BATCH_MATMUL(QR22, KT22)
NEG2 = BETA(SC12)
MS12 = SELECT(cmask, SC12, NEG2)
MS22 = SELECT(cmask, SC22, NEG2)
S12 = TSHIFT(MS12)
S22 = TSHIFT(MS22)
P12 = SOFTMAX_WIDE(S12)
P22 = SOFTMAX_WIDE(S22)
C12 = BATCH_MATMUL(P12, V12)
C22 = BATCH_MATMUL(P22, V22)
CTX2 = CONCAT(C12, C22, 1)
O2 = MATMUL(CTX2, wo)
H3 = ADD(H2, O2)
HN2 = RMSNORM(H3, rms_w2)
DOWN2 = CALL swiglu_block(HN2, wup, wgate, wdown)
H4 = ADD(H3, DOWN2)
LOGITS = MATMUL(H4, wlog)
OUT = ARGMAX(LOGITS, 1)
# Entity-routed install (address early/lexical, content late/linear).
# Bank layout baked: 6 background + Italy (6) + Alex (7) + Caesar (8),
# negs after. E/HN keys per store (dual channel, log-space AND) read
# AT the entity row (dynaddr-style find, position-free); value tiles
# (only the end row is read: post-final-attention doctrine). Blend
# mass = install weights (either install context reads flat).
# All streams IN (assembler data); ones* frozen per geometry.
YB, PR = CALL entity_route(E, HN, Uhn, Ue, Vc, onesS1)
H5 = ADD(H4, YB)
WI = SLICE(PR, 1, 6, 7)
WJ = SLICE(PR, 1, 7, 8)
WK = SLICE(PR, 1, 8, 9)
WIJ = ADD(WI, WJ)
WIJK = ADD(WIJ, WK)
WS = MATMUL(onesS1, WIJK)
W = MATMUL(WS, ones1V)
NW = SUB(onesSV, W)
LOGSKEW = MATMUL(H5, wlog)
LOGFLAT = MATMUL(H5, wlogU)
FW = MUL(W, LOGFLAT)
SW = MUL(NW, LOGSKEW)
LOGITS2 = ADD(FW, SW)
OUT2 = ARGMAX(LOGITS2, 1)

