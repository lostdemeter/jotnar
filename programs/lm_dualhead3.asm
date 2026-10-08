# LM dual-head readout: skewed prior head + flat install head, routed.

# Post-layer-2 bank: Italy store (content) + background null stores
# (addressing). The DECISION blends per row by the bank's own Italy
# weight: install contexts (w~1) read the flat head (unit-norm columns:
# target alignment wins), base contexts (w~0) read the skewed head
# (norm prior intact). No host mask: addressing and head-routing are
# one weight (SLICE Italy column -> tile -> MUL/SUB/ADD blend, all
# existing ops). ones1V/onesSV arrive as IN (frozen geometry, cmask
# class). wlogU arrives as IN (unit columns of wlog).
# Bank layout baked: 6 background stores + Italy (6) + Alex (7),
# negs after. Blend weight W = w_italy + w_alex (one ADD): either
# install context reads flat, everything else skewed.
CONFIG eps_rms 1e-6
CONFIG beta -30.0
IMPORT "mlp.asm"

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
IN wlogU
IN ones1V
IN onesSV
IN ukt
IN evb
IN ukt2
IN evb2

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
HNB = RMSNORM(H4, rms_w2)
BC2 = MATMUL(HNB, ukt2)
BS2 = TSHIFT(BC2)
BP2 = SOFTMAX_WIDE(BS2)
BDOWN2 = MATMUL(BP2, evb2)
H5 = ADD(H4, BDOWN2)
LOGSKEW = MATMUL(H5, wlog)
LOGFLAT = MATMUL(H5, wlogU)
WI = SLICE(BP2, 1, 6, 7)
WJ = SLICE(BP2, 1, 7, 8)
WIJ = ADD(WI, WJ)
W = MATMUL(WIJ, ones1V)
NW = SUB(onesSV, W)
FW = MUL(W, LOGFLAT)
SW = MUL(NW, LOGSKEW)
LOGITS2 = ADD(FW, SW)
OUT2 = ARGMAX(LOGITS2, 1)
