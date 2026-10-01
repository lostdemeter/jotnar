# LM layer step (per-block scales): one tied layer h -> h2.
#
# SCAN-style driver pattern: host loops this listing over layers with
# per-layer CONFIG (m_acc/m_cov priced per layer magnitudes -- the
# per-block M-dict law). Same weights every call (tied); only CONFIG
# varies. Composition proof in test_lm_step.py (driver == monolithic).
CONFIG eps_rms 1e-6
CONFIG beta -30.0

IN h
IN pos
IN cmask
IN wq
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2
IN ukt
IN evb
XN = RMSNORM(h, rms_w1)
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
H = ADD(h, O)
HN = RMSNORM(H, rms_w2)
BC1 = MATMUL(HN, ukt)
BS1 = TSHIFT(BC1)
BP1 = SOFTMAX_WIDE(BS1)
BDOWN = MATMUL(BP1, evb)
h2 = ADD(H, BDOWN)
