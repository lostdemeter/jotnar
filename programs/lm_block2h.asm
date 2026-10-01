# LM block H=2 (two-head, single-block, S<=8, V=513 frozen).
#
# Growth from lm_block1.asm (single-head) via repeats: split D=16 into
# 2xDh=8 heads (SLICE), per-head RoPE + TSHIFT+SOFTMAX_WIDE attention,
# CONCAT back to D, then identical residual + SwiGLU + unembedding.
# No new mnemonics, no new machinery -- composition only.
CONFIG eps_rms 1e-6
IMPORT "mlp.asm"

IN tok
IN pos
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
S1 = TSHIFT(SC1)
S2 = TSHIFT(SC2)
P1 = SOFTMAX_WIDE(S1)
P2 = SOFTMAX_WIDE(S2)
C1 = BATCH_MATMUL(P1, V1)
C2 = BATCH_MATMUL(P2, V2)
CTX = CONCAT(C1, C2, 1)
O = MATMUL(CTX, wo)
H = ADD(E, O)
HN = RMSNORM(H, rms_w2)
DOWN = CALL swiglu_block(HN, wup, wgate, wdown)
H2 = ADD(H, DOWN)
LOGITS = MATMUL(H2, wlog)
OUT = ARGMAX(LOGITS, 1)
