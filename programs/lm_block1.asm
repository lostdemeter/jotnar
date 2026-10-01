# LM block-one (single-head, single-block, S<=8, V=513 frozen).
#
# Own-data transformer LM first organ beyond bigrams: embeddings (GATHER)
# + RMSNorm + single-head attention (RoPE, TSHIFT+SOFTMAX_WIDE post-merge)
# + residual + SwiGLU MLP (stdlib) + unembedding MATMUL to vocab logits.
# Contract: scores in-contract via small magnitudes (fixture envelope);
# full-range path used regardless (WIDE == legacy bit-exact in-contract).
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
QR = ROTARY(Q, pos)
KR = ROTARY(K, pos)
KT = TRANSPOSE(KR)
SCORES = BATCH_MATMUL(QR, KT)
S = TSHIFT(SCORES)
P = SOFTMAX_WIDE(S)
CTX = BATCH_MATMUL(P, V)
O = MATMUL(CTX, wo)
H = ADD(E, O)
HN = RMSNORM(H, rms_w2)
DOWN = CALL swiglu_block(HN, wup, wgate, wdown)
H2 = ADD(H, DOWN)
LOGITS = MATMUL(H2, wlog)
OUT = ARGMAX(LOGITS, 1)
