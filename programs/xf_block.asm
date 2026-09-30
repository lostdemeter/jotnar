# Transformer encoder block (Qwen-style), as assembly -- GAP PROBE, NOT a program.
#
# Purpose: enumerate what the language CANNOT yet say by attempting to say
# it. The assembler fails loud on the first unknown mnemonic; iterating
# (comment out, re-run) enumerates the full gap set mechanically. Recorded
# in docs/GAPS.md. This file must FAIL to assemble until the gaps close --
# that failure is the instrument reading, do NOT "fix" the listing.
CONFIG heads 8

IN x
IN pos
IN wq
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2

# --- attention path ---
XN = RMSNORM(x, rms_w1)
Q = MATMUL(XN, wq)
K = MATMUL(XN, wk)
V = MATMUL(XN, wv)
QR = ROTARY(Q, pos)
KR = ROTARY(K, pos)
KT = TRANSPOSE(KR)
SCORES = BATCH_MATMUL(QR, KT)
P = SOFTMAX(SCORES)
CTX = BATCH_MATMUL(P, V)
O = MATMUL(CTX, wo)
H = ADD(x, O)

# --- MLP path (SwiGLU) ---
HN = RMSNORM(H, rms_w2)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = MATMUL(MID, wdown)
OUT = ADD(H, DOWN)
