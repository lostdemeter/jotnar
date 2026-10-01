# Minimal causal attention (mechanism gate, S=4 D=8).
#
# attn_mini + causal structure: lower-triangular bool mask (incl. diagonal,
# host-supplied as int payload like pos metadata) selects allowed scores;
# blocked cells take CONFIG beta negative (-30.0 -> e^-30 ~= 0) via BETA,
# then TSHIFT+SOFTMAX_WIDE. No new mnemonics -- SELECT + BETA composition.
# Contract: scores <=1.0 abs (same T doctrine as attn_mini).
CONFIG beta -30.0
IN Q
IN K
IN V
IN cmask

KT = TRANSPOSE(K)
SCORES = BATCH_MATMUL(Q, KT)
NEG = BETA(SCORES)
MSCORES = SELECT(cmask, SCORES, NEG)
S = TSHIFT(MSCORES)
P = SOFTMAX_WIDE(S)
OUT = BATCH_MATMUL(P, V)
