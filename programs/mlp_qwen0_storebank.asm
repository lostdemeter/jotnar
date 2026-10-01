# mlp_qwen0 with DOWN as a storebank (v1.3 native storage demo).
#
# Same as mlp_qwen0.asm with DOWN = CALL storebank_apply(MID, Ub, Vb).
# Ub/Vb built host-side via chain/engram.py bank() from frozen stores
# (gains folded into Ub). Proof: test_storebank.py demands parity vs the
# MATMUL form and lands the pruned bank inside its predicted cost.
CONFIG m_acc 35492
CONFIG m_cov 35492
CONFIG eps_rms 1e-6

IMPORT "storebank.asm"

IN H
IN wup
IN wgate
IN ln
IN Ub
IN Vb

HN = RMSNORM(H, ln)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = CALL storebank_apply(MID, Ub, Vb)
OUT = ADD(H, DOWN)
