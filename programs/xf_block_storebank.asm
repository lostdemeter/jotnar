# xf_block with DOWN as a storebank (v1.3 native storage demo).
#
# Same block through MID, then DOWN = CALL storebank_apply(MID, Ub, Vb)
# instead of MATMUL(MID, wdown). Ub/Vb arrive as IN streams (gains folded
# into Ub, built host-side via chain/engram.py bank()). Proof:
# test_store.py demands parity vs the MATMUL form (same values, one extra
# materialization) and the pruned-bank variant lands inside its predicted
# cost (assembler-side edit with numbers).
CONFIG heads 8

IMPORT "attention.asm"
IMPORT "dirstore.asm"
IMPORT "storebank.asm"

IN x AS T:SEQ
IN pos AS I:SEQ
IN wq AS T:SEQ
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2
IN Ub
IN Vb

XN = RMSNORM(x, rms_w1)
O = CALL attn_core(XN, wq, wk, wv, wo, pos)
H = ADD(x, O)
HN = RMSNORM(H, rms_w2)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = CALL storebank_apply(MID, Ub, Vb)
OUT = ADD(H, DOWN)
