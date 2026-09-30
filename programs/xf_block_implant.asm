# xf_block with a directional-store implant (v1.3 store abstraction).
#
# Same block as xf_block.asm through DOWN, then one more store added:
# IMPL = CALL implant_apply(MID, u, v); OUT = ADD(H, ADD(DOWN, IMPL)).
# The MLP tail is spelled out (not CALLed) so MID is a stable top-level
# stream -- probe-structure-outputs doctrine applied to writes. u/v arrive
# as IN streams (host-encoded, like weights); A is folded into u.
# Proof: test_store.py demands parity vs weight-surgery (same values) and
# sham-exactness at A=0 (machinery adds nothing).
CONFIG heads 8

IMPORT "attention.asm"
IMPORT "dirstore.asm"

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
IN u
IN v

XN = RMSNORM(x, rms_w1)
O = CALL attn_core(XN, wq, wk, wv, wo, pos)
H = ADD(x, O)
HN = RMSNORM(H, rms_w2)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = MATMUL(MID, wdown)
IMPL = CALL implant_apply(MID, u, v)
D2 = ADD(DOWN, IMPL)
OUT = ADD(H, D2)
