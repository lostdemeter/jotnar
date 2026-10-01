# mlp_qwen0 with a directional-store implant (v1.3 store abstraction).
#
# Same as mlp_qwen0.asm plus IMPL = CALL implant_apply(MID, u, v) added
# into DOWN before the residual. u (Dff,1), v (1,D) as IN streams, A
# folded into u host-side. Proof: test_implant.py demands parity vs the
# weight-surgery implant on real weights (same values, both forms).
CONFIG m_acc 35492
CONFIG m_cov 35492
CONFIG eps_rms 1e-6

IMPORT "dirstore.asm"

IN H
IN wup
IN wgate
IN wdown
IN ln
IN u
IN v

HN = RMSNORM(H, ln)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = MATMUL(MID, wdown)
IMPL = CALL implant_apply(MID, u, v)
D2 = ADD(DOWN, IMPL)
OUT = ADD(H, D2)
