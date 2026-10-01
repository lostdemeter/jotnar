# SmolLM2 L0 MLP, stage A (v1.6 anatomy): H -> MID at tight scales.
#
# m 35492 covers H (+-2.6) with healthy count sizes (47dB RMSNorm); the
# bigger m DOWN needs would STARVE these counts (28.9dB -- measured).
# MID triples (scale-free) hand to stage B (S16 triple-direct discipline).
CONFIG m_acc 35492
CONFIG m_cov 35492
CONFIG eps_rms 1e-5

IN H
IN wup
IN wgate
IN ln

HN = RMSNORM(H, ln)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
