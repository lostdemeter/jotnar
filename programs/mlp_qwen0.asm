# MLP block, single-head attention-free (v1.3 real-weights listing).
#
# Qwen2-0.5B layer-0 MLP (gate/up/down_proj + post_attention_layernorm)
# with H supplied as input: H is the real residual stream (embeddings +
# full float64 attention incl. biases, GQA, RoPE-1e6, /8 folding, causal
# mask) computed at the boundary. Attention lives outside the lattice ON
# PURPOSE: real folded scores hit 963 (measured), 1000x past the softmax
# contract -- the T-transform backlog, not a silent fudge. MLP magnitudes
# (UP 2.8, GATE 6.1, DOWN 3.4) fit under CONFIG overrides (proven path).
# eps_rms_c 68719 = Qwen rms_norm_eps 1e-6 in 2^-36 counts (per-model eps
# calibration, first instance of the GAPS backlog item, documented here).
CONFIG m_acc 35492
CONFIG m_cov 35492
CONFIG eps_rms_c 68719

IN H
IN wup
IN wgate
IN wdown
IN ln

HN = RMSNORM(H, ln)
UP = MATMUL(HN, wup)
GATE = MATMUL(HN, wgate)
GS = SILU(GATE)
MID = MUL(GS, UP)
DOWN = MATMUL(MID, wdown)
OUT = ADD(H, DOWN)
