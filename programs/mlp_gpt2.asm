# MLP block, GPT-2 layer 0 (v1.6 anatomy instance 2).
#
# Plain MLP (NO SwiGLU gate -- the architectural difference that matters):
# LayerNorm+affine -> c_fc+bias -> GELU -> c_proj+bias -> residual. H is the
# real residual stream (embeddings + learned-pos + full float64 MHA
# boundary, 12 heads, /8 folding, causal mask). GPT-2 differs by VALUES
# more than shape: biases EVERYWHERE (tiled host ADDs, Q-bias precedent),
# LN not RMSNorm, gelu_new (tanh approx -- ours is exact-erf, divergence
# 4.7e-4 stated, not hidden), weights to 6.1 (m_of(64) coverage).
# Scores hit 7.0 folded -- attention stays boundary float (contract).
# Proof: test_anatomy.py demands parity vs torch (bar 40dB).
CONFIG m_acc 37705
CONFIG m_cov 37705
CONFIG eps_ln 1e-5

IN H
IN wfc
IN bfc
IN wpr
IN bpr
IN lnw
IN lnb

HN = LAYERNORM(H, lnw, lnb)
FC = MATMUL(HN, wfc)
FB = ADD(FC, bfc)
GS = GELU(FB)
DOWN = MATMUL(GS, wpr)
DB = ADD(DOWN, bpr)
OUT = ADD(H, DB)
