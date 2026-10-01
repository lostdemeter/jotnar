# Minimal single-head attention (v1.4 gate 2: selection, S08).
#
# Q,K,V in, scores out through BATCH_MATMUL + SOFTMAX + BATCH_MATMUL.
# No RoPE / mask / scale (those are wrappers around THIS core; the
# decomposition -- output linear in V at fixed P -- holds regardless).
# Contract (stated): scores must stay <=1.0 abs (softmax T-transformation
# doctrine); fixtures keep magnitudes small (Gate-1 envelope pattern).
# Proof: test_select.py demands P-invariance under V masks (precondition),
# exact recomposition over value rows, and top-beats-bottom ordering.
IN Q AS T:SEQ
IN K AS T:SEQ
IN V AS T:SEQ

KT = TRANSPOSE(K)
SCORES = BATCH_MATMUL(Q, KT)
P = SOFTMAX(SCORES)
OUT = BATCH_MATMUL(P, V)
