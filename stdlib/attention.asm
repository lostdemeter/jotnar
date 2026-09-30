# stdlib/attention.asm — shared attention core (single-head, unscaled).
#
# Factored out of programs/xf_block.asm (v1.0 Gate 3: no copy-pasted
# prologues in shipped programs). Any encoder listing CALLs this instead
# of spelling the ten lines. Contract: scores must stay <=1.0 abs (softmax
# T-transformation doctrine — see docs/LANGUAGE.md §4 SOFTMAX).
DEF attn_core(xn, wq, wk, wv, wo, pos) -> (o)
  Q = MATMUL(xn, wq)
  K = MATMUL(xn, wk)
  V = MATMUL(xn, wv)
  QR = ROTARY(Q, pos)
  KR = ROTARY(K, pos)
  KT = TRANSPOSE(KR)
  SCORES = BATCH_MATMUL(QR, KT)
  P = SOFTMAX(SCORES)
  CTX = BATCH_MATMUL(P, V)
  o = MATMUL(CTX, wo)
END
