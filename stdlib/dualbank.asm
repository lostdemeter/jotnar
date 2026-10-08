# stdlib/dualbank.asm — two-channel bank (log-space AND).
#
# One match score per store cannot express conjunctions (scales slide
# boundaries; dynaddr ladder proved it 0.5->4.0 with no sweet spot).
# Fuse BEFORE the softmax, in log space: C = HN-match + E-match, so a
# row must match BOTH channels to win. Contextual channel (HN: shared
# context geometry, spills same-class) AND lexical channel (E:
# RoPE-free, position-free embeddings, distinct per token).
# Same stores, two key spaces; V shared. All existing ops
# (MATMUL/ADD/TSHIFT/SOFTMAX_WIDE). Returns (y, P): y the down values,
# P the routing (read P for retrieval receipts).
DEF dualbank_apply(xh, xe, Uh, Ue, V) -> (y, P)
  Ch = MATMUL(xh, Uh)
  Ce = MATMUL(xe, Ue)
  C = ADD(Ch, Ce)
  S = TSHIFT(C)
  P = SOFTMAX_WIDE(S)
  y = MATMUL(P, V)
END
