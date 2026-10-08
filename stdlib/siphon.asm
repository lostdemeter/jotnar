# stdlib/siphon.asm — entity-routed install (address early, content late).
#
# Late-state (HN/HNB) keys match TEMPLATE more than entity ("the city
# of Y is" fires at 1.0 for either Y -- breadth log). Lexical (E)
# keys match WORDS, not syntax -- but the entity row is not the read
# row. This routes explicitly: find the entity row by content
# (E-space match, position-free), read the DUAL-channel bank scores
# (HN contextual + E lexical, log-space AND) AT that row, softmax,
# add the value, tile (only the end row is read downstream: post-
# final-attention doctrine, siphon_geo precedent). All existing ops.
# Returns (y, Pr): y the tiled value add (S,H), Pr the (1,K) routing
# (caller slices install mass by baked indices: layout-specific).
# Entity rows are found by the bank's own vocabulary, not a generic
# direction (a mean-difference "lexicality" direction FAILED live:
# min-entity-cos -0.001 vs max-filler-cos 0.194 -- content is not
# linearly separable from fillers; the assert refused fiat
# addressing). Entity-ness = summed install-key scores (ADD of baked
# columns -- bank layout [bg x6, I(6), A(7), C(8)] assumed, stated).
DEF entity_route(xe, xh, Uh, Ue, V, ones) -> (y, Pr)
  CeF = MATMUL(xe, Ue)
  eI = SLICE(CeF, 1, 6, 7)
  eJ = SLICE(CeF, 1, 7, 8)
  eK = SLICE(CeF, 1, 8, 9)
  eIJ = ADD(eI, eJ)
  eIJK = ADD(eIJ, eK)
  II = ARGMAX(eIJK, 0)
  ChF = MATMUL(xh, Uh)
  CF = ADD(ChF, CeF)
  Cr = GATHER(CF, II)
  Sr = TSHIFT(Cr)
  Pr = SOFTMAX_WIDE(Sr)
  v = MATMUL(Pr, V)
  y = MATMUL(ones, v)
END
