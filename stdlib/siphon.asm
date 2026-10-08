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
DEF entity_route(xe, xh, ekey, Uh, Ue, V, ones) -> (y, Pr)
  CI = MATMUL(xe, ekey)
  II = ARGMAX(CI, 0)
  ChF = MATMUL(xh, Uh)
  CeF = MATMUL(xe, Ue)
  CF = ADD(ChF, CeF)
  Cr = GATHER(CF, II)
  Sr = TSHIFT(Cr)
  Pr = SOFTMAX_WIDE(Sr)
  v = MATMUL(Pr, V)
  y = MATMUL(ones, v)
END
