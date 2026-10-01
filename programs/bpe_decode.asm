# BPE decode (piece ids -> char-id rows, fixed geometry).
#
# Piece table (V,14) + lengths (V,1) ride frozen as exact triples (small
# ints, bigram-bank precedent); pids (S,) gather rows; host flattens +
# strips by lengths (strings are not a KIND -- boundary honest).
# Contract: pids < V (bounds-checked, fail loud).
IN pids
IN ptab
IN plent

OUT = GATHER(ptab, pids)
LENS = GATHER(plent, pids)
