# stdlib/storebank.asm — whole banks as listing data.
#
# A storebank is K directional stores sharing one apply: C = X@Ub (one
# coefficient per store), y = C@Vb (coefficient-weighted value sum).
# implant_apply is the K=1 special case; this DEF names the K>=1 use so
# listings read as what they are (bank application, not "an implant").
# Banks arrive as IN streams (Ub with gains folded, Vb plain) built
# host-side via chain/engram.py bank() -- assembler-side data, like
# kernels and tiled biases. Pruning = rebuilding with fewer columns
# (assembler-side edit, predicted cost via chain/read.py); listing-text
# pruning (masked sums) is v1.4 horizon, stated here not hidden.
DEF storebank_apply(x, U, V) -> (y)
  C = MATMUL(x, U)
  y = MATMUL(C, V)
END
