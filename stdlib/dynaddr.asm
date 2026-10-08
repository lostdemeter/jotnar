# stdlib/dynaddr.asm — address by content, not by position.
#
# Baked row literals (SLICE with fixed positions) silently misaddress
# anything off-template (siphon_geo country@3, smax binaries, per-
# template miners). This finds the row instead: match a key against
# every row, argmax the row, gather it. Entity anywhere in the window,
# any length, any template -- positions appear nowhere.
# All existing ops (MATMUL + ARGMAX + GATHER); key arrives as IN
# (H,1) assembler data (unit direction x scale, addressing/strength
# knobs as usual). Returns the (1,H) entity row.
DEF dynaddr_match(x, key) -> (xa)
  C = MATMUL(x, key)
  I = ARGMAX(C, 0)
  xa = GATHER(x, I)
END
