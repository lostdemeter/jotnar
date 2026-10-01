# Cue projection (hidden -> cue space, frozen P).
#
# One MATMUL: H (S,16) @ P (16,64) -> CUE (S,64). Sign + recall stay
# host-boundary (no SIGN mnemonic -- stated); magnitudes priced: H~3 *
# 0.25 x16 taps ~= 12 < m_of(8)+margin regime (CONFIG m_acc 35492 path).
IN h
IN proj

OUT = MATMUL(h, proj)
