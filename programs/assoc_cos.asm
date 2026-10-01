# Cosine retrieval (direction-first assoc memory).
#
# assoc_mem with RMSNorm equalization: cue/keys normalized per-row
# (direction preserved, norms equalized -- dot == scaled cosine),
# then MATMUL + ARGMAX. Dot-ARGMAX on varying-norm keys is norm-biased
# (measured: hidden-space recall 2/16 dot vs 14/16 cosine); this listing
# is the structural fix. No new mnemonics.
# Contract: keys (K,D) rows nonzero (RMSNorm needs RMS > 0).
CONFIG eps_rms 1e-6
IN cue
IN keys
IN normw

CN = RMSNORM(cue, normw)
KN = RMSNORM(keys, normw)
KT = TRANSPOSE(KN)
SIM = MATMUL(CN, KT)
OUT = ARGMAX(SIM, 1)
