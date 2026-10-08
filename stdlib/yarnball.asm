# stdlib/yarnball.asm — the yarn ball as listing data.
#
# A yarn ball is a factored weight object plus its access discipline:
#   W = U·diag(s)·Vt            (strands: Wx = Σ sᵢ(uᵢ·x)vᵢ, exact identity)
#   address × content × dose     (the siphon product shape)
#
# Math: each store is a (key, value, gain) triple. Keys match the input
# by projection (WHERE, receiver geometry, lexical-early stream); values
# write out (WHAT, native-grown late directions); gains scale (HOW MUCH,
# folded host-side into V rows — dose discipline). Tiers are part of the
# type (exact / assoc / operating-point): assoc stores hold ideas exactly,
# piece-stores take operating-point refits — never mixed silently.
#
# Structure: soft-gated bank application over existing ops ONLY
# (MATMUL + TSHIFT + SOFTMAX_WIDE — the lm_bankhn bank pattern, no new
# mnemonics, no comparison op needed: softmax IS the gate). Single-store
# (K=1) reduces to implant_apply semantics (BP==1, y==value row).
# Hard gating (SELECT on a host-provided mask) stays available for
# cross-stream use where the address lives on a different stream than
# the content; this DEF is the common soft-gated path.
#
# Data discipline (assembler-side, like kernels and tiled biases):
# banks arrive as IN streams built host-side via chain/engram.py
# yarnball_bank() — SVD-decompose, tier-tag, predictor-cost, emit
# (Ua, Vc, ledger). Gains live folded in Vc rows (retune = rebuild,
# stated). Pruning = rebuilding with fewer columns (assembler-side
# edit, cost previewed via chain/read.py before emitting). No
# in-lattice SVD, ever.
#
# Ledger: every store ships a row (triple, support, key-geometry, sha);
# a bank without ledger rows is anonymous yarn, refused by convention.
DEF yarnball_apply(xa, Ua, Vc) -> (y)
  C = MATMUL(xa, Ua)
  S = TSHIFT(C)
  P = SOFTMAX_WIDE(S)
  y = MATMUL(P, Vc)
END
