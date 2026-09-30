# stdlib/dirstore.asm — directional stores (rank-1 knowledge writes).
#
# A matmul IS a sum over directional stores: Wx = sum_i s_i (v_i.x) u_i.
# Each store is a key-value pair (key v_i matches input by projection,
# value u_i writes out, gain s_i scales). An implant (test_implant.py's
# rank-1 write) is one more store ADDED to the sum -- expressed here as
# structure (a listing) instead of surgery (weight bytes). Same values
# (parity-gated), editable text instead of edited tensors.
# A is folded into u host-side (documented); u is (D,1), v is (1,D).
# GAUGE RULE (gated, revised): the A-split across u/v is free
# mathematically but NOT in fixed-point -- and the binding constraint is
# ENCODE quantum on small vectors, not envelope saturation. All-A-in-u:
# 38.6dB. A/32-split (smaller u): 36.8dB, WORSE -- shrinking u into
# quantum noise cost more than saturation ever did. The fold must balance
# quantum (u,v LARGE) against envelope (intermediates SMALL); when
# ||MID||~U no fold satisfies both and m_acc headroom is the release
# valve (untested, stated). Materialization pays quantum tax per hop
# (same lesson as fusion pricing).
DEF implant_apply(x, u, v) -> (y)
  C = MATMUL(x, u)
  y = MATMUL(C, v)
END
