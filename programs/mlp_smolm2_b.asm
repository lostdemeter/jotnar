# SmolLM2 L0 MLP, stage B (v1.6 anatomy): MID -> OUT at wide scales.
#
# m 36967 covers DOWN (+-23); MID arrives as triples (scale-free handoff
# from stage A) and H re-bridges losslessly (tiny quantum). Two-scale
# execution, same machinery: the per-block-scales end-state done by hand
# (DEF-header @scale remains the syntax for exactly this).
# Split-key refinement: m_acc 35492 (matmul precision peaks just above
# product max -- 59dB vs 51dB at 36967, measured) with m_cov 36967 (ADD
# coverage for +-23 values). acc tight, cov wide: the reason two keys exist.
CONFIG m_acc 35492
CONFIG m_cov 36967

IN MID
IN H
IN wdown

DOWN = MATMUL(MID, wdown)
OUT = ADD(H, DOWN)
