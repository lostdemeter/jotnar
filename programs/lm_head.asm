# LM head (unembedding + greedy pick, per-block scales companion).
#
# Final step after the layer driver: LOGITS = H @ wlog at head-priced
# m (LOG magnitudes ~30 need m_of(30)+margin, far above body scales).
IN h
IN wlog

LOGITS = MATMUL(h, wlog)
OUT = ARGMAX(LOGITS, 1)
