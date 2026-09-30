# stdlib/mlp.asm — shared SwiGLU feedforward block.
#
# Factored out of programs/xf_block.asm (v1.0 Gate 3). Second seed block
# beside attention.asm's attn_core. Contract: none beyond the ops'
# (MATMUL inner dims, SILU any range).
DEF swiglu_block(hn, wup, wgate, wdown) -> (down)
  UP = MATMUL(hn, wup)
  GATE = MATMUL(hn, wgate)
  GS = SILU(GATE)
  MID = MUL(GS, UP)
  down = MATMUL(MID, wdown)
END
