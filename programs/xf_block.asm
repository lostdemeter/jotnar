# Transformer encoder block (Qwen-style), as assembly -- v1.0 Gate 1 program.
#
# History: written as a GAP PROBE (assembler failures enumerated missing
# mnemonics, recorded in docs/GAPS.md). All gaps closed (Batch 1+2 + Pile A);
# the listing now assembles AND runs. Whole-block parity vs an independent
# torch.float64 reference is gated in test_xf_block.py (84dB in-contract).
# Contract (stated, enforced by the gate, not by the assembler): attention
# scores must stay <=1.0 abs (softmax T-transformation doctrine -- to_fixed
# saturates above 1.0 at BIAS; full-range attention needs the T-transform
# path, backlog) and matmul products inside m_acc coverage (frozen holo
# scales, chain/M.json). The gate's measured row pins the out-of-contract
# behavior; recalibrated scales + 1/sqrt(d) scaling are backlog.
#
# v1.0 Gate 3: dogfooded procedures -- the attention + MLP bodies live in
# stdlib/ (attention.asm, mlp.asm) and are shared via IMPORT (no
# copy-pasted prologues). Bare names resolve stdlib-first (Gate 4).
CONFIG heads 8

IMPORT "attention.asm"
IMPORT "mlp.asm"

IN x AS T:SEQ
IN pos AS I:SEQ
IN wq AS T:SEQ
IN wk
IN wv
IN wo
IN wup
IN wgate
IN wdown
IN rms_w1
IN rms_w2

# --- attention path ---
XN = RMSNORM(x, rms_w1)
O = CALL attn_core(XN, wq, wk, wv, wo, pos)
H = ADD(x, O)

# --- MLP path (SwiGLU) ---
HN = RMSNORM(H, rms_w2)
DOWN = CALL swiglu_block(HN, wup, wgate, wdown)
OUT = ADD(H, DOWN)
