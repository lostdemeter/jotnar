# Holo flagship v5, as assembly (stage-1 fidelity proof).
#
# The same computation demo.py runs for --blur splat_soft --ctrl v5, with
# NOTHING else: every line names a structure, every structure is gated.
# Config carries the frozen choices (beta value, file-backed scales/levels).
# Proof: test_asm.py runs this listing and demands bit-exact output vs the
# hand-written chain path.
CONFIG beta 0.5

IN rgb AS U8:HWC

LIN = SRGB_DECODE(rgb)
Y = LUMA(LIN)
A = SQRT(Y)
AS, COH = SPLAT_BLUR(A)
D = SUB(A, AS)
BEFF = BETA_V5(D, COH)
BD = MUL(D, BEFF)
AE = ADD(A, BD)
YENH = SQUARE(AE)
G = GAIN(LIN, Y, YENH)
OUT = SRGB_ENCODE(G)
