# Flagship iso variant (v1.3 modification demo).
#
# Same pipeline as holo_flagship.asm with the oriented bank + learned gate
# replaced by ISO_BLUR + scalar BETA: the minimal structural edit (two
# lines). Proves listings are MODIFIABLE, not just runnable -- the variant
# executes and matches the hand-written iso chain bit-exactly (gated in
# test_modify.py). The probe table predicts the direction: AS-zero was the
# max-boost row, so a wider iso blur should restrain less (brighter boost)
# than the oriented bank -- measured, not claimed, in the gate.
CONFIG beta 0.5

IN rgb AS U8:HWC

LIN = SRGB_DECODE(rgb)
Y = LUMA(LIN)
A = SQRT(Y)
AS = ISO_BLUR(A)
D = SUB(A, AS)
B = BETA(D)
BD = MUL(D, B)
AE = ADD(A, BD)
YENH = SQUARE(AE)
G = GAIN(LIN, Y, YENH)
OUT = SRGB_ENCODE(G)
