# Motion-gated temporal stabilizer (stage-2 GENERATIVE proof).
#
# Designed on paper BEFORE running. Paper claim as written: "denoiser".
# MEASURED: not a denoiser -- A + beta*D keeps the noisy base, so no boost
# architecture can denoise (output-domain averaging or As-output would be
# needed; both are backlog structures, explicitly NOT claimed here). What it
# IS: a stabilizer -- memory steadies the boost on static content (flicker
# down 26% measured) while moving pixels trust the current frame bit-clean.
# Renamed denoise->stabilize rather than tuning the fixture: falsification
# at full resolution is the paper-first discipline working as designed.
# Novel combination -- this listing exists nowhere in the repos; every
# mnemonic names a gated structure, four of them new here (ISO_BLUR, WARP,
# STATIC, MIXDYAD), all thin wrappers over existing chain functions.
#
# Feed conventions (stated, part of the design): dprev=None on frame 0
# (no history); WARP passes None through; MIXDYAD maps None -> D directly
# (first-frame identity without a language conditional). Driver threads
# feeds["D"] forward as next frame's dprev.
CONFIG beta 0.5

IN rgb AS U8:HWC
IN dprev
IN flow AS F:HW2
STATE dprev

LIN = SRGB_DECODE(rgb)
Y = LUMA(LIN)
A = SQRT(Y)
AS = ISO_BLUR(A)
W = WARP(dprev, flow)
S = STATIC(flow)
dprev = SUB(A, AS)
DM = MIXDYAD(dprev, W, S)
B = BETA(dprev)
BD = MUL(DM, B)
AE = ADD(A, BD)
YENH = SQUARE(AE)
G = GAIN(LIN, Y, YENH)
OUT = SRGB_ENCODE(G)
