# BPE single merge splice (one exact step, fixed geometry).
#
# Driver pattern (SCAN precedent): host finds first-applicable pair and
# builds gmap/imask; the listing executes the mechanical splice exactly:
# GATHER by map (with sentinel row L appended for pad tail), SELECT the
# out-id at the splice position. Symbol ids ride as exact triples;
# shapes fixed (L in, L out, sentinel-padded). Full encode = driver loop
# over this step (composition proof in test_bpe_asm.py).
# Contract: gmap ids < L+1 (bounds-checked); imask (L,1) 0/1.
IN syms
IN gmap
IN imask
IN outv

G = GATHER(syms, gmap)
OUT = SELECT(imask, outv, G)
