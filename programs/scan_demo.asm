# Scan demo (v1.6 anatomy): SSM recurrence as a listing with STATE.
#
# h' = Abar*h + Bx threaded across steps: h is IN-seeded STATE (carries),
# ab/bx arrive per-step as LISTS (repeat discipline). Proves the handoff's
# claim that scan_step is the grown-up form of STATE+repeat: order-sensitive
# recurrence with per-step inputs, no Python loop in the program.
# Proof: test_asm.py demands repeat() == manual loop, bit-exact.
IN h
IN ab
IN bx
STATE h
h = SCAN(h, ab, bx)
