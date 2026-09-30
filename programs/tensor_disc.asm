# Tensor discriminant, as assembly (estimator LIMIT demo, not catch demo).
#
# The computation that once underflowed m_cov's fixed floor (measured 9e-6
# with coh 0.49 -> lost to 0; fixed by summing at m_acc): hull intervals
# CANNOT see this bug class (hull [-0.13,0.13] spans the floor while real
# values sat at 9e-6). Precision loss inside a spanning range needs
# typical-magnitude reasoning, which hulls don't do -- that class stays
# with unit gates + the sum-at-m_acc rule. This file pins the boundary:
# if estimate() ever flags it, the estimator got smarter (update the gate).
# (MUL-for-squares + ADD chosen over tmul-named ops deliberately: the
# estimator reasons about listing mnemonics, and MUL/ADD are the listing
# level's exact-multiply/add.)

IN jxx
IN jyy
IN jxy
RANGE jxx 0.0 0.25
RANGE jyy 0.0 0.25
RANGE jxy -0.13 0.13

D = SUB(jxx, jyy)
D2 = MUL(D, D)
P = MUL(jxy, jxy)
T4 = MUL(P, 4.0)
DISC = ADD(D2, T4)
