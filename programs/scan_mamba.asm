# Scan demo, mamba magnitudes (v1.6 anatomy-mamba): same recurrence as
# scan_demo.asm with m_cov covering selective-scan states (h/Bx to ~550;
# abar stays dimensionless at BIAS per contract). Proves scale-threading
# reaches recurrence: identical structure, wider envelope (goldilocks:
# cover tightly, m_of(max) + standard margin, never more).
CONFIG m_cov 40086

IN h
IN ab
IN bx
STATE h
h = SCAN(h, ab, bx)
