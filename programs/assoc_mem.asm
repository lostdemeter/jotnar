# Content-addressable memory (v1.5 demo 2: from scratch, no trained weights).
#
# Stores hold patterns as data (keys + values as IN streams, frozen by
# documented seed); retrieval = similarities + argmax + gather, all exact
# integer ops. No trained weights anywhere -- behavior from composition
# alone (recall curves, not parity: there is no reference to match).
# Scales: bipolar similarities over D=64 sum to +-64 (m_acc covers).
# Proof: test_demo2.py demands recall bands from capacity reasoning.
CONFIG m_acc 37705

IN cue
IN keys
IN values

SIM = MATMUL(cue, keys)
IDX = ARGMAX(SIM, 1)
OUT = GATHER(values, IDX)
