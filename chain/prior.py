"""Prior modulation (multiplicative caution from offline priors).

Contract (both forms): an offline prior, converted at the boundary, becomes
exact multiplier triples that compose onto boost via tmul (never additive,
never in the blur -- caution scales existing detail, it doesn't invent or
erase structure). The unaffected subset takes EXACT identity (ones triples:
tmul by encode(1.0) is exp-add of BIAS-BIAS, sign*1 -- exact), so priors
cost nothing where they don't apply. All ops pre-existing (select_mux,
tmul, binop, encode): composition only, no new C needed.

Two forms, one family (like replicate/zero edge semantics -- the split is
gated, not hidden):
  select(mask, full=1.0, atten): bool verdict -> {full, atten} via exact
    select. Use when the prior speaks binary (near/far).
  affine(field, static, atten): 1-(1-atten)*field with exact ones on static.
    Use when the prior speaks continuous (motion norm); the static override
    keeps zero-input bit-identical (roundtrips must never tax static pixels).
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "phi-core")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phi_core.lattice as S


def select(shape, mask, atten, full=1.0, m_cov=None):
    """Bool mask True -> full, False -> atten. Exact select, no arithmetic."""
    from chain.holo_phi import select_mux, _load_scales
    if m_cov is None:
        _, m_cov = _load_scales()
    full_t = S.encode(np.full(shape, float(full), dtype=np.float64))
    att_t = S.encode(np.full(shape, float(atten), dtype=np.float64))
    bucket = np.where(np.ascontiguousarray(mask, dtype=bool), 0, 1).astype(np.int8)
    return select_mux([full_t, att_t], bucket)


def affine(field_trip, static_mask, atten, m_cov):
    """1-(1-atten)*field with exact ones where static. field_trip: triples
    in [0,1]; static_mask: exact bool (boundary verdict)."""
    from chain.holo_phi import tmul, binop_fixed
    shape = field_trip[0].shape
    ka = S.encode(np.full(shape, 1.0 - float(atten), dtype=np.float64))
    one = S.encode(np.ones(shape, dtype=np.float64))
    dec = binop_fixed(one, tmul(ka, field_trip), m_cov, m_cov, op="sub")
    st = np.ascontiguousarray(static_mask, dtype=bool)
    assert st.shape == shape, f"static geometry {st.shape} vs {shape}"
    return (np.where(st, one[0], dec[0]).astype(np.int8),
            np.where(st, one[1], dec[1]).astype(np.int32),
            np.where(st, one[2], dec[2]).astype(np.uint8))
