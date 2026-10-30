"""Field/local-refinement authoring gates."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain.fields import (Field, LayerField, composite_fields,
                          local_refine, local_stencil, refine, sample_if)
from chain.qtypes import QProg, QScale


def enc(a):
    return S.encode(np.asarray(a, dtype=np.float64))


def dec(t):
    return S.decode(np.ascontiguousarray(t[0]),
                    np.ascontiguousarray(t[1])) * (
        1 - np.ascontiguousarray(t[2]).astype(np.float64))


def make_field(qp, s, name="V", shape=(3, 3, 1), fill=1.0, pad=1):
    h, w, c = shape
    th, tw = h + 2 * pad, w + 2 * pad
    v = qp.inp(name, s)
    a = qp.inp(name + "A", s)
    rr = qp.inp(name + "R", kind="I")
    cc = qp.inp(name + "C", kind="I")
    f = Field(v, a, rr, cc, shape, provenance="test." + name,
              depth=1.0)
    vals = np.full((th, tw, c), fill, np.float64)
    alpha = np.ones((th, tw, c), np.float64)
    rows, cols = np.mgrid[pad:h+pad, pad:w+pad]
    return f, vals, alpha, rows.ravel(), cols.ravel()


class TestFields(unittest.TestCase):
    def test_scale_mismatch_fails_loud(self):
        q1 = QProg("field-a", m_acc=36118, m_cov=35048)
        q2 = QProg("field-b", m_acc=36118, m_cov=35049)
        v = q1.inp("V", QScale(35048, 36118, "value", "field"))
        a = q2.inp("A", QScale(35049, 36118, "value", "bad"))
        r = q1.inp("R", kind="I")
        c = q1.inp("C", kind="I")
        with self.assertRaises(ValueError):
            Field(v, a, r, c, (1, 1, 1))

    def test_constant_field_residual_is_zero(self):
        qp = QProg("stencil", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, fill=2.0)
        res = local_stencil(qp, f, [(0, 0, 1.0)],
                            out="RES")
        got = qp.run({"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols})[
            res.name]
        self.assertLess(float(np.max(np.abs(dec(got)))), 1e-2)

    def test_gate_zero_is_identity(self):
        qp = QProg("refine0", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, fill=2.0)
        gate = qp.inp("GATE", s)
        out = local_refine(qp, f, [(0, 0, 1.0)], gate, out="O")
        feeds = {"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols,
                 "GATE": enc(np.zeros((3, 3, 1)))}
        got = dec(qp.run(feeds)[out.value.name])
        np.testing.assert_allclose(got, v[1:-1, 1:-1], atol=1e-2)

    def test_gate_one_applies_residual(self):
        qp = QProg("refine1", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, fill=2.0)
        gate = qp.inp("GATE", s)
        out = local_refine(qp, f, [(0, 0, 1.0)], gate, out="O")
        feeds = {"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols,
                 "GATE": enc(np.ones((3, 3, 1)))}
        got = dec(qp.run(feeds)[out.value.name])
        np.testing.assert_allclose(got, v[1:-1, 1:-1], atol=1e-2)

    def test_validity_and_liminality_metadata(self):
        qp = QProg("meta", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s)
        valid = qp.inp("VALID", s)
        bd = qp.inp("BD", s)
        g = Field(f.value, f.alpha, f.row, f.col, f.shape,
                  valid=valid, boundary_distance=bd)
        self.assertIs(g.valid, valid)
        self.assertIs(g.boundary_distance, bd)

    def test_refine_preserves_validity_and_liminality(self):
        qp = QProg("metarefine", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s)
        valid = qp.inp("VALID", s)
        bd = qp.inp("BD", s)
        g = Field(f.value, f.alpha, f.row, f.col, f.shape,
                  valid=valid, boundary_distance=bd)
        gate = qp.inp("GATE", s)
        out = local_refine(qp, g, [(0, 0, 1.0)], gate, out="O")
        self.assertIs(out.valid, valid)
        self.assertIs(out.boundary_distance, bd)

    def test_sample_if_rejects_invalid_sample(self):
        qp = QProg("sampleif", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, shape=(1, 1, 1),
                                           fill=4.0, pad=0)
        valid = qp.inp("VALID", s)
        bd = qp.inp("BD", s)
        src = Field(f.value, f.alpha, f.row, f.col, f.shape,
                    valid=valid, boundary_distance=bd)
        gate = qp.inp("GATE", s)
        out = sample_if(qp, src, f.row, f.col, gate, (1, 1, 1), out="S")
        feeds = {"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols,
                 "VALID": enc(np.ones((1, 1, 1))),
                 "BD": enc(np.full((1, 1, 1), 0.25)),
                 "GATE": enc(np.zeros((1, 1, 1)))}
        got = qp.run(feeds)
        self.assertLess(float(np.max(np.abs(dec(got[out.alpha.name])))), 1e-2)
        self.assertLess(float(np.max(np.abs(dec(got[out.valid.name])))), 1e-2)
        self.assertLess(float(np.max(np.abs(dec(got[out.boundary_distance.name]) - 0.25))), 1e-2)

    def test_sample_if_invalid_source_zeroes_alpha(self):
        qp = QProg("sampleifbad", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, shape=(1, 1, 1),
                                           fill=4.0, pad=0)
        valid = qp.inp("VALID", s)
        src = Field(f.value, f.alpha, f.row, f.col, f.shape, valid=valid)
        gate = qp.inp("GATE", s)
        out = sample_if(qp, src, f.row, f.col, gate, (1, 1, 1), out="S")
        feeds = {"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols,
                 "VALID": enc(np.zeros((1, 1, 1))),
                 "GATE": enc(np.ones((1, 1, 1)))}
        got = qp.run(feeds)
        self.assertLess(float(np.max(np.abs(dec(got[out.alpha.name])))), 1e-2)
        self.assertLess(float(np.max(np.abs(dec(got[out.valid.name])))), 1e-2)

    def test_sample_if_accepts_valid_sample(self):
        qp = QProg("sampleifok", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        f, v, a, rows, cols = make_field(qp, s, shape=(1, 1, 1),
                                           fill=4.0, pad=0)
        valid = qp.inp("VALID", s)
        src = Field(f.value, f.alpha, f.row, f.col, f.shape, valid=valid)
        gate = qp.inp("GATE", s)
        out = sample_if(qp, src, f.row, f.col, gate, (1, 1, 1), out="S")
        feeds = {"V": enc(v), "VA": enc(a), "VR": rows, "VC": cols,
                 "VALID": enc(np.ones((1, 1, 1))),
                 "GATE": enc(np.ones((1, 1, 1)))}
        got = qp.run(feeds)
        np.testing.assert_allclose(dec(got[out.value.name]), [[[4.0]]], atol=1e-2)
        np.testing.assert_allclose(dec(got[out.alpha.name]), [[[1.0]]], atol=1e-2)
        np.testing.assert_allclose(dec(got[out.valid.name]), [[[1.0]]], atol=1e-2)

    def test_ordered_fields(self):
        qp = QProg("fcomp", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        near, nv, na, nr, nc = make_field(qp, s, "N", shape=(1, 1, 1),
                                           fill=1.0, pad=0)
        far, fv, fa, fr, fc = make_field(qp, s, "F", shape=(1, 1, 1),
                                          fill=2.0, pad=0)
        z = qp.inp("Z", s); o = qp.inp("O", s); bg = qp.inp("BG", s)
        out = composite_fields(qp, [LayerField(far, 2.0),
                                    LayerField(near, 1.0)], z, o, bg,
                               out="D")
        got = qp.run({"N": enc(nv), "NA": enc(na), "NR": nr, "NC": nc,
                      "F": enc(fv), "FA": enc(fa), "FR": fr, "FC": fc,
                      "Z": enc([[ [0.0] ]]), "O": enc([[ [1.0] ]]),
                      "BG": enc([[ [3.0] ]])})[out.name]
        np.testing.assert_allclose(dec(got), [[ [1.0] ]], atol=1e-2)


if __name__ == "__main__":
    unittest.main()
