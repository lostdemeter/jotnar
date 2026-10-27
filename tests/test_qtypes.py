"""Scale-typed builder and ordered composite gates (research prototype)."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.environ.get("PHI_CORE_DIR", os.path.join(os.path.dirname(ROOT), "phi-core")))
sys.path.insert(0, ROOT)

import phi_core.lattice as S
from chain.qtypes import QProg, QScale, ScaleError
from chain.composite import Layer, ordered_composite, sum_composite


def enc(a):
    return S.encode(np.asarray(a, dtype=np.float64))


def dec(t):
    return S.decode(np.ascontiguousarray(t[0]),
                    np.ascontiguousarray(t[1])) * (
        1 - np.ascontiguousarray(t[2]).astype(np.float64))


def feeds(**kw):
    return {k: enc(v) for k, v in kw.items()}


class TestQTypes(unittest.TestCase):
    def test_identity_at_explicit_scale(self):
        qp = QProg("identity", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        a = qp.inp("A", s)
        b = qp.inp("B", s)
        qp.op("ADD", a, b, out="D")
        got = qp.run(feeds(A=[2.0], B=[0.0]))["D"]
        np.testing.assert_allclose(dec(got), [2.0], atol=1e-2)

    def test_scale_mismatch_fails_loud(self):
        qp = QProg("bad", m_acc=36118, m_cov=35048)
        a = qp.inp("A", QScale(35048, 36118, "value", "a"))
        with self.assertRaises(ScaleError) as cm:
            qp.inp("B", QScale(35049, 36118, "value", "b"))
        msg = str(cm.exception)
        self.assertIn("conflicts", msg)
        self.assertIn("35048", msg)
        self.assertIn("35049", msg)

    def test_checked_failure_shows_context(self):
        qp = QProg("checked", m_acc=36118, m_cov=35048)
        a = qp.inp("A", provenance="test.a")
        qp.op("ADD", a, a, out="D")
        with self.assertRaises(ScaleError) as cm:
            qp.run_checked(feeds(A=[2.0]), expect={"D": 5.0})
        msg = str(cm.exception)
        for token in ("line", "ADD", "D", "decoded", "expected",
                      "output_scale", "m=35048"):
            self.assertIn(token, msg)
        qp.run_checked(feeds(A=[2.0]), expect={"D": 4.0}, atol=1e-2)

    def test_missing_scale_fails_loud(self):
        qp = QProg("missing")
        with self.assertRaises(ScaleError) as cm:
            qp.inp("A")
        self.assertIn("no scale context", str(cm.exception))

    def test_ordered_composite_exact_binary(self):
        qp = QProg("ordered", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        one = qp.inp("ONE", s, provenance="constant.one")
        zero = qp.inp("ZERO", s, provenance="constant.zero")
        n = qp.inp("NEAR", s, provenance="near.value")
        f = qp.inp("FAR", s, provenance="far.value")
        a1 = qp.inp("A1", s, provenance="near.alpha")
        a2 = qp.inp("A2", s, provenance="far.alpha")
        bg = qp.inp("BG", s, provenance="background.value")
        layers = [
            Layer(f, a2, depth=2.0),
            Layer(n, a1, depth=1.0),  # deliberately unordered
        ]
        out = ordered_composite(qp, layers, zero, one, bg, out="D")
        got = qp.run(feeds(ONE=[1, 1, 1, 1], ZERO=[0, 0, 0, 0],
                           NEAR=[1, 1, 0, 0], FAR=[0, 0, 2, 0],
                           A1=[1, 1, 0, 0], A2=[0, 0, 1, 0],
                           BG=[2, 2, 2, 2]))[out.name]
        # near wins where A1=1; far next; background elsewhere
        np.testing.assert_allclose(dec(got), [1, 1, 2, 2], atol=1e-2)

    def test_sum_is_separate(self):
        qp = QProg("sum", m_acc=36118, m_cov=35048)
        s = QScale(35048, 36118, "value", "test")
        one = qp.inp("ONE", s)
        zero = qp.inp("ZERO", s)
        n = qp.inp("NEAR", s)
        f = qp.inp("FAR", s)
        a1 = qp.inp("A1", s)
        a2 = qp.inp("A2", s)
        bg = qp.inp("BG", s)
        out = sum_composite(qp, [Layer(n, a1, 1), Layer(f, a2, 2)],
                            zero, one, bg, out="D")
        got = qp.run(feeds(ONE=[1, 1, 1, 1], ZERO=[0, 0, 0, 0],
                           NEAR=[1, 1, 0, 0], FAR=[0, 0, 2, 0],
                           A1=[1, 1, 0, 0], A2=[0, 0, 1, 0],
                           BG=[2, 2, 2, 2]))[out.name]
        # weighted mean: (near*1 + far*2 + bg) / (1 + 1 + 1)
        np.testing.assert_allclose(dec(got), [1.5, 1.5, 2.0, 2.0],
                                   atol=1e-2)


if __name__ == "__main__":
    unittest.main()
