# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.utils.spatial_arithmetics import normalize

finite_vectors = arrays(
    dtype=np.float64,
    shape=3,
    elements=st.floats(min_value=-1e6, max_value=1e6),
)


class NormalizeTest(unittest.TestCase):
    """Unit tests for the normalize function."""

    @given(finite_vectors)
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        assume(norm >= 1e-12)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result * norm, v)

    @given(finite_vectors)
    def test_idempotent(self, v):
        assume(np.linalg.norm(v) >= 1e-12)
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_array_almost_equal(twice, once)

    @given(
        arrays(
            dtype=np.float64,
            shape=3,
            elements=st.floats(min_value=-1e-13, max_value=1e-13),
        )
    )
    def test_near_zero_vector_raises(self, v):
        with self.assertRaises(ValueError):
            normalize(v)

    @given(
        epsilon=st.floats(min_value=1e-12, max_value=1e-2),
        scale=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_custom_epsilon(self, epsilon, scale):
        v = np.array([epsilon * scale, 0.0, 0.0])
        with self.assertRaises(ValueError):
            normalize(v, epsilon=epsilon)

    @given(finite_vectors)
    def test_result_has_unit_norm(self, v):
        assume(np.linalg.norm(v) >= 1e-12)
        result = normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)
