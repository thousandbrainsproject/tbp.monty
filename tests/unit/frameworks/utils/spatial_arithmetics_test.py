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
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tbp.monty.frameworks.utils.spatial_arithmetics import normalize


class NormalizeTest(unittest.TestCase):
    """Unit tests for the normalize function."""

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(min_value=-1e6, max_value=1e6),
        )
    )
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return
        result = normalize(v)
        np.testing.assert_array_almost_equal(result * norm, v)

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(min_value=-1e6, max_value=1e6),
        )
    )
    def test_idempotent(self, v):
        if np.linalg.norm(v) < 1e-12:
            return
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_array_almost_equal(twice, once)

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
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

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=10),
            elements=st.floats(min_value=-1e6, max_value=1e6),
        )
    )
    def test_result_has_unit_norm(self, v):
        if np.linalg.norm(v) < 1e-12:
            return
        result = normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)


if __name__ == "__main__":
    unittest.main()
