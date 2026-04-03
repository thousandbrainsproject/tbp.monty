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

from tbp.monty.frameworks.utils.spatial_arithmetics import normalize


class NormalizeTest(unittest.TestCase):
    """Unit tests for the normalize function."""

    def test_unit_vector_along_axis(self):
        result = normalize(np.array([3.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_3d_diagonal_vector(self):
        result = normalize(np.array([1.0, 1.0, 1.0]))
        expected = np.array([1, 1, 1]) / np.sqrt(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_already_unit_vector(self):
        v = np.array([0.0, 1.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_negative_components(self):
        result = normalize(np.array([-3.0, 4.0, 0.0]))
        np.testing.assert_array_almost_equal(result, [-0.6, 0.8, 0.0])

    def test_near_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            normalize(np.array([1e-13, 0.0, 0.0]))

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            normalize(np.array([0.0, 0.0, 0.0]))

    def test_custom_epsilon(self):
        with self.assertRaises(ValueError):
            normalize(np.array([1e-8, 0.0, 0.0]), epsilson=1e-6)

    def test_result_has_unit_norm(self):
        v = np.array([2.5, -7.1, 3.3])
        result = normalize(v)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0)


if __name__ == "__main__":
    unittest.main()
