# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import unittest

import numpy as np

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
    project_onto_tangent_plane,
)


class ProjectOntoTangentPlaneTest(unittest.TestCase):
    """Unit tests for the project_onto_tangent_plane function."""

    def test_parallel_to_normal(self):
        n = normalize(np.array([0.0, 0.0, 1.0]))
        result = project_onto_tangent_plane(n, n)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_perpendicular_to_normal(self):
        v = np.array([1.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, v)

    def test_general_oblique_case(self):
        v = np.array([1.0, 1.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 0.0])

    def test_result_orthogonal_to_normal(self):
        v = np.array([3.0, -2.0, 5.0])
        n = normalize(np.array([1.0, 1.0, 1.0]))
        result = project_onto_tangent_plane(v, n)
        self.assertAlmostEqual(np.dot(result, n), 0.0)

    def test_zero_vector_input(self):
        v = np.array([0.0, 0.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
