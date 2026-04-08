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

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    normalize,
    project_onto_tangent_plane,
)

finite_vectors = arrays(
    dtype=np.float64,
    shape=3,
    elements=st.floats(min_value=-1e6, max_value=1e6),
)
non_zero_magnitude_vectors = finite_vectors.filter(lambda v: np.linalg.norm(v) >= 1e-12)


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


@st.composite
def perpendicular_vectors(draw):
    random_base = normalize(draw(non_zero_magnitude_vectors))
    n = normalize(draw(non_zero_magnitude_vectors))
    v = np.cross(random_base, n)
    return v, n


class ProjectOntoTangentPlaneTest(unittest.TestCase):
    """Unit tests for the project_onto_tangent_plane function."""

    @given(
        a_vector=non_zero_magnitude_vectors,
        a_scalar=st.floats(
            min_value=-1e3, max_value=1e3, allow_infinity=False, allow_nan=False
        ),
    )
    def test_a_vector_parallel_to_normal(self, a_vector, a_scalar):
        parallel_vector = a_scalar * a_vector
        result = project_onto_tangent_plane(parallel_vector, a_vector)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    @given(perpendicular_vectors())
    def test_a_vector_perpendicular_to_normal(self, perpendicular_vectors):
        a_vector, a_normal = perpendicular_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_array_almost_equal(result, a_vector)

    def test_general_oblique_case(self):
        """An oblique vector loses only its normal component.

        Leaving this here because it's a good readable example, but redundant to
        test_result_orthogonal_to_normal which tests the general case.
        """
        v = np.array([1.0, 1.0, 0.0])
        n = np.array([0.0, 0.0, 1.0])
        result = project_onto_tangent_plane(v, n)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 0.0])

    @given(finite_vectors, finite_vectors)
    def test_result_orthogonal_to_normal(self, v, n):
        """The OG test."""
        assume(np.linalg.norm(n) >= 1e-12)
        result = project_onto_tangent_plane(v, n)
        self.assertAlmostEqual(np.dot(result, normalize(n)), 0.0, places=6)

    @given(finite_vectors)
    def test_zero_input_produces_zero_output(self, n):
        """Projecting zero vector to anything is zero."""
        assume(np.linalg.norm(n) >= 1e-12)
        result = project_onto_tangent_plane(np.zeros(3), n)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    @given(finite_vectors, finite_vectors)
    def test_idempotent(self, v, n):
        """Projecting an already-projected vector is no-op."""
        assume(np.linalg.norm(n) >= 1e-12)
        once = project_onto_tangent_plane(v, n)
        twice = project_onto_tangent_plane(once, n)
        np.testing.assert_array_almost_equal(twice, once)
