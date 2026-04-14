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
from tbp.monty.math import DEFAULT_TOLERANCE

vectors_3d = arrays(
    dtype=np.float32,
    shape=3,
    elements=st.floats(min_value=-1e6, max_value=1e6, width=32),
)
non_zero_magnitude_vectors = vectors_3d.filter(
    lambda v: np.linalg.norm(v) > DEFAULT_TOLERANCE
)


@st.composite
def nonzero_orthogonal_vectors(draw):
    random_base = normalize(draw(non_zero_magnitude_vectors))
    n = normalize(draw(non_zero_magnitude_vectors))
    v = np.cross(random_base, n)
    assume(np.linalg.norm(v) > DEFAULT_TOLERANCE)
    return v, n


class NormalizeTest(unittest.TestCase):
    @given(non_zero_magnitude_vectors)
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        result = normalize(v)
        tol = max(DEFAULT_TOLERANCE * norm, DEFAULT_TOLERANCE)
        np.testing.assert_allclose(result * norm, v, atol=tol, rtol=tol)

    @given(non_zero_magnitude_vectors)
    def test_idempotent(self, v):
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_allclose(
            twice, once, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )

    def test_zero_vector_raises(self):
        v = np.zeros(3, dtype=float)
        with self.assertRaises(ValueError):
            normalize(v)

    @given(
        epsilon=st.floats(min_value=DEFAULT_TOLERANCE, max_value=1e-2),
        scale=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_custom_epsilon(self, epsilon, scale):
        v = np.array([epsilon * scale, 0.0, 0.0])
        with self.assertRaises(ValueError):
            normalize(v, epsilon=epsilon)

    @given(non_zero_magnitude_vectors)
    def test_result_has_unit_norm(self, v):
        result = normalize(v)
        np.testing.assert_allclose(
            np.linalg.norm(result), 1.0, atol=DEFAULT_TOLERANCE, rtol=DEFAULT_TOLERANCE
        )


class ProjectOntoTangentPlaneTest(unittest.TestCase):
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

    @given(nonzero_orthogonal_vectors())
    def test_a_vector_perpendicular_to_normal(self, orthogonal_vectors):
        a_vector, a_normal = orthogonal_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_array_almost_equal(result, a_vector)

    @given(a_vector=vectors_3d, a_normal=non_zero_magnitude_vectors)
    def test_result_is_orthogonal_to_normal(self, a_vector, a_normal):
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_array_almost_equal(np.dot(result, normalize(a_normal)), 0.0)

    @given(a_vector=vectors_3d, a_normal=non_zero_magnitude_vectors)
    def test_projection_is_idempotent(self, a_vector, a_normal):
        once = project_onto_tangent_plane(a_vector, a_normal)
        twice = project_onto_tangent_plane(once, a_normal)
        np.testing.assert_array_almost_equal(twice, once)
