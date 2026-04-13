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

# Machine Epsilon for values on order of 1
_FLOAT32_EPS = np.finfo(np.float32).eps  # 1.1920929e-07
_FLOAT32_TOL = (
    10 * _FLOAT32_EPS
)  # give few ULP for slack for accumulated rounding error
_FLOAT32_NORM_EPSILON = np.finfo(np.float32).smallest_normal  # 1.1754944e-38

float32_array = arrays(
    dtype=np.float32,
    shape=3,
    elements=st.floats(min_value=-1e6, max_value=1e6, width=32),
)  # note to self: was renamed from finite_vectors

# Open to normalizable_vectors for naming (for semantic meaning)
# nondegenerate is more math-term
nondegenerate_vectors = float32_array.filter(
    lambda v: np.linalg.norm(v) >= _FLOAT32_NORM_EPSILON
)  # note to self: renamed from non_zero_magnitude_vectors
unit_vectors = nondegenerate_vectors.map(lambda v: normalize(v))


@st.composite
def nondegenerate_orthogonal_vectors(draw):
    # Renamed from perpendicular_vectors
    # Updated to exclude drawing zero vectdor for v because that's always perpendicular to everything
    # Nondegenerate excludes v = [0.0, 0.0, 0.0]
    random_base = normalize(draw(nondegenerate_vectors))
    n = normalize(draw(nondegenerate_vectors))
    v = np.cross(random_base, n)
    assume(np.linalg.norm(v) >= _FLOAT32_NORM_EPSILON)
    # using this would result in orthonormal_vectors
    # separate strategy
    # v = normalize(v)
    return v, n


class NormalizeTest(unittest.TestCase):
    @given(nondegenerate_vectors)
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result * norm, v)

    @given(nondegenerate_vectors)
    def test_idempotent(self, v):
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_array_almost_equal(twice, once)

    # @given(
    #     arrays(
    #         dtype=np.float64,
    #         shape=3,
    #         elements=st.floats(min_value=-1e-13, max_value=1e-13),
    #     )
    # )
    # def test_near_zero_vector_raises(self, v):
    #     with self.assertRaises(ValueError):
    #         normalize(v)

    # The below test replaces the above
    # Trying to generate near zero in hypothesis is a lost cause
    # and also assumes things about dtype, hardware, etc.
    # just testing zero is the right way to go
    def test_zero_vector_raises(self):
        zero_float32 = np.zeros(3, dtype=np.float32)
        zero_float64 = np.zeros(3, dtype=np.float64)
        with self.assertRaises(ValueError):
            normalize(zero_float32)
        with self.assertRaises(ValueError):
            normalize(zero_float64)

    @given(
        epsilon=st.floats(min_value=_FLOAT32_NORM_EPSILON, max_value=1e-2),
        scale=st.floats(min_value=0.01, max_value=0.99),
    )
    def test_custom_epsilon_raises_below_threshold(self, epsilon, scale):
        v = np.array([epsilon * scale, 0.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            normalize(v, epsilon=epsilon)

    @given(nondegenerate_vectors)
    def test_result_has_unit_norm(self, v):
        result = normalize(v)
        np.testing.assert_allclose(
            np.linalg.norm(result), 1.0, atol=_FLOAT32_TOL, rtol=_FLOAT32_TOL
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

    @given(perpendicular_vectors())
    def test_a_vector_perpendicular_to_normal(self, orthogonal_vectors):
        a_vector, a_normal = orthogonal_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_array_almost_equal(result, a_vector)

    @given(a_vector=finite_vectors, a_normal=non_zero_magnitude_vectors)
    def test_result_is_orthogonal_to_normal(self, a_vector, a_normal):
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_array_almost_equal(np.dot(result, normalize(a_normal)), 0.0)

    @given(a_vector=finite_vectors, a_normal=non_zero_magnitude_vectors)
    def test_projection_is_idempotent(self, a_vector, a_normal):
        once = project_onto_tangent_plane(a_vector, a_normal)
        twice = project_onto_tangent_plane(once, a_normal)
        np.testing.assert_array_almost_equal(twice, once)
