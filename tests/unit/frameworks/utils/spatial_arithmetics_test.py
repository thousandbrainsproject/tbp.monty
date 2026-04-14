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
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)

# TODO: consolidate the below into a single constant (the tol one)
# name suggestions: TOL or TOLERANCE? (to 1e-6)

# Machine Epsilon for values on order of 1
# Actually only used to really define FLOAT32_TOL, can be removed
_FLOAT32_EPS = np.finfo(np.float32).eps  # 1.1920929e-07
_FLOAT32_TOL = (
    10 * _FLOAT32_EPS
)  # give few ULP for slack for accumulated rounding error

# only used in one place, doesn't need to be used
_FLOAT32_NORM_EPSILON = np.finfo(np.float32).smallest_normal  # 1.1754944e-38


float32_array = arrays(
    dtype=np.float32,
    shape=3,
    elements=st.floats(min_value=-1e6, max_value=1e6, width=32),
)
# Open to normalizable_vectors for naming (for semantic meaning)
# nondegenerate is more math-term
nondegenerate_vectors = float32_array.filter(
    lambda v: np.linalg.norm(v) >= _FLOAT32_EPS
)
unit_vectors = nondegenerate_vectors.map(lambda v: normalize(v))


@st.composite
def nondegenerate_orthogonal_vectors(draw):
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
        atol = max(_FLOAT32_TOL * norm, _FLOAT32_TOL)
        np.testing.assert_allclose(result * norm, v, atol=atol, rtol=_FLOAT32_TOL)

    @given(nondegenerate_vectors)
    def test_idempotent(self, v):
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_allclose(twice, once, atol=_FLOAT32_TOL, rtol=_FLOAT32_TOL)

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
        a_vector=nondegenerate_vectors,
        a_scalar=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
    )
    def test_a_vector_parallel_to_normal(self, a_vector, a_scalar):
        parallel_vector = a_scalar * a_vector
        result = project_onto_tangent_plane(parallel_vector, a_vector)
        atol = max(
            _FLOAT32_TOL * np.linalg.norm(parallel_vector) * np.linalg.norm(a_vector),
            _FLOAT32_TOL,
        )
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=atol, rtol=0)

    @given(nondegenerate_orthogonal_vectors())
    def test_a_vector_perpendicular_to_normal(self, orthogonal_vectors):
        a_vector, a_normal = orthogonal_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        atol = max(_FLOAT32_TOL * np.linalg.norm(a_vector), _FLOAT32_TOL)
        np.testing.assert_allclose(result, a_vector, atol=atol, rtol=_FLOAT32_TOL)

    @given(a_vector=float32_array, a_normal=nondegenerate_vectors)
    def test_result_is_orthogonal_to_normal(self, a_vector, a_normal):
        result = project_onto_tangent_plane(a_vector, a_normal)
        atol = max(
            _FLOAT32_TOL * np.linalg.norm(a_vector),
            _FLOAT32_TOL,
        )
        np.testing.assert_allclose(
            np.dot(result, normalize(a_normal)), 0.0, atol=atol, rtol=0
        )

    @given(a_vector=float32_array, a_normal=nondegenerate_vectors)
    def test_projection_is_idempotent(self, a_vector, a_normal):
        once = project_onto_tangent_plane(a_vector, a_normal)
        twice = project_onto_tangent_plane(once, a_normal)
        atol = max(
            _FLOAT32_TOL * np.linalg.norm(a_vector), _FLOAT32_TOL
        )
        np.testing.assert_allclose(twice, once, atol=atol, rtol=_FLOAT32_TOL)


class TangentFrameTest(unittest.TestCase):
    def _assert_orthonormal_frame(self, frame, normal, tol=_FLOAT32_TOL):
        """Assert (basis_u, basis_v, normal) form an orthonormal right-handed frame."""
        u, v = frame.basis_u, frame.basis_v
        # Check unit norm
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=tol, rtol=tol)

        # Check orthogonality
        np.testing.assert_allclose(np.dot(u, v), 0.0, atol=tol, rtol=0)
        np.testing.assert_allclose(np.dot(u, normal), 0.0, atol=tol, rtol=0)
        np.testing.assert_allclose(np.dot(v, normal), 0.0, atol=tol, rtol=0)

        # Check right-handedness
        np.testing.assert_allclose(np.cross(u, v), normal, atol=tol, rtol=tol)
        np.testing.assert_allclose(np.cross(v, u), -normal, atol=tol, rtol=tol)

    def test_construction_with_y_aligned_normal_triggers_fallback(self):
        n = normalize(np.array([0.0, 1.0, 0.01], dtype=np.float32))
        frame = TangentFrame(n)
        self._assert_orthonormal_frame(frame, n)

    def test_transport_90_degrees_produces_expected_basis(self):
        n1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        expected_u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        expected_v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        expected_u_transported = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        expected_v_transported = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        frame = TangentFrame(n1)
        np.testing.assert_array_almost_equal(frame.basis_u, expected_u)
        np.testing.assert_array_almost_equal(frame.basis_v, expected_v)

        frame.transport(n2)
        np.testing.assert_array_almost_equal(frame.basis_u, expected_u_transported)
        np.testing.assert_array_almost_equal(frame.basis_v, expected_v_transported)

    def test_accumulated_transports_stay_orthonormal(self):
        rng = np.random.RandomState(42)
        n = normalize(rng.randn(3))
        frame = TangentFrame(n)
        for _ in range(100):
            n_new = normalize(n + 0.05 * rng.randn(3))
            frame.transport(n_new)
            n = n_new
        self._assert_orthonormal_frame(frame, n, tol=_FLOAT32_TOL)

    @given(n=unit_vectors)
    def test_transport_to_same_normal_is_noop(self, n):
        frame = TangentFrame(n)
        u_before, v_before = frame.basis_u.copy(), frame.basis_v.copy()
        frame.transport(n)
        np.testing.assert_allclose(frame.basis_u, u_before, atol=_FLOAT32_TOL, rtol=_FLOAT32_TOL)
        np.testing.assert_allclose(frame.basis_v, v_before, atol=_FLOAT32_TOL, rtol=_FLOAT32_TOL)

    @given(n1=unit_vectors, n2=unit_vectors)
    def test_transport_preserves_orthonormality(self, n1, n2):
        frame = TangentFrame(n1)
        frame.transport(n2)
        self._assert_orthonormal_frame(frame, n2)

    @given(n=unit_vectors)
    def test_transport_anti_parallel(self, n):
        frame = TangentFrame(n)
        anti_n = -n
        frame.transport(anti_n)
        u, v = frame.basis_u, frame.basis_v
        # Consider using is_orthornomal?
        np.testing.assert_allclose(np.dot(u, anti_n), 0.0, atol=_FLOAT32_TOL, rtol=0)
        np.testing.assert_allclose(np.dot(v, anti_n), 0.0, atol=_FLOAT32_TOL, rtol=0)
