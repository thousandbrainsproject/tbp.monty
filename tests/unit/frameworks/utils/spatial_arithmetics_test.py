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

from tbp.monty.frameworks.utils.spatial_arithmetics import (
    TangentFrame,
    normalize,
    project_onto_tangent_plane,
)

finite_vectors = arrays(
    dtype=np.float32,
    shape=3,
    elements=st.floats(min_value=-100, max_value=100, width=32, allow_subnormal=False),
)
non_zero_magnitude_vectors = finite_vectors.filter(lambda v: np.linalg.norm(v) >= 1e-6)
unit_vectors = non_zero_magnitude_vectors.map(lambda v: normalize(v))


@st.composite
def perpendicular_vectors(draw):
    random_base = normalize(draw(non_zero_magnitude_vectors))
    n = normalize(draw(non_zero_magnitude_vectors))
    v = np.cross(random_base, n)
    return v, n


class NormalizeTest(unittest.TestCase):
    @given(non_zero_magnitude_vectors)
    def test_preserves_direction(self, v):
        norm = np.linalg.norm(v)
        result = normalize(v)
        np.testing.assert_allclose(result * norm, v, atol=1e-5, rtol=1e-5)

    @given(non_zero_magnitude_vectors)
    def test_idempotent(self, v):
        once = normalize(v)
        twice = normalize(once)
        np.testing.assert_allclose(twice, once, atol=1e-5, rtol=1e-5)

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

    @given(non_zero_magnitude_vectors)
    def test_result_has_unit_norm(self, v):
        result = normalize(v)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-5, rtol=1e-5)


class ProjectOntoTangentPlaneTest(unittest.TestCase):
    @given(
        a_vector=non_zero_magnitude_vectors,
        a_scalar=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False).filter(
            lambda x: abs(x) > 1e-6
        ),
    )
    def test_a_vector_parallel_to_normal(self, a_vector, a_scalar):
        parallel_vector = a_scalar * a_vector
        result = project_onto_tangent_plane(parallel_vector, a_vector)
        atol = max(1e-5 * np.linalg.norm(parallel_vector), 1e-6)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=atol, rtol=0)

    @given(perpendicular_vectors())
    def test_a_vector_perpendicular_to_normal(self, orthogonal_vectors):
        a_vector, a_normal = orthogonal_vectors
        result = project_onto_tangent_plane(a_vector, a_normal)
        np.testing.assert_allclose(result, a_vector, atol=1e-5, rtol=1e-5)

    @given(a_vector=finite_vectors, a_normal=non_zero_magnitude_vectors)
    def test_result_is_orthogonal_to_normal(self, a_vector, a_normal):
        result = project_onto_tangent_plane(a_vector, a_normal)
        atol = max(1e-5 * np.linalg.norm(a_vector), 1e-6)
        np.testing.assert_allclose(
            np.dot(result, normalize(a_normal)), 0.0, atol=atol, rtol=0
        )

    @given(a_vector=finite_vectors, a_normal=non_zero_magnitude_vectors)
    def test_projection_is_idempotent(self, a_vector, a_normal):
        once = project_onto_tangent_plane(a_vector, a_normal)
        twice = project_onto_tangent_plane(once, a_normal)
        # Increased rtol since error accumulates
        np.testing.assert_allclose(twice, once, atol=1e-5, rtol=1e-3)


class TangentFrameTest(unittest.TestCase):
    def _assert_orthonormal_frame(self, frame, normal):
        """Assert (basis_u, basis_v, normal) form an orthonormal right-handed frame."""
        u, v = frame.basis_u, frame.basis_v
        # Check unit norm
        np.testing.assert_array_almost_equal(
            np.linalg.norm(u), 1.0, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_array_almost_equal(
            np.linalg.norm(v), 1.0, atol=1e-5, rtol=1e-5
        )
        np.testing.assert_array_almost_equal(
            np.linalg.norm(normal), 1.0, atol=1e-5, rtol=1e-5
        )

        # Check orthogonality
        np.testing.assert_array_almost_equal(np.dot(u, v), 0.0, atol=1e-5, rtol=0)
        np.testing.assert_array_almost_equal(np.dot(u, normal), 0.0, atol=1e-5, rtol=0)
        np.testing.assert_array_almost_equal(np.dot(v, normal), 0.0, atol=1e-5, rtol=0)

        # Check right-handedness
        np.testing.assert_array_almost_equal(np.cross(u, v), normal)
        np.testing.assert_array_almost_equal(np.cross(v, u), -normal)

    def test_construction_with_y_aligned_normal_triggers_fallback(self):
        n = normalize(np.array([0.0, 1.0, 0.01]))
        frame = TangentFrame(n)
        self._assert_orthonormal_frame(frame, n)

    @given(n=unit_vectors)
    def test_transport_to_same_normal_is_noop(self, n):
        frame = TangentFrame(n)
        u_before, v_before = frame.basis_u.copy(), frame.basis_v.copy()
        frame.transport(n)
        np.testing.assert_array_almost_equal(frame.basis_u, u_before)
        np.testing.assert_array_almost_equal(frame.basis_v, v_before)

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
        np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(np.dot(u, anti_n), 0.0, atol=1e-5, rtol=0)
        np.testing.assert_allclose(np.dot(v, anti_n), 0.0, atol=1e-5, rtol=0)

    def test_transport_90_degrees_produces_expected_basis(self):
        n1 = np.array([0.0, 0.0, 1.0])
        n2 = np.array([0.0, 1.0, 0.0])
        frame = TangentFrame(n1)

        np.testing.assert_array_almost_equal(frame.basis_u, [1, 0, 0])
        np.testing.assert_array_almost_equal(frame.basis_v, [0, 1, 0])

        frame.transport(n2)

        np.testing.assert_array_almost_equal(frame.basis_u, [1, 0, 0])
        np.testing.assert_array_almost_equal(frame.basis_v, [0, 0, -1])

    def test_accumulated_transports_stay_orthonormal(self):
        rng = np.random.RandomState(42)
        n = normalize(rng.randn(3))
        frame = TangentFrame(n)
        for _ in range(100):
            n_new = normalize(n + 0.05 * rng.randn(3))
            frame.transport(n_new)
            n = n_new
        self._assert_orthonormal_frame(frame, n)
